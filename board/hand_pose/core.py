"""Per-episode ACE-Ego-Hand pipeline: source camera -> virtual pinhole -> VAE -> overlapping windows ->
blend -> knot-rate smoothing -> 2D keypoints in the source clip's own pixels. Runs inside the Modal image of
board/hand_pose/modal_app.py, next to the ACE-Ego-Hand code; it is not imported on your machine.

Why each step: the released checkpoints are pinhole only, so fisheye clips (Egocentric-100K Kannala-Brandt,
Gen-HumanEgo Double Sphere) are resampled to a virtual pinhole with the dataset's own calibration, and the
predictions are mapped back through it. The model predicts at the VAE latent rate (every 4th frame) over
22-latent windows; windows overlap by half and are blended with a triangular weight, so every frame is the
mean of two independent predictions and there are no seams (two windows disagree by about 1 px on an 832 px
frame). The 2D keypoints come from the model's direct 2D head, which does not use MANO and does not read the
camera: MANO and the camera enter only the 3D translation decode. Smoothing is a zero-phase one-euro filter at
the knot rate (min cutoff 1.5 Hz, beta 1 hand size per second): slow hands lose their knot-to-knot wiggle, fast
hands keep their motion; a Savitzky-Golay filter moved fast real motion by a median 5 px and was not used.

Coordinates. Everything 2D here uses continuous pixel coordinates: x in [0, W], the centre of pixel
column j is x = j + 0.5. OpenCV-style calibrations (Egocentric-100K KB, Gen-HumanEgo DS) put pixel
centres at integers, so a calibrated projection u_cv maps to u_cv + 0.5 here. ACE's direct 2D head is
normalised by image size, so pinhole pixel = n * W.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

OP21 = ["wrist",
        "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
         (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


# ----------------------------------------------------------------------------- camera models

def project_src(rays: np.ndarray, cam: dict) -> np.ndarray:
    """Camera-frame rays (..., 3) -> source pixels (..., 2), continuous convention."""
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    m = cam["model"]
    if m == "kb":           # OpenCV fisheye / Kannala-Brandt equidistant
        r = np.sqrt(x * x + y * y)
        th = np.arctan2(r, z)
        t2 = th * th
        td = th * (1 + cam["k1"] * t2 + cam["k2"] * t2 ** 2 + cam["k3"] * t2 ** 3 + cam["k4"] * t2 ** 4)
        s = np.where(r > 1e-12, td / np.maximum(r, 1e-12), 1.0 / np.maximum(z, 1e-12))
        u = cam["fx"] * x * s + cam["cx"]
        v = cam["fy"] * y * s + cam["cy"]
    elif m == "ds":         # Double Sphere (Usenko et al. 2018)
        xi, al = cam["xi"], cam["alpha"]
        d1 = np.sqrt(x * x + y * y + z * z)
        zz = xi * d1 + z
        d2 = np.sqrt(x * x + y * y + zz * zz)
        den = al * d2 + (1 - al) * zz
        u = cam["fx"] * x / den + cam["cx"]
        v = cam["fy"] * y / den + cam["cy"]
    elif m == "pinhole":
        u = cam["fx"] * x / z + cam["cx"]
        v = cam["fy"] * y / z + cam["cy"]
    else:
        raise ValueError(m)
    off = 0.5 if cam.get("convention", "opencv") == "opencv" else 0.0
    return np.stack([u + off, v + off], -1)


def pinhole_rays(uv: np.ndarray, pin: dict) -> np.ndarray:
    """Continuous pinhole pixels (..., 2) -> rays (..., 3) in the SOURCE camera frame. A virtual pinhole may be
    pitched down by pin["pitch_deg"] (rotation about x; positive looks toward +y, i.e. down the image)."""
    x = (uv[..., 0] - pin["cx"]) / pin["fx"]
    y = (uv[..., 1] - pin["cy"]) / pin["fy"]
    r = np.stack([x, y, np.ones_like(x)], -1)
    th = math.radians(pin.get("pitch_deg", 0.0))
    if th:
        c, s = math.cos(th), math.sin(th)
        r = np.stack([r[..., 0], c * r[..., 1] + s * r[..., 2], -s * r[..., 1] + c * r[..., 2]], -1)
    return r


def virtual_pinhole(w: int, h: int, hfov_deg: float, pitch_deg: float = 0.0) -> dict:
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return {"fx": f, "fy": f, "cx": w / 2.0, "cy": h / 2.0, "image_width": w, "image_height": h,
            "hfov_deg": hfov_deg, "vfov_deg": math.degrees(2 * math.atan((h / 2.0) / f)), "pitch_deg": pitch_deg}


def remap_tables(src_cam: dict, pin: dict):
    """cv2.remap tables (OpenCV index convention) sampling the source for every pinhole pixel."""
    W, H = pin["image_width"], pin["image_height"]
    jj, ii = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    uv = project_src(pinhole_rays(np.stack([jj, ii], -1), pin), src_cam) - 0.5  # back to index coords
    return uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)


# ----------------------------------------------------------------------------- video IO

def _display():
    # prepare/display.py, the one rule for how a file is shown; the Modal image carries it beside this file
    try:
        from prepare import display
    except ImportError:
        import display
    return display


def probe(path: str) -> dict:
    """The video as it is shown (prepare/display.py): ffmpeg turns the frames upright, so a phone's portrait video
    is its stored size turned, and pixels that are not square are made square (resample), so the keypoints are in
    the same picture the board's clip shows."""
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets", "-of", "json", path],
                       capture_output=True, text=True, check=True).stdout
    s = json.loads(o)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    d = _display()
    g = d.geometry(path)
    w, h = d.shown_size(g) if g["stored"][0] else (int(s["width"]), int(s["height"]))
    return {"width": w, "height": h, "fps": float(num) / float(den), "n_packets": int(s["nb_read_packets"]),
            "resample": d.needs_resample(g)}


def read_frames(path: str, W: int, H: int, resample: bool = False):
    """Yield RGB uint8 frames W x H as shown, decoded by ffmpeg (every frame, no dup/drop): turned upright as the
    file says, and scaled to square pixels when they are not (resample)."""
    vf = ["-vf", f"scale={W}:{H}:flags=lanczos,setsar=1"] if resample else []
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-vsync", "passthrough", *vf,
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    n = W * H * 3
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
    finally:
        p.stdout.close()
        p.wait()


def load_pinhole_frames(path: str, meta: dict, src_cam: dict | None, pin: dict):
    """All frames resampled to the pinhole grid, uint8 (F, H, W, 3)."""
    import cv2
    W, H = pin["image_width"], pin["image_height"]
    maps = remap_tables(src_cam, pin) if src_cam is not None else None
    out = []
    for f in read_frames(path, meta["width"], meta["height"], meta.get("resample", False)):
        if maps is not None:
            g = cv2.remap(f, maps[0], maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            g = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        out.append(g)
    return np.stack(out)


# ----------------------------------------------------------------------------- model

BLEND_KEYS = ("direct_joints2d", "exists_2d", "exists_3d", "betas", "cam_trans", "direct_joints_cam",
              "direct_wrist_cam", "global_orient", "hand_pose")
ROT_KEYS = ("global_orient", "hand_pose")


def window_anchors(f_lat: int, w: int, stride: int) -> list[int]:
    if f_lat <= w:
        return [0]
    a = list(range(0, f_lat - w + 1, stride))
    if a[-1] != f_lat - w:
        a.append(f_lat - w)
    return a


def run_windows(model, ctrl, intr: dict, w: int, stride: int, tap: int = 15, want_rays: bool = False,
                anchors: list[int] | None = None, log=print):
    """Overlapping-window inference with triangular blending.

    Returns (blended dict over px frames, stats, rays). Each px frame t covered by windows W_i gets
    sum_i w_i(t) x_i(t) / sum_i w_i(t), with w_i peaking at the window centre. For the 2D head the
    per-frame weighted spread across windows is kept (j2d_window_std in raw.npz). rays holds each window's
    ray field when want_rays is set (the K-free checkpoint), else it is empty.
    """
    import torch
    f_lat = ctrl.shape[1]
    n_px = 4 * (f_lat - 1) + 1
    anchors = window_anchors(f_lat, w, stride) if anchors is None else anchors
    acc, wsum = {}, np.zeros(n_px, np.float64)
    sq2d = None
    rays = []
    t0 = time.time()
    for wi, a in enumerate(anchors):
        ww = min(w, f_lat)
        nv = 4 * (ww - 1) + 1
        with torch.no_grad():
            out = model.net.forward_emode(ctrl[:, a:a + ww].unsqueeze(0).float(), nv, [intr])
        p = out["preds"][tap][0]
        if want_rays and out.get("raymap"):
            rays.append(out["raymap"][tap][0].float().cpu())
        tt = np.arange(nv)
        wt = np.minimum(tt + 1, nv - tt).astype(np.float64)       # triangular, >= 1
        lo = 4 * a
        wsum[lo:lo + nv] += wt
        for k in BLEND_KEYS:
            arr = p[k].float().cpu().numpy().astype(np.float64)
            if k not in acc:
                acc[k] = np.zeros((n_px,) + arr.shape[1:], np.float64)
            acc[k][lo:lo + nv] += arr * wt.reshape((-1,) + (1,) * (arr.ndim - 1))
            if k == "direct_joints2d":
                if sq2d is None:
                    sq2d = np.zeros((n_px,) + arr.shape[1:], np.float64)
                sq2d[lo:lo + nv] += (arr ** 2) * wt.reshape((-1,) + (1,) * (arr.ndim - 1))
        if wi == 0 or (wi + 1) % 20 == 0 or wi == len(anchors) - 1:
            log(f"[windows] {wi + 1}/{len(anchors)} anchor={a} {time.time() - t0:.1f}s")
    res = {}
    for k, v in acc.items():
        res[k] = v / wsum.reshape((-1,) + (1,) * (v.ndim - 1))
    var = sq2d / wsum.reshape((-1,) + (1,) * (sq2d.ndim - 1)) - res["direct_joints2d"] ** 2
    res["j2d_window_std"] = np.sqrt(np.clip(var, 0, None)).astype(np.float32)   # normalised units
    nonfinite = {k: int((~np.isfinite(v)).sum()) for k, v in res.items()}
    for k in ROT_KEYS:                                   # project blended matrices back onto SO(3)
        m = res[k]
        bad = ~np.isfinite(m).all(axis=(-1, -2))
        m = np.where(bad[..., None, None], np.eye(3), m)
        U, _, Vt = np.linalg.svd(m)
        d = np.sign(np.linalg.det(U @ Vt))
        U[..., :, -1] *= d[..., None]
        r = U @ Vt
        res[k] = np.where(bad[..., None, None], np.nan, r)
    res["n_windows_covering"] = np.zeros(n_px, np.int16)
    for a in anchors:
        res["n_windows_covering"][4 * a:4 * a + 4 * (min(w, f_lat) - 1) + 1] += 1
    stats = {"anchors": anchors, "window": w, "stride": stride, "sec": time.time() - t0,
             "nonfinite": {k: v for k, v in nonfinite.items() if v}}
    return res, stats, rays


def fit_K_from_rays(rays: list, intr: dict) -> dict:
    """One pinhole for the whole clip from the K-free ray fields of several windows (ACE-Ego-Hand's own
    least-squares fit)."""
    import torch
    from ace_ego_hand.inference import _fit_pred_K
    return _fit_pred_K(torch.cat(rays, dim=1), intr)


# ----------------------------------------------------------------------------- smoothing

def knots_to_frames(k: np.ndarray, n_px: int) -> np.ndarray:
    """Knot samples at px 0,4,8,... -> every px frame by linear interpolation (the model's own scheme)."""
    t_k = np.arange(k.shape[0]) * 4
    t = np.arange(n_px)
    flat = k.reshape(k.shape[0], -1)
    out = np.stack([np.interp(t, t_k, flat[:, i]) for i in range(flat.shape[1])], -1)
    return out.reshape((n_px,) + k.shape[1:])


# ----------------------------------------------------------------------------- output

def to_source_px(j2d_norm: np.ndarray, pin: dict, src_cam: dict | None, src_w: int, src_h: int) -> np.ndarray:
    """(..., 2) normalised pinhole coords -> source continuous pixels."""
    if src_cam is None:     # plain resize of a pinhole source
        return j2d_norm * np.array([src_w, src_h], np.float64)
    uv = j2d_norm * np.array([pin["image_width"], pin["image_height"]], np.float64)
    return project_src(pinhole_rays(uv, pin), src_cam)


def write_outputs(out_dir: Path, ep: dict, meta: dict, pin: dict, src_cam, K_used: dict, K_source: str,
                  blended: dict, n_src: int, smooth: dict, extra: dict):
    """hands2d.json (what board/hands.py reads) and raw.npz (every blended output, unsmoothed); the caller writes
    run.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_px = blended["direct_joints2d"].shape[0]
    j2d = blended["direct_joints2d"]                     # (n_px, 2, 21, 2) normalised
    e2d = blended["exists_2d"]
    # smooth the model's own knots (every 4th frame) in pinhole pixels, then re-interpolate like the model does
    scale = np.array([pin["image_width"], pin["image_height"]], np.float64)
    kn = j2d[::4] * scale
    ek = e2d[::4]
    for s in (0, 1):
        kn[:, s] = one_euro_zero_phase(kn[:, s], ek[:, s] > 0.5, meta["fps"] / 4.0, smooth["min_cut"], smooth["beta"])
    j2d = knots_to_frames(kn / scale, n_px)
    px = to_source_px(j2d, pin, src_cam, meta["width"], meta["height"])     # (n_px, 2, 21, 2)
    # frames past the last 4k+1 frame (at most 3) hold the last prediction
    if n_src > n_px:
        px = np.concatenate([px, np.repeat(px[-1:], n_src - n_px, 0)])
        e2d = np.concatenate([e2d, np.repeat(e2d[-1:], n_src - n_px, 0)])
    bad = ~np.isfinite(px).all(axis=(-1, -2)) | ~np.isfinite(e2d)          # (n, 2) frames with no prediction
    px = np.where(bad[..., None, None], 0.0, px)
    e2d = np.where(bad, 0.0, e2d)
    hands = {}
    for s, name in ((0, "left"), (1, "right")):
        hands[name] = {
            "conf": np.round(e2d[:, s], 3).tolist(),
            "kp": np.round(px[:, s].reshape(len(px), 42), 1).tolist(),
        }
    doc = {
        "format": "ace_ego_hand_2d/v1",
        "dataset": ep["ds"], "episode": ep["ep"],
        "video": {"path": ep["board_path"], "width": meta["width"], "height": meta["height"],
                  "fps": meta["fps"], "n_frames": n_src},
        "coords": "source-video pixels, continuous (x=0 left edge, x=W right edge; pixel j centre = j+0.5)",
        "joints": OP21, "edges": EDGES,
        "handedness": "fixed model slots: left = slot 0, right = slot 1",
        "conf": "per-hand presence probability from the model's 2D presence head; draw when conf >= 0.5",
        "kp": "per frame, 21 joints flattened as x0,y0,...,x20,y20",
        "hands": hands,
        "model": {"name": "ACE-Ego-Hand", "paper": "arXiv:2608.20308", "code_commit": extra["ace_commit"],
                  "checkpoint": extra.get("ckpt"), "head": "direct 2D head (MANO-free)"},
        "camera": {"source_model": (src_cam or {}).get("model", "pinhole"), "virtual_pinhole": pin,
                   "K_used_encode_px": K_used, "K_source": K_source},
        "temporal": {"window_latents": extra["window"], "stride_latents": extra["stride"],
                     "blend": "triangular", "smoothing": {"kind": "one_euro_zero_phase", **smooth},
                     "raw_unsmoothed": "raw.npz direct_joints2d (normalised pinhole coords)"},
    }
    (out_dir / "hands2d.json").write_text(json.dumps(doc, separators=(",", ":")))
    np.savez_compressed(out_dir / "raw.npz", **{k: v.astype(np.float32) for k, v in blended.items()},
                        pinhole=json.dumps(pin), K_used=json.dumps(K_used))
    return doc


def _one_euro_pass(x: np.ndarray, rate: float, min_cut: float, beta: float, d_cut: float, scale: np.ndarray):
    """Causal one-euro filter along axis 0. x (T, ...); scale (T,) normalises speed (hand size in px)."""
    def alpha(cut):
        tau = 1.0 / (2 * np.pi * cut)
        return 1.0 / (1.0 + tau * rate)
    y = np.empty_like(x)
    y[0] = x[0]
    dx_prev = np.zeros_like(x[0])
    ad = alpha(d_cut)
    for t in range(1, len(x)):
        dx = (x[t] - y[t - 1]) * rate
        dx_hat = ad * dx + (1 - ad) * dx_prev
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True) / max(scale[t], 1e-6)   # hand sizes / s
        a = alpha(min_cut + beta * speed)
        y[t] = a * x[t] + (1 - a) * y[t - 1]
        dx_prev = dx_hat
    return y


def one_euro_zero_phase(k: np.ndarray, present: np.ndarray, rate: float, min_cut: float = 1.5, beta: float = 1.0,
                        d_cut: float = 1.0) -> np.ndarray:
    """Speed-adaptive smoothing without lag: one-euro run forward and backward over each present run, averaged.
    Slow hands get a low cutoff (wiggle removed), fast hands a high one (motion kept). k (T, 21, 2) px."""
    y = k.copy()
    idx = np.flatnonzero(present)
    if len(idx) == 0:
        return y
    diag = np.linalg.norm(k.max(1) - k.min(1), axis=-1)
    for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(r) < 3:
            continue
        seg, sc = k[r], np.maximum(diag[r], 1.0)
        f = _one_euro_pass(seg, rate, min_cut, beta, d_cut, sc)
        b = _one_euro_pass(seg[::-1], rate, min_cut, beta, d_cut, sc[::-1])[::-1]
        y[r] = 0.5 * (f + b)
    return y


def stream_vae_encode(vae, frames: np.ndarray, device):
    """Exactly the Wan2.2 VAE's own causal encode (1 frame, then chunks of 4, with its feature cache), but
    fed chunk by chunk from CPU uint8 so GPU memory no longer grows with clip length. Returns mu
    (C, F_lat, h, w) fp16 on the GPU, i.e. what `vae.encode(x).latent_dist.mode()` returns."""
    import torch
    from videox_fun.models.wan_vae3_8 import patchify
    m = vae.model
    m.clear_cache()
    dt = next(m.parameters()).dtype
    scale = [s.to(device, dt) if isinstance(s, torch.Tensor) else s for s in vae.scale]
    T = frames.shape[0]
    outs = []
    with torch.no_grad():
        for i in range(1 + (T - 1) // 4):
            lo, hi = (0, 1) if i == 0 else (1 + 4 * (i - 1), 1 + 4 * i)
            c = torch.from_numpy(frames[lo:hi]).to(device)                     # (t, H, W, 3) uint8
            x = (c.permute(3, 0, 1, 2).unsqueeze(0).to(dt) / 127.5 - 1.0)      # (1, 3, t, H, W)
            m._enc_conv_idx = [0]
            outs.append(m.encoder(patchify(x, patch_size=2), feat_cache=m._enc_feat_map,
                                  feat_idx=m._enc_conv_idx))
        out = torch.cat(outs, 2)
        mu, _ = m.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, m.z_dim, 1, 1, 1)) * scale[1].view(1, m.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
    m.clear_cache()
    return mu[0].to(torch.float16)
