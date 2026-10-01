"""2D hand keypoints of head-camera episodes with ACE-Ego-Hand (arXiv 2608.20308), on Modal GPUs, for the board's
"Hand pose" overlay and keypoint downloads. For non-commercial use only: the weights are CC BY-NC 4.0 and the model
uses MANO, which is licensed for non-commercial scientific research.

You need a Modal account (`modal setup` once) and MANO: register at mano.is.tue.mpg.de, download the MANO models
and convert MANO_LEFT.pkl and MANO_RIGHT.pkl once with scripts/convert_mano_pkls.py of ACE-Ego-Hand (at ACE_COMMIT
below). MANO drives only the model's 3D translation decode; the 2D keypoints come from its MANO-free 2D head. Then,
from the repository root:

    MODAL="uvx --from modal==1.5.3 modal"          # the Modal client in its own environment
    uv run python -m board.hand_pose.specs data/episodes/openaoe/quickstart --out data/hand_pose/specs.json
    $MODAL run board/hand_pose/modal_app.py::prep_weights       # once: Wan2.2 backbone, ACE-Ego-Hand weights
    $MODAL volume put hand-pose-weights MANO_LEFT.pkl mano/mano/MANO_LEFT.pkl      # once, your converted MANO
    $MODAL volume put hand-pose-weights MANO_RIGHT.pkl mano/mano/MANO_RIGHT.pkl
    $MODAL run board/hand_pose/modal_app.py::upload --specs data/hand_pose/specs.json
    $MODAL run --detach board/hand_pose/modal_app.py::full --specs data/hand_pose/specs.json
    $MODAL run board/hand_pose/modal_app.py::pull --out data/hand_pose/keypoints

`full` skips episodes that already have an output (add --force to redo them), so it can be run again after an
interruption; --detach keeps it running if your machine disconnects. `pull` writes index.json and
<dataset>/<episode>/{hands2d.json, raw.npz, run.json}. Name that folder in the board manifest,
"hands": {"src": "../../hand_pose/keypoints", "clips": "../../clips"}, and `python -m board build` turns it into
the overlay (board/hands.py).

This file imports only modal and the standard library on your machine; everything else runs in the image.

A rerun beside an earlier one names its own output volume and, when the specs change, its own specs volume, so
nothing of the earlier run is overwritten:

    HAND_POSE_VOLUMES=ace HAND_POSE_SPECS=ace-specs-v2 $MODAL run board/hand_pose/modal_app.py::upload_specs --specs v2.json
    HAND_POSE_VOLUMES=ace HAND_POSE_SPECS=ace-specs-v2 HAND_POSE_OUT=ace-out-v2 $MODAL run --detach \
        board/hand_pose/modal_app.py::full --specs v2.json

Volumes (created on first use, names from HAND_POSE_VOLUMES, default "hand-pose"):
  <prefix>-weights  ckpt/Wan2.2-Fun-5B-Control/..., checkpoints/ace_ego_hand_{k,kfree}.pt, cache/caption_embed.pt,
                    mano/mano/MANO_{LEFT,RIGHT}.pkl
  <prefix>-clips    <dataset>/<episode>/source.mp4 and spec.json
  <prefix>-out      <dataset>/<episode>/{hands2d.json, raw.npz, run.json}

Measured on H100 80GB: about 80 s of GPU per 3-minute Egocentric-100K clip, 130 s per Gen-HumanEgo episode and
180 s per 5-minute 1080p OpenAoE clip with the model loaded, plus about a minute per container start.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PREFIX = os.environ.get("HAND_POSE_VOLUMES", "hand-pose")
ACE_COMMIT = "97578680931b3f1c8111396c100d595b98857fb8"          # github.com/ggxxii/ACE-Ego-Hand, code MIT
VIDEOX_COMMIT = "968f0e2192ba4c7a12868bf36d73260d135424ca"       # github.com/aigc-apps/VideoX-Fun

app = modal.App("hand-pose")
wvol = modal.Volume.from_name(f"{PREFIX}-weights", create_if_missing=True)
cvol = modal.Volume.from_name(f"{PREFIX}-clips", create_if_missing=True)
# a rerun writes to its own output volume and may read its specs from their own volume, so an earlier run's outputs
# and specs are never overwritten (HAND_POSE_OUT=ace-out-v2 HAND_POSE_SPECS=ace-specs-v2)
OUT = os.environ.get("HAND_POSE_OUT") or f"{PREFIX}-out"
SPECS = os.environ.get("HAND_POSE_SPECS") or None
ovol = modal.Volume.from_name(OUT, create_if_missing=True)
svol = modal.Volume.from_name(SPECS, create_if_missing=True) if SPECS else None

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "numpy==1.24.4", "opencv-python-headless==4.11.0.86", "omegaconf==2.3.0", "diffusers==0.36.0",
        "transformers==4.57.3", "accelerate==1.12.0", "safetensors==0.4.5", "einops==0.8.1", "peft==0.18.1",
        "smplx==0.1.28", "PyYAML==6.0.2", "scipy==1.13.1", "huggingface_hub[hf_transfer]==0.36.0",
        "sentencepiece", "ftfy", "timm", "imageio", "imageio-ffmpeg", "beautifulsoup4", "func_timeout",
        "scikit-image", "tomesd", "torchdiffeq", "torchsde", "librosa", "decord", "datasets", "albumentations",
        "av", "onnxruntime",
    )
    .run_commands("git clone https://github.com/ggxxii/ACE-Ego-Hand /opt/ace",
                  f"cd /opt/ace && git checkout {ACE_COMMIT}",
                  "git clone https://github.com/aigc-apps/VideoX-Fun /opt/ace/third_party",
                  f"cd /opt/ace/third_party && git checkout {VIDEOX_COMMIT}")
    .env({"ACE_EGO_HAND_MANO_DIR": "/w/mano", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1",
          # the container must name the same volumes as the machine that launched it
          "HAND_POSE_VOLUMES": PREFIX, "HAND_POSE_OUT": OUT, "HAND_POSE_SPECS": SPECS or ""})
    .add_local_file(HERE / "core.py", "/opt/ace/ace_core.py", copy=True)
    .add_local_file(HERE.parent.parent / "prepare" / "display.py", "/opt/ace/display.py", copy=True)
)

WINDOW, STRIDE = 22, 11                  # latents per window, and the step between windows (half overlap)
CALIB_WINDOWS = 12                       # windows the K-free checkpoint reads to estimate an unknown camera
SMOOTH = {"min_cut": 1.5, "beta": 1.0}   # one-euro filter at the knot rate (core.py says why)
READOUT_RADIUS = 3                       # the second 2D readout, near each slot's attention peak (core.combine_readouts)
WAN = "alibaba-pai/Wan2.2-Fun-5B-Control"


def _numpy2_pickles():
    """Pickles written under numpy 2 (the converted MANO files) name numpy._core.*; this image has numpy 1.24, where
    the same modules are numpy.core.*. Map one to the other so pickle finds them; nothing else changes."""
    import importlib
    import importlib.abc
    import importlib.util
    import sys
    import numpy

    class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path=None, target=None):
            if name == "numpy._core" or name.startswith("numpy._core."):
                return importlib.util.spec_from_loader(name, self)
            return None

        def create_module(self, spec):
            return importlib.import_module("numpy.core" + spec.name[len("numpy._core"):])

        def exec_module(self, module):
            pass

    if not hasattr(numpy, "_core") and not any(type(f).__name__ == "_Alias" for f in sys.meta_path):
        sys.meta_path.insert(0, _Alias())


def _link_weights():
    """ACE-Ego-Hand's own scripts read ckpt/ and cache/ under its checkout; point them at the weights volume."""
    for name in ("ckpt", "cache", "checkpoints"):
        p = Path("/opt/ace") / name
        if not p.exists():
            (Path("/w") / name).mkdir(parents=True, exist_ok=True)
            p.symlink_to(Path("/w") / name)


@app.function(image=image, volumes={"/w": wvol}, gpu="L4", timeout=3 * 3600, cpu=4, memory=49152)
def prep_weights():
    """The Wan2.2-Fun-5B-Control backbone and ACE-Ego-Hand checkpoints from the Hub, then the fixed caption embedding
    (ACE-Ego-Hand's scripts/precompute_caption.py, run once). MANO is not downloaded: put your own files in."""
    import subprocess
    from huggingface_hub import hf_hub_download, snapshot_download
    _link_weights()
    t0 = time.time()
    for f in ["config.json", "configuration.json", "Wan2.2_VAE.pth", "diffusion_pytorch_model.safetensors",
              "models_t5_umt5-xxl-enc-bf16.pth"]:
        hf_hub_download(WAN, f, local_dir="/w/ckpt/Wan2.2-Fun-5B-Control")
        print(f, round(time.time() - t0), "s", flush=True)
    snapshot_download(WAN, allow_patterns=["google/umt5-xxl/*"], local_dir="/w/ckpt/Wan2.2-Fun-5B-Control")
    for f in ["ace_ego_hand_k.pt", "ace_ego_hand_kfree.pt"]:
        hf_hub_download("acerobotics2025/ACE-Ego-Hand", f, local_dir="/w/checkpoints")
        print(f, round(time.time() - t0), "s", flush=True)
    if not Path("/w/cache/caption_embed.pt").exists():
        subprocess.run(["python", "scripts/precompute_caption.py"], cwd="/opt/ace", check=True)
    wvol.commit()
    missing = [f for f in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl") if not Path(f"/w/mano/mano/{f}").exists()]
    print("weights ready" + (f"; still missing {missing}: modal volume put {PREFIX}-weights <file> mano/mano/<file>"
                             if missing else ""), flush=True)


@app.local_entrypoint()
def upload(specs: str):
    """Each spec's source clip and the spec itself to the clips volume (skips clips already there at that size)."""
    have = {e.path.lstrip("/"): e.size for e in cvol.iterdir("/", recursive=True)}
    todo = [s for s in json.loads(Path(specs).read_text())
            if have.get(f"{s['ds']}/{s['ep']}/source.mp4") != s["size"]]
    with tempfile.TemporaryDirectory() as tmp, cvol.batch_upload(force=True) as b:
        for s in todo:
            key = f"{s['ds']}/{s['ep']}"
            b.put_file(s["src"], f"/{key}/source.mp4")
            spec = Path(tmp) / f"{s['ds']}__{s['ep']}.spec.json"
            spec.write_text(json.dumps(s))
            b.put_file(str(spec), f"/{key}/spec.json")
    print(f"uploaded {len(todo)} clips")


@app.local_entrypoint()
def upload_specs(specs: str):
    """Only each spec to the specs volume (HAND_POSE_SPECS), leaving the clips volume and its specs untouched."""
    if svol is None:
        raise SystemExit("set HAND_POSE_SPECS to the specs volume")
    items = json.loads(Path(specs).read_text())
    with tempfile.TemporaryDirectory() as tmp, svol.batch_upload(force=True) as b:
        for s in items:
            spec = Path(tmp) / f"{s['ds']}__{s['ep']}.spec.json"
            spec.write_text(json.dumps(s))
            b.put_file(str(spec), f"/{s['ds']}/{s['ep']}/spec.json")
    print(f"uploaded {len(items)} specs to {SPECS}")


def _opt(name: str) -> dict:
    import yaml
    opt = yaml.safe_load(open(f"/opt/ace/options/{name}.yml"))
    opt["paths"]["model_root"] = "/w/ckpt/Wan2.2-Fun-5B-Control"
    opt["paths"]["caption_embed"] = "/w/cache/caption_embed.pt"
    opt["paths"]["videox_config"] = "/opt/ace/third_party/config/wan2.2/wan_civitai_5b.yaml"
    return opt


@app.cls(image=image, gpu="H100", volumes={"/w": wvol, "/c": cvol, "/o": ovol, **({"/s": svol} if svol else {})},
         timeout=3 * 3600,
         memory=65536, cpu=8, max_containers=40)
class Ace:
    @modal.enter()
    def load(self):
        import sys
        import torch
        sys.path.insert(0, "/opt/ace")
        os.chdir("/opt/ace")
        _numpy2_pickles()
        import ace_ego_hand.video_vae as vv
        vv.MODEL_ROOT = Path("/w/ckpt/Wan2.2-Fun-5B-Control")
        vv.VIDEOX_CFG = Path("/opt/ace/third_party/config/wan2.2/wan_civitai_5b.yaml")
        from ace_ego_hand.models.geodit_model import GeoDitModel
        t0 = time.time()
        self.dev = torch.device("cuda")
        self.vae = vv.load_vae(self.dev)
        self.k_model = GeoDitModel(_opt("ace_ego_hand_k"), self.dev)
        self.k_model.load_inference("/w/checkpoints/ace_ego_hand_k.pt")
        self.kfree = None
        self.load_sec = time.time() - t0

    def _kfree(self):
        if self.kfree is None:
            from ace_ego_hand.models.geodit_model import GeoDitModel
            m = GeoDitModel(_opt("ace_ego_hand_kfree"), self.dev)
            m.load_inference("/w/checkpoints/ace_ego_hand_kfree.pt")
            self.kfree = m
        return self.kfree

    @modal.method()
    def run(self, key: str) -> dict:
        """One episode: decode to the virtual pinhole, VAE-encode (streamed, so GPU memory is flat in clip length),
        the K-given model over half-overlapping windows, blend, smooth, map back to the source pixels."""
        import numpy as np
        import torch
        import ace_core as C
        t_all = time.time()
        ovol.reload()
        cvol.reload()
        if svol:
            svol.reload()
        spec = json.loads(Path(f"/{'s' if svol else 'c'}/{key}/spec.json").read_text())
        out_dir = Path(f"/o/{key}")
        video = f"/c/{key}/source.mp4"
        meta = C.probe(video)
        cam = spec["camera"]
        if cam["model"] in ("kb", "ds"):
            src_cam = cam
            pin = C.virtual_pinhole(spec["pinhole"]["w"], spec["pinhole"]["h"], spec["pinhole"]["hfov_deg"],
                                    spec["pinhole"].get("pitch_deg", 0.0))
        else:
            src_cam = None
            ew = 832
            eh = max(32, int(round(meta["height"] * ew / meta["width"] / 32)) * 32)
            pin = {"image_width": ew, "image_height": eh}
        frames = C.load_pinhole_frames(video, meta, src_cam, pin)
        n_src = len(frames)
        frames = frames[:4 * ((n_src - 1) // 4) + 1]
        ctrl = C.stream_vae_encode(self.vae, frames, self.dev)
        del frames
        torch.cuda.empty_cache()
        extra = {"key": key, "pinhole": pin, "gpu": torch.cuda.get_device_name(), "n_src": n_src,
                 "window": WINDOW, "stride": STRIDE, "ckpt": "ace_ego_hand_k.pt", "ace_commit": ACE_COMMIT,
                 "readout_radius": READOUT_RADIUS,
                 "load_sec": self.load_sec}
        if src_cam is not None:
            K_used = {k: pin[k] for k in ("fx", "fy", "cx", "cy", "image_width", "image_height")}
            K_source = "calibration (virtual pinhole of the calibrated source camera)"
        else:
            # no trustworthy camera: the K-free checkpoint's ray field over a spread of windows gives one camera
            # for the whole clip (only the 3D decode reads it; the 2D head does not)
            allw = C.window_anchors(ctrl.shape[1], WINDOW, WINDOW)
            pick = sorted(set(int(round(v)) for v in np.linspace(0, len(allw) - 1, min(CALIB_WINDOWS, len(allw)))))
            foc = 0.5 * pin["image_width"] / np.tan(np.deg2rad(30.0))
            ph = {"fx": foc, "fy": foc, "cx": pin["image_width"] / 2, "cy": pin["image_height"] / 2,
                  "image_width": pin["image_width"], "image_height": pin["image_height"]}
            _, _, rays = C.run_windows(self._kfree(), ctrl, ph, WINDOW, WINDOW, want_rays=True,
                                       anchors=[allw[i] for i in pick])
            K_est = C.fit_K_from_rays(rays, ph)
            f = 0.5 * (K_est["fx"] + K_est["fy"])
            K_used = {"fx": f, "fy": f, "cx": K_est["cx"], "cy": K_est["cy"],
                      "image_width": pin["image_width"], "image_height": pin["image_height"]}
            K_source = "estimated by the K-free checkpoint's ray field over the whole clip"
        blended, st, _ = C.run_windows(self.k_model, ctrl, K_used, WINDOW, STRIDE)
        extra["windows"] = st
        # the same windows again with the 2D read near each hand's attention peak; where the model's own reading is a
        # blend of two hands, that reading is drawn instead (core.combine_readouts)
        C.set_readout(self.k_model, READOUT_RADIUS)
        try:
            peak, _, _ = C.run_windows(self.k_model, ctrl, K_used, WINDOW, STRIDE)
        finally:
            C.set_readout(self.k_model, None)
        j2d, extra["readout"] = C.combine_readouts(blended["direct_joints2d"], peak["direct_joints2d"],
                                                   blended["exists_2d"] >= 0.5, pin)
        blended["direct_joints2d_model"] = blended["direct_joints2d"]
        blended["direct_joints2d_peak"] = peak["direct_joints2d"]
        blended["direct_joints2d"] = j2d
        C.write_outputs(out_dir, {"ds": spec["ds"], "ep": spec["ep"], "board_path": spec["board_path"]},
                        meta, pin, src_cam, K_used, K_source, blended, n_src, SMOOTH, extra)
        extra["total_gpu_fn_sec"] = time.time() - t_all
        (out_dir / "run.json").write_text(json.dumps(extra, indent=1, default=str))
        ovol.commit()
        return extra


@app.local_entrypoint()
def full(specs: str, force: bool = False):
    """Every spec's episode (resumable: skips episodes with an output unless --force). Failures of Modal's own
    infrastructure (preemption, a container lost) are retried twice; an error in the pipeline is reported."""
    keys_all = [f"{s['ds']}/{s['ep']}" for s in json.loads(Path(specs).read_text())]
    done = {e.path.lstrip("/") for e in ovol.iterdir("/", recursive=True)}
    todo = [k for k in keys_all if force or f"{k}/hands2d.json" not in done]
    print(f"{len(todo)} episodes to run ({len(keys_all) - len(todo)} already done)", flush=True)
    t0, ok, failed = time.time(), [], {}
    for attempt in range(3):
        infra = []
        for k, r in zip(todo, Ace().run.map(todo, return_exceptions=True)):
            if not isinstance(r, Exception):
                ok.append(k)
                failed.pop(k, None)
                print(f"{time.time() - t0:.0f}s {k} {r['total_gpu_fn_sec']:.0f}s", flush=True)
                continue
            failed[k] = f"{type(r).__module__}.{type(r).__name__}: {str(r)[:400]}"
            print(f"{time.time() - t0:.0f}s {k} FAILED {failed[k]}", flush=True)
            if type(r).__module__.startswith("modal"):
                infra.append(k)
        if not infra:
            break
        todo = infra
    print(f"ok={len(ok)} failed={len(failed)}", flush=True)


@app.local_entrypoint()
def pull(out: str):
    """Every output to OUT/<dataset>/<episode>/ and OUT/index.json, which board/hands.py reads."""
    root = Path(out)
    index = {}
    for e in ovol.iterdir("/", recursive=True):
        p = e.path.lstrip("/").split("/")
        if len(p) != 3 or p[2] not in ("hands2d.json", "raw.npz", "run.json"):
            continue
        dst = root / p[0] / p[1] / p[2]
        if not (dst.exists() and dst.stat().st_size == e.size):
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                for chunk in ovol.read_file(e.path):
                    f.write(chunk)
        if p[2] == "hands2d.json":
            index[f"{p[0]}/{p[1]}"] = {"hands2d": f"{p[0]}/{p[1]}/hands2d.json", "raw": f"{p[0]}/{p[1]}/raw.npz"}
    (root / "index.json").write_text(json.dumps(dict(sorted(index.items())), indent=1))
    print(f"{len(index)} episodes in {root}")
