"""Every place a picture is resized, cropped, turned, resampled or given a default, run on a corpus of unusual
recordings made here with ffmpeg: phones held upright and upside down, square, ultrawide, odd, tiny and 4K frames, a
fisheye circle, a variable frame rate, dropped frames, a copy-trimmed file, HEVC 10-bit, grayscale, five containers,
two cameras of different shapes in one take, a LeRobot v3 dataset and an MCAP with a portrait camera.

Each recording goes through the pipeline with no model call: the reader (prepare/formats.py), the dry run's request
(label/route.py, label/episode.py build_request, label/frames.py), the board's clips (board/clips.py), the static
board's web copies and goal frames (board/static.py, board/serve.py extract_frame), the footage download
(board/serve.py footage_command), and the hand tracker's frames and drawing (board/hand_pose/core.py, board/hands.py).

Every frame shows a white square at the top left and a black one at the top right of the picture as it is meant to be
seen, and its own time in 1/30 s as eight blocks. At every artifact the test reads them back and requires the picture
upright, its displayed aspect ratio kept within one pixel, never larger than the source, every frame there and no
other, each frame at its own time, and every camera on the episode's one clock.

tests/fixtures/transform_table.json is each place's measured input and output for every case, and this test keeps it
true; after a deliberate change, rewrite it with TRANSFORM_TABLE=write."""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

FF, FP = shutil.which("ffmpeg"), shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(not (FF and FP), reason="no ffmpeg")

TABLE = Path(__file__).with_name("fixtures") / "transform_table.json"
CODE_RATE = 30            # the time each frame shows, in steps of 1/30 s

H264 = ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "15")


# ---------------------------------------------------------------- the picture every frame shows

def pattern(W: int, H: int) -> str:
    """drawbox filters for a W x H upright picture: mid gray, a white square at the top left, a black one at the top
    right, and the frame's time in 1/30 s as eight blocks (bit i white when set) across the middle."""
    def box(x0, y0, x1, y1, color, enable=None):
        x, y = int(round(x0 * W)), int(round(y0 * H))
        w, h = max(1, int(round(x1 * W)) - x), max(1, int(round(y1 * H)) - y)
        return f"drawbox=x={x}:y={y}:w={w}:h={h}:color={color}:t=fill" + (f":enable='{enable}'" if enable else "")
    f = [box(.22, .22, .36, .36, "white"), box(.64, .22, .78, .36, "black"), box(.2, .5, .8, .62, "black")]
    for i in range(8):
        f.append(box(.2 + .075 * i, .5, .2 + .075 * (i + 1), .62, "white",
                     f"mod(floor(floor(t*{CODE_RATE}+0.5)/{2 ** i})\\,2)"))
    return ",".join(f)


def look(a) -> tuple[str, int]:
    """(orientation, time code) of a picture (a 2-D luminance array, or a PIL image): "upright", or which corners
    the white and the black squares are in."""
    if not isinstance(a, np.ndarray):
        a = np.asarray(a.convert("L"), dtype=float)
    a = a.astype(float)
    h, w = a.shape
    r = max(1, int(0.02 * min(w, h)))

    def at(fx, fy):
        x, y = int(fx * w), int(fy * h)
        return a[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].mean()
    names = ("top left", "top right", "bottom left", "bottom right")
    c = [at(.29, .29), at(.71, .29), at(.29, .71), at(.71, .71)]
    code = sum(int(at(.2 + .075 * (i + .5), .56) > 128) << i for i in range(8))
    if c[0] > 200 and c[1] < 60 and all(80 < x < 180 for x in c[2:]):
        return "upright", code
    return f"white {names[int(np.argmax(c))]}, black {names[int(np.argmin(c))]}", code


# ---------------------------------------------------------------- the corpus

@dataclass
class Cam:
    W: int                      # the picture as it is meant to be seen
    H: int
    frames: int = 60
    rate: int = 30
    pre: str = ""               # filters before the picture is drawn (timing: dropped frames, a variable rate)
    post: str = ""              # filters after it (how the file stores it: turned for a display rotation, a mask)
    enc: tuple = H264
    ext: str = "mp4"
    rot: int | None = None      # the display rotation the file declares
    start: float = 0.0          # an MCAP camera's first message, seconds after the recording's start


HEVC = ("-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error", "-tag:v", "hvc1")
FISHEYE = "geq=lum='if(lte(hypot(X-W/2\\,Y-H/2)\\,0.48*min(W\\,H))\\,lum(X\\,Y)\\,0)':cb=128:cr=128"


@dataclass
class Case:
    rig: str
    kind: str                   # file, folder, lerobot or mcap
    cams: dict                  # file or camera name: Cam
    need: tuple = ("libx264",)  # the encoders it is made with
    trim: float | None = None   # cut without re-encoding from this second (ffmpeg -ss X -c copy)


ONE = "ego_head"
CASES = {
    "phone_portrait_rot90": Case(ONE, "file", {"v.mp4": Cam(360, 640, post="transpose=1", rot=90)}),
    "phone_portrait_rot270": Case(ONE, "file", {"v.mp4": Cam(360, 640, post="transpose=2", rot=270)}),
    "upside_down_rot180": Case(ONE, "file", {"v.mp4": Cam(640, 360, post="hflip,vflip", rot=180)}),
    "square_480x480": Case(ONE, "file", {"v.mp4": Cam(480, 480)}),
    "ultrawide_1280x320": Case(ONE, "file", {"v.mp4": Cam(1280, 320)}),
    "odd_455x255": Case(ONE, "file", {"v.mp4": Cam(455, 255, enc=H264[:4] + ("-pix_fmt", "yuv444p"))}),
    "tiny_160x120": Case(ONE, "file", {"v.mp4": Cam(160, 120)}),
    "uhd_3840x2160_4_frames": Case(ONE, "file", {"v.mp4": Cam(3840, 2160, frames=4)}),
    "uhd_portrait_hevc_rot90_4_frames": Case(ONE, "file", {"v.mp4": Cam(2160, 3840, frames=4, post="transpose=1",
                                                                         rot=90, enc=HEVC + ("-pix_fmt", "yuv420p"))},
                                             need=("libx265",)),
    "fisheye_circle": Case(ONE, "file", {"v.mp4": Cam(640, 480, post=FISHEYE)}),
    "variable_rate_30_then_10": Case(ONE, "file", {"v.mp4": Cam(
        640, 360, frames=40, pre="setpts='if(lt(N,30),N/30,1+(N-30)/10)/TB'")}),
    "dropped_frames": Case(ONE, "file", {"v.mp4": Cam(640, 360, frames=56,
                                                      pre="select='not(between(n,10,12)+eq(n,40))'")}),
    "copy_trimmed": Case(ONE, "file", {"v.mp4": Cam(640, 360, frames=90)}, trim=1.5),
    "hevc_10bit": Case(ONE, "file", {"v.mp4": Cam(640, 360, enc=HEVC + ("-pix_fmt", "yuv420p10le"))},
                       need=("libx265",)),
    "grayscale_hevc": Case(ONE, "file", {"v.mp4": Cam(320, 240, enc=HEVC + ("-pix_fmt", "gray"))},
                           need=("libx265",)),
    "mkv": Case(ONE, "file", {"v.mkv": Cam(640, 360, ext="mkv")}),
    "avi_mpeg4": Case(ONE, "file", {"v.avi": Cam(640, 360, ext="avi", enc=("-c:v", "mpeg4", "-q:v", "4"))},
                      need=("mpeg4",)),
    "webm_vp9": Case(ONE, "file", {"v.webm": Cam(640, 360, ext="webm", enc=(
        "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-b:v", "300k"))}, need=("libvpx-vp9",)),
    "mov": Case(ONE, "file", {"v.mov": Cam(640, 360, ext="mov")}),
    "two_cameras_of_different_shapes": Case("teleop_arms", "folder", {
        "top.mp4": Cam(640, 360), "left_wrist.mp4": Cam(277, 480, frames=30, rate=15,
                                                        enc=H264[:4] + ("-pix_fmt", "yuv444p"))}),
    "lerobot_v3_portrait_wrist": Case("teleop_arms", "lerobot", {
        "observation.images.cam_high": Cam(320, 240, frames=90),
        "observation.images.cam_left_wrist": Cam(240, 320, frames=90, post="transpose=2", rot=270)}),
    "mcap_portrait_wrist": Case("teleop_arms", "mcap", {
        "/cam_high/image/compressed": Cam(320, 240, frames=30, rate=15),
        "/cam_left_wrist/image/compressed": Cam(240, 320, frames=18, rate=10, start=0.2)}, need=("mjpeg",)),
}
LEROBOT_EPISODE_FRAMES = 45


@lru_cache(maxsize=1)
def encoders() -> set:
    out = subprocess.run([FF, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return {ln.split()[1] for ln in out.splitlines() if len(ln.split()) > 1 and ln.startswith(" V")}


def _ff(*a):
    r = subprocess.run([FF, "-v", "error", "-y", *map(str, a)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg {' '.join(map(str, a))[:200]}: {r.stderr[-400:]}")


def make_video(c: Cam, out: Path, trim: float | None = None) -> None:
    """The camera's file: the upright picture drawn at each frame's time, stored as c.post says, then given its
    display rotation (and a copy trim) without re-encoding."""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = ",".join(x for x in (c.pre, pattern(c.W, c.H), c.post) if x)
    sw, sh = (c.H, c.W) if c.rot in (90, 270) else (c.W, c.H)
    tmp = out.with_name("made." + out.name) if (c.rot is not None or trim) else out
    _ff("-f", "lavfi", "-i", f"color=c=0x808080:s={c.W}x{c.H}:r={c.rate}", "-vf", vf, "-frames:v", c.frames,
        "-fps_mode", "passthrough", *c.enc, tmp)
    assert (sw, sh) == video_facts(tmp)["stored"]
    if trim:
        _ff("-ss", trim, "-i", tmp, "-c", "copy", out)
    elif c.rot is not None:
        _ff("-display_rotation", c.rot, "-i", tmp, "-c", "copy", out)
    if tmp != out:
        tmp.unlink()


def make_jpegs(c: Cam, d: Path) -> list[tuple[float, bytes]]:
    """(time from the recording's start, JPEG) for each frame of an MCAP camera."""
    d.mkdir(parents=True, exist_ok=True)
    _ff("-f", "lavfi", "-i", f"color=c=0x808080:s={c.W}x{c.H}:r={c.rate}", "-vf",
        f"setpts=PTS+{c.start}/TB," + pattern(c.W, c.H), "-frames:v", c.frames, "-fps_mode", "passthrough",
        "-c:v", "mjpeg", "-q:v", "3", d / "f%04d.jpg")
    return [(c.start + i / c.rate, p.read_bytes()) for i, p in enumerate(sorted(d.glob("f*.jpg")))]


def make_lerobot(case: Case, root: Path) -> None:
    import pandas as pd
    n, fps, keys = LEROBOT_EPISODE_FRAMES, 30, list(case.cams)
    feats = {k: {"dtype": "video", "shape": [c.H, c.W, 3]} for k, c in case.cams.items()}
    for k, c in case.cams.items():
        sw, sh = (c.H, c.W) if c.rot in (90, 270) else (c.W, c.H)
        # LeRobot writes the stream's own (stored) size
        feats[k]["info"] = {"video.width": sw, "video.height": sh, "video.codec": "h264", "video.fps": fps}
    feats["observation.state"] = {"dtype": "float32", "shape": [14]}
    feats["action"] = {"dtype": "float32", "shape": [14]}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": fps, "robot_type": "two arms",
                                                       "features": feats}))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "move the cube"}) + "\n")
    n_eps = case.cams[keys[0]].frames // n
    rows = []
    for e in range(n_eps):
        r = {"episode_index": e, "length": n, "tasks": ["move the cube"], "data/chunk_index": 0, "data/file_index": 0}
        for k in keys:
            r.update({f"videos/{k}/chunk_index": 0, f"videos/{k}/file_index": 0,
                      f"videos/{k}/from_timestamp": e * n / fps, f"videos/{k}/to_timestamp": (e + 1) * n / fps})
        rows.append(r)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    total = n * n_eps
    state = np.cumsum(np.random.default_rng(0).normal(0, 0.01, (total, 14)), axis=0).astype(np.float32)
    (root / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame({"observation.state": list(state), "action": list(state + 0.01), "index": np.arange(total),
                  "episode_index": np.repeat(np.arange(n_eps), n), "frame_index": np.tile(np.arange(n), n_eps),
                  "timestamp": np.tile(np.arange(n) / fps, n_eps), "task_index": np.zeros(total, dtype=np.int64)}
                 ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    for k, c in case.cams.items():
        make_video(c, root / "videos" / k / "chunk-000" / "file-000.mp4")


def make_mcap(case: Case, path: Path, work: Path) -> dict:
    from mcap.writer import Writer
    msgs, truth = [], {}
    for topic, c in case.cams.items():
        frames = make_jpegs(c, work / "jpeg" / topic.strip("/").replace("/", "_"))
        truth[topic] = frames
        msgs += [(t, topic, j) for t, j in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = 1_790_000_000_000_000_000
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        sid = w.register_schema(name="foxglove.CompressedImage", encoding="jsonschema", data=b"{}")
        ch = {t: w.register_channel(topic=t, message_encoding="json", schema_id=sid) for t in case.cams}
        for t, topic, jpg in sorted(msgs):
            ns = t0 + int(round(t * 1e9))
            w.add_message(ch[topic], log_time=ns, publish_time=ns, data=json.dumps(
                {"timestamp": {"sec": ns // 10 ** 9, "nsec": ns % 10 ** 9}, "frame_id": topic, "format": "jpeg",
                 "data": base64.b64encode(jpg).decode()}).encode())
        w.finish()
    return truth


# ---------------------------------------------------------------- measuring a video

def video_facts(path: Path) -> dict:
    """ffprobe's view of a file's first video stream: stored size, display rotation, sample aspect ratio, and every
    decoded frame's time and length (what a player shows, an edit list applied)."""
    r = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,sample_aspect_ratio:stream_side_data=rotation:"
                        "frame=best_effort_timestamp_time,duration_time", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    st = j["streams"][0]
    rot = next((int(round(float(s["rotation"]))) for s in st.get("side_data_list") or [] if "rotation" in s), 0)
    fr = [f for f in j.get("frames", []) if f.get("best_effort_timestamp_time") not in (None, "N/A")]
    t = np.array([float(f["best_effort_timestamp_time"]) for f in fr])
    d = [float(f["duration_time"]) for f in fr if f.get("duration_time") not in (None, "N/A")]
    return {"stored": (int(st["width"]), int(st["height"])), "rot": rot % 360,
            "sar": st.get("sample_aspect_ratio", "1:1"), "t": t, "last": d[-1] if d else 0.0}


def gray_frames(path: Path, w: int, h: int) -> np.ndarray:
    """Every frame ffmpeg shows, turned upright as the file says, as w x h luminance."""
    raw = subprocess.run([FF, "-v", "error", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-f",
                          "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def shown_of(f: dict) -> tuple:
    w, h = f["stored"]
    return (h, w) if f["rot"] in (90, 270) else (w, h)


def desc(w, h, n=None, t0=None, end=None, rot=0) -> str:
    s = f"{w}x{h}" + (f" rot {rot}" if rot else "")
    if n is not None:
        s += f", {n} frames"
    if end is not None:
        s += f", {t0:.3f}-{end:.3f} s"
    return s


def aspect_kept(w, h, W, H) -> bool:
    """w x h is a W x H picture scaled with each side within one pixel of the same exact scale."""
    lo, hi = max((w - 1) / W, (h - 1) / H), min((w + 1) / W, (h + 1) / H)
    return lo <= hi + 1e-12


class Run:
    """One case through the pipeline: its table rows and every rule it breaks."""

    def __init__(self, name):
        self.name, self.rows, self.failures = name, [], []

    def row(self, cam, place, inp, out):
        self.rows.append({"case": self.name, "camera": cam, "place": place, "in": inp, "out": out})

    def fail(self, cam, place, why):
        self.failures.append(f"{self.name} {cam} {place}: {why}")

    def picture(self, cam, place, size, shown, *, source=None, exact=False):
        """The size rules: never larger than the source (or than what it was made from), the aspect within a pixel."""
        w, h = size
        W, H = shown
        if exact and (w, h) != (W, H):
            self.fail(cam, place, f"{w}x{h}, the picture is {W}x{H}")
        if not aspect_kept(w, h, W, H):
            self.fail(cam, place, f"{w}x{h} does not keep the {W}x{H} picture's aspect ratio within a pixel")
        for sw, sh in [shown] + ([source] if source else []):
            if w > sw or h > sh:
                self.fail(cam, place, f"{w}x{h} is larger than the {sw}x{sh} it was made from")

    def upright(self, cam, place, im, want_code=None, what=""):
        o, code = look(im)
        if o != "upright":
            self.fail(cam, place, f"{what}not upright ({o})")
        elif want_code is not None and code != want_code:
            self.fail(cam, place, f"{what}shows the frame of time code {code}, wanted {want_code}")
        return o == "upright", code


# ---------------------------------------------------------------- the pipeline, one case

def build_upload(name: str, case: Case, work: Path) -> tuple[Path, dict]:
    """The upload folder, and the truth about each camera: {key: {"shown", "t" (s on the file's or recording's
    clock), "code", "facts", "path"}}."""
    up = work / "upload"
    truth = {}
    if case.kind == "mcap":
        frames = make_mcap(case, up / f"{name}.mcap", work)
        for topic, c in case.cams.items():
            truth[topic] = {"shown": (c.W, c.H), "t": np.array([t for t, _ in frames[topic]]),
                            "code": [int(round(t * CODE_RATE)) for t, _ in frames[topic]], "path": None,
                            "facts": {"stored": (c.W, c.H), "rot": 0, "last": 1 / c.rate}}
        return up, truth
    if case.kind == "lerobot":
        make_lerobot(case, up / name)
        files = {k: up / name / "videos" / k / "chunk-000" / "file-000.mp4" for k in case.cams}
    else:
        files = {}
        for fn, c in case.cams.items():
            p = up / (f"{name}{Path(fn).suffix}" if case.kind == "file" else f"{name}/{fn}")
            make_video(c, p, case.trim)
            files[fn] = p
    for k, p in files.items():
        f = video_facts(p)
        W, H = shown_of(f)
        c = case.cams[k]
        assert (W, H) == (c.W, c.H), (k, (W, H))
        codes, bad = [], 0
        for a in gray_frames(p, W, H):
            o, code = look(a)
            bad += o != "upright"
            codes.append(code)
        assert not bad and len(codes) == len(f["t"]), f"{name}: the source itself does not read back"
        truth[k] = {"shown": (W, H), "t": f["t"], "code": codes, "path": p.resolve(), "facts": f}
    return up, truth


def episode_truth(case: Case, truth: dict, src: dict) -> dict:
    """Each view's frames on the episode's clock (0 at the recording's start for an MCAP, at the episode's offset in
    a packed LeRobot file, at each file's own first frame for separate video files)."""
    out = {}
    for v, s in src.items():
        key = s.get("camera_key")
        tr = truth[key] if key in truth else next(x for x in truth.values() if x["path"] == Path(s["packed"]))
        t, code = np.asarray(tr["t"], dtype=float), list(tr["code"])
        if case.kind == "lerobot":
            k0 = int(round(float(s["base_s"]) * 30))
            t, code = t[k0:k0 + int(s["n_frames"])] - float(s["base_s"]), code[k0:k0 + int(s["n_frames"])]
        elif case.kind != "mcap":
            t = t - t[0]
        out[v] = {**tr, "t": t, "code": code}
    return out


def run_case(name: str, work: Path) -> Run:
    from board import clips, hands, serve, static
    from label import episode as me, harness, route
    from prepare import formats
    serve.FFMPEG = FF
    case, run = CASES[name], Run(name)
    up, truth = build_upload(name, case, work)
    for k, tr in truth.items():
        f = tr["facts"]
        run.row(k, "source", "", desc(*f["stored"], len(tr["t"]), tr["t"][0], tr["t"][-1] + f["last"], f["rot"]))
    eps, clip_dir = work / "eps", work / "clips"
    rep = formats.convert(up, case.rig, eps, "corpus", float("inf"))
    if rep["failed"] or not rep["episodes"]:
        run.fail("-", "reader", f"read nothing: {rep['failed']}")
        return run
    for e in rep["episodes"]:
        ep_dir = eps / e["episode_id"]
        tag = "" if len(rep["episodes"]) == 1 else f"{e['episode_id']}/"
        check_episode(run, case, tag, ep_dir, truth, clip_dir, work / "out" / e["episode_id"],
                      clips=clips, hands=hands, serve=serve, static=static, me=me, harness=harness, route=route)
    return run


def check_episode(run, case, tag, ep_dir, truth, clip_dir, out, *, clips, hands, serve, static, me, harness, route):
    ctx = json.loads((ep_dir / "context.json").read_text())
    src = json.loads((ep_dir / "sources.json").read_text())
    et = episode_truth(case, truth, src)
    views = me.order_views(src)

    # the reader: each camera's size as shown, its frames, the episode's length
    for v in views:
        cam, tr = ctx["cameras"][v], et[v]
        n = int(src[v]["n_frames"])
        dur = float(ctx.get("duration_s") or n / float(ctx["fps"]))
        run.row(tag + v, "reader (prepare/formats.py)", desc(*tr["shown"], len(tr["t"])),
                desc(cam["width"], cam["height"], n) + (f", episode {dur:.3f} s" if v == views[0] else ""))
        run.picture(tag + v, "reader", (cam["width"], cam["height"]), tr["shown"], exact=True)
        if n != len(tr["t"]):
            run.fail(tag + v, "reader", f"{n} frames, the source shows {len(tr['t'])}")
    a = et[views[0]]
    span = float(a["t"][-1] - a["t"][0])
    dur = float(ctx.get("duration_s") or int(src[views[0]]["n_frames"]) / float(ctx["fps"]))
    if not span < dur <= span + a["facts"]["last"] + 1e-3:
        run.fail(views[0], "reader", f"episode length {dur:.3f} s, its frames span {span:.3f} s and the last one "
                                     f"lasts {a['facts']['last']:.3f} s")

    # the dry run: the request exactly as it would be sent
    route_w, _ = route.route_width(ep_dir, None, None)
    req = me.build_request(ep_dir, detail=harness.DETAIL, grid_cols=harness.GRID_COLS, max_cell_w=route_w)
    ep, pl = me.load(ep_dir), req["plan"]
    imgs = me.frames(ep, pl)
    cell_w = req["cell"][0]
    steps = me.timesteps(ep, pl, imgs, cell_w)
    name_of = {me.cam_name(ep, v): v for v in views}
    from PIL import Image
    for v in views:
        tr, ks = et[v], pl["ks"]
        sizes = {imgs[v][k].size for k in ks}
        run.row(tag + v, "model frame (label/frames.py extract_frames, upright)", desc(*tr["shown"]),
                f"{', '.join(desc(*s) for s in sorted(sizes))}, {len(ks)} instants")
        for s in sizes:
            run.picture(tag + v, "model frame", s, tr["shown"], exact=True)
        for k in ks:
            if not me.recording_at(ep, v, k):
                continue
            t = me.frame_time(ep, k)
            gap = np.abs(tr["t"] - t)
            near = [tr["code"][i] for i in np.flatnonzero(gap <= gap.min() + 1e-6)]       # either of two at a tie
            ok, code = run.upright(tag + v, "model frame", imgs[v][k], None, f"instant {k} ({t:.3f} s) ")
            if ok and code not in near:
                run.fail(tag + v, "model frame", f"instant {k} ({t:.3f} s) shows time code {code}, the frame "
                                                 f"nearest it is {near}")
    cells = {}
    for (t, row), k in zip(steps, pl["ks"]):
        for nm, jpg in row:
            v = name_of[nm]
            im = Image.open(io.BytesIO(jpg))
            cells.setdefault(v, set()).add(im.size)
            run.picture(tag + v, "grid cell", im.size, et[v]["shown"])
            if im.width > cell_w:
                run.fail(tag + v, "grid cell", f"{im.width} px wide in a {cell_w} px cell")
            run.upright(tag + v, "grid cell", im, look(imgs[v][k])[1], f"instant {k} ")
    for v, s in cells.items():
        full = sorted({imgs[v][k].size for k in pl["ks"]})
        run.row(tag + v, "grid cell (label/frames.py to_jpeg)", f"{', '.join(desc(*x) for x in full)}, cell {cell_w} px",
                ", ".join(desc(*x) for x in sorted(s)))
    check_grids(run, tag, req, steps, pl, name_of, imgs, me)
    check_details(run, tag, req, ep, pl, imgs, et, me)

    # the board's clips, on the episode's clock
    jobs = clips.episode_jobs(ep_dir, clip_dir, True)
    for (pk, b, du, o, fps, is_main, off, skip, t, _ep, cam) in jobs:
        clips.extract_one(pk, b, du, o, FF, 1, fps, is_main, off, skip, t)
    main = clips.main_cam(src)
    offsets = clips.start_offsets(ep_dir, src, float(ctx.get("fps") or 30))
    clip_times = {}
    for v in views:
        tr, p = et[v], clips.clip_path(clip_dir, ep_dir.name, v)
        skip = offsets.get(v, (0.0, 0))[1]
        f = video_facts(p)
        run.row(tag + v, "board clip (board/clips.py)", desc(*tr["shown"], len(tr["t"])),
                desc(*f["stored"], len(f["t"]), f["t"][0], f["t"][-1] + f["last"], f["rot"]))
        half = 0.5 / float(ctx.get("fps") or 30)
        clip_times[v] = check_copy(run, tag + v, "board clip", p, f, tr, skip, half)
        out.mkdir(parents=True, exist_ok=True)
        web = out / f"web_{v}.mp4"
        static.transcode(p, web, 1, v == main)
        wf = video_facts(web)
        run.row(tag + v, "static web copy (board/static.py transcode)", desc(*f["stored"], len(f["t"])),
                desc(*wf["stored"], len(wf["t"]), wf["t"][0], wf["t"][-1] + wf["last"], wf["rot"]))
        check_copy(run, tag + v, "static web copy", web, wf, tr, skip, half)
        check_goal_frames(run, tag + v, p, f, tr, serve, v == main)
        if case.kind == "file":
            # a clip made some other way (here the recording itself) is made into the same kind of web copy
            web = out / "web_source.mp4"
            r = static.transcode(tr["path"], web, 1, True)
            wf = video_facts(web)
            run.row(tag + v, "static web copy of another clip (board/static.py transcode)",
                    desc(*tr["facts"]["stored"], len(tr["t"]), rot=tr["facts"]["rot"]),
                    f"{r['mode']}: " + desc(*wf["stored"], len(wf["t"]), wf["t"][0], wf["t"][-1] + wf["last"],
                                            wf["rot"]))
            check_copy(run, tag + v, "static web copy of another clip", web, wf, tr, 0, half)
    check_footage(run, tag, clip_dir, ep_dir.name, views, et, clip_times, out, serve)
    if case.rig == "ego_head":
        check_hands(run, tag, views[0], et[views[0]], clips.clip_path(clip_dir, ep_dir.name, views[0]), out, hands)


def check_copy(run, cam, place, p, f, tr, skip, half):
    """A clip (or its web copy): upright, the source's frames from skip on and no other, each at its own time on the
    episode's clock, where the main camera's first frame is 0 (a camera that started half a frame or more after it
    starts that much later)."""
    run.picture(cam, place, shown_of(f), tr["shown"])
    if f["rot"] or f["sar"] not in ("1:1", "N/A", "0:1"):
        run.fail(cam, place, f"carries a rotation of {f['rot']} or a pixel shape {f['sar']}, so a player turns or "
                             "stretches it again")
    w, h = shown_of(f)
    want_t, want_c = tr["t"][skip:], tr["code"][skip:]
    got = [look(x) for x in gray_frames(p, w, h)]
    if len(got) != len(want_c):
        run.fail(cam, place, f"{len(got)} frames, the source shows {len(want_c)}")
    bad = [i for i, (o, c) in enumerate(got[:len(want_c)]) if o != "upright" or c != want_c[i]]
    if bad:
        run.fail(cam, place, f"{len(bad)} frames not upright or not the source's frame (first at {bad[0]}: "
                             f"{got[bad[0]]}, wanted code {want_c[bad[0]]})")
    t = f["t"]
    expect = want_t - (want_t[0] if want_t[0] < half else 0.0)
    end, want_end = t[-1] + f["last"], expect[-1] + tr["facts"]["last"]
    if abs(end - want_end) > 2e-3:
        run.fail(cam, place, f"ends at {end:.4f} s, its last frame ends at {want_end:.4f} s in the source")
    if len(t) == len(expect) and np.max(np.abs(t - expect)) > 2e-3:
        i = int(np.argmax(np.abs(t - expect)))
        run.fail(cam, place, f"frame {i} plays at {t[i]:.4f} s, it was recorded at {expect[i]:.4f} s on the "
                             "episode's clock")
    return t, want_c


def check_goal_frames(run, cam, clip, f, tr, serve, main):
    """Goal frames (the live /api/frame, the static board's media/f): the main camera's frame nearest the time asked
    and every camera's first frame (its poster), at most 640 px wide."""
    t = f["t"] - f["t"][0]
    w, h = shown_of(f)
    codes = [look(x)[1] for x in gray_frames(clip, w, h)]
    asks = sorted({0.0, round(float(t[len(t) // 2]) + 0.004, 3), round(float(t[-1]), 3),
                   round(float(t[-1]) + 0.5, 3)}) if main else [0.0]
    from PIL import Image
    sizes = set()
    for ask in asks:
        jpg = serve.extract_frame(clip, ask, 640)
        if not jpg:
            run.fail(cam, "goal frame", f"none at {ask} s")
            continue
        im = Image.open(io.BytesIO(jpg))
        sizes.add(im.size)
        run.picture(cam, "goal frame", im.size, tr["shown"], source=(w, h))
        if im.width > 640:
            run.fail(cam, "goal frame", f"{im.width} px wide, at most 640")
        run.upright(cam, "goal frame", im, codes[int(np.argmin(np.abs(t - ask)))], f"at {ask} s ")
    run.row(cam, "goal frame (board/serve.py extract_frame, 640)", desc(w, h),
            ", ".join(desc(*s) for s in sorted(sizes)) + f" at {', '.join(f'{x:g}' for x in asks)} s")


def check_footage(run, tag, clip_dir, eid, views, et, clip_times, out, serve):
    """The footage download: the main camera at its clip's size, the others beside it, never enlarged; at each of its
    frames every camera shows the frame it shows at that time on the page, to the last frame."""
    cams = serve.footage_cams(clip_dir, eid)
    probes = [serve._probe(p) for _, p in cams]
    dst = out / "footage.mp4"
    out.mkdir(parents=True, exist_ok=True)
    cmd = serve.footage_command([(p, pr[:2], pr[3]) for (_, p), pr in zip(cams, probes)], 0.0,
                                round(probes[0][2], 3), dst, 1)
    subprocess.run(cmd, check=True, capture_output=True)
    f = video_facts(dst)
    W, H, cells = serve.footage_layout([pr[:2] for pr in probes])
    run.row(tag + "+".join(c for c, _ in cams), "footage download (board/serve.py footage_command)",
            " + ".join(desc(*pr[:2]) for pr in probes) + f", {probes[0][2]:.3f} s",
            desc(*f["stored"], len(f["t"]), f["t"][0], f["t"][-1] + f["last"]) + " with "
            + " + ".join(f"{w}x{h} at {x},{y}" for x, y, w, h in cells))
    if f["stored"] != (W, H):
        run.fail(tag, "footage", f"{f['stored']}, its layout is {W}x{H}")
    end = f["t"][-1] + f["last"]
    if abs(end - probes[0][2]) > f["last"] + 1e-3:
        run.fail(tag, "footage", f"runs to {end:.3f} s, the main clip to {probes[0][2]:.3f} s")
    frames = gray_frames(dst, *f["stored"])
    for (cam, _), pr, (x, y, w, h) in zip(cams, probes, cells):
        v = cam
        run.picture(tag + v, "footage cell", (w, h), et[v]["shown"], source=pr[:2])
        ct, codes = clip_times[v]
        bad = 0
        for i, t in enumerate(f["t"]):
            j = int(np.searchsorted(ct, t + 5e-4, side="right")) - 1
            if j < 0:
                continue
            o, c = look(frames[i][y:y + h, x:x + w])
            bad += o != "upright" or c != codes[j]
            if bad == 1 and (o != "upright" or c != codes[j]):
                run.fail(tag + v, "footage cell", f"at {t:.3f} s shows {o} code {c}, the page shows {codes[j]}")
        if bad > 1:
            run.fail(tag + v, "footage cell", f"{bad} of {len(f['t'])} frames wrong")


def check_hands(run, tag, v, tr, clip, out, hands):
    """The hand tracker reads the source's frames upright at its shown size, every one; a joint it puts on the white
    square lands on the white square in the board clip."""
    from board.hand_pose import core
    path = str(tr["path"])
    meta = core.probe(path)
    fr = list(core.read_frames(path, meta["width"], meta["height"], meta.get("resample", False)))
    run.row(tag + v, "hand tracker frames (board/hand_pose/core.py)", desc(*tr["shown"], len(tr["t"])),
            desc(meta["width"], meta["height"], len(fr)))
    run.picture(tag + v, "hand tracker", (meta["width"], meta["height"]), tr["shown"], exact=True)
    if len(fr) != len(tr["t"]):
        run.fail(tag + v, "hand tracker", f"read {len(fr)} frames, the source shows {len(tr['t'])}")
    bad = [i for i, x in enumerate(fr) if look(x.mean(axis=2)) != ("upright", tr["code"][min(i, len(tr["code"]) - 1)])]
    if bad:
        run.fail(tag + v, "hand tracker", f"{len(bad)} frames not upright or not the source's (first {bad[0]})")
    W, H = meta["width"], meta["height"]
    kp = [[0.29 * W if c % 2 == 0 else 0.29 * H for c in range(42)] for _ in fr]
    d = {"video": {"width": W, "height": H}, "joints": [f"j{i}" for i in range(21)], "edges": [[0, 1]],
         "hands": {h: {"kp": kp, "conf": [0.9] * len(fr)} for h in ("left", "right")}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "hands2d.json").write_text(json.dumps(d))
    (out / "hands").mkdir(exist_ok=True)
    r = hands.build_one(("k", out / "hands2d.json", "drawing.json", clip, out / "hands", {}))
    if "skip" in r:
        run.fail(tag + v, "hand drawing", r["skip"])
        return
    doc = json.loads((out / "hands" / "drawing.json").read_text())
    pts = hands.decode_hand(doc["left"], doc["clip"]["frames"], doc["step"])
    cw, ch = doc["clip"]["w"], doc["clip"]["h"]
    x, y = pts[0][0], pts[0][1]
    run.row(tag + v, "hand drawing (board/hands.py)", f"{W}x{H} source pixels, joint at "
            f"{0.29 * W:.1f},{0.29 * H:.1f}", f"{cw}x{ch} clip pixels, joint at {x:g},{y:g}")
    if abs(x - 0.29 * cw) > 1 or abs(y - 0.29 * ch) > 1:
        run.fail(tag + v, "hand drawing", f"the joint lands at {x},{y} in the {cw}x{ch} clip, the square is at "
                                          f"{0.29 * cw:.1f},{0.29 * ch:.1f}")


def _images(req, prefix):
    from PIL import Image
    out, last = [], ""
    for c in req["content"]:
        if c["type"] == "text":
            last = c["text"]
        elif last.startswith(prefix):
            out.append((last, Image.open(io.BytesIO(base64.b64decode(c["image_url"]["url"].split(",", 1)[1])))))
    return out


def check_grids(run, tag, req, steps, pl, name_of, imgs, me):
    """Each grid as sent: every cell in place, upright, the instant's frame."""
    from PIL import Image
    grids = _images(req, "=== grid")
    labels = req["cam_labels"]
    cols = len(steps) // len(grids) + (len(steps) % len(grids) > 0) if grids else 0
    sizes = []
    for gi, (_, g) in enumerate(grids):
        block = steps[gi * cols:(gi + 1) * cols] if gi < len(grids) - 1 else steps[gi * cols:]
        dec = {(ri, ci): Image.open(io.BytesIO(dict(row).get(nm))) for ci, (_, row) in enumerate(block)
               for ri, nm in enumerate(labels) if dict(row).get(nm) is not None}
        cw = max(im.width for im in dec.values())
        rh = [max((im.height for (r, _), im in dec.items() if r == ri), default=0) for ri in range(len(labels))]
        y0 = [me.GRID_HEADER + sum(h + 4 for h in rh[:ri]) for ri in range(len(labels))]
        sizes.append(g.size)
        for (ri, ci), im in dec.items():
            x = me.GRID_GUTTER + ci * (cw + 4) + (cw - im.width) // 2
            crop = g.crop((x, y0[ri], x + im.width, y0[ri] + im.height))
            v = name_of[labels[ri]]
            run.upright(tag + v, "grid image", crop, look(im)[1], f"grid {gi} column {ci} ")
    run.row(tag + "+".join(name_of[n] for n in labels), "grid image (label/frames.py compose_grid)",
            f"{len(steps)} instants, {cols} per grid", ", ".join(sorted({desc(*s) for s in sizes})))


def check_details(run, tag, req, ep, pl, imgs, et, me):
    """The first and last frames at detail size (at most 768 px wide), stacked under their name strips."""
    found = _images(req, "=== detail view, first frame") + _images(req, "=== detail view, last frame")
    for (text, g), k, which in zip(found, (pl["ks"][0], pl["ks"][-1]), ("first", "last")):
        here = [v for v in me.order_views(imgs) if me.recording_at(ep, v, k)]
        y, outs = 0, []
        for v in here:
            w, h = me.detail_size(*imgs[v][k].size)
            crop = g.crop((0, y + 26, w, y + 26 + h))
            outs.append(f"{v} {w}x{h}")
            run.picture(tag + v, "detail view", (w, h), et[v]["shown"])
            if w > me.DETAIL_MAX_W:
                run.fail(tag + v, "detail view", f"{w} px wide, at most {me.DETAIL_MAX_W}")
            run.upright(tag + v, "detail view", crop, look(imgs[v][k])[1], f"{text[17:27]} ")
            y += h + 26
        if g.size != (max(int(o.split()[1].split("x")[0]) for o in outs), y):
            run.fail(tag, "detail view", f"the image is {g.size}, its cameras stack to {y} px")
        run.row(tag + "+".join(here), f"detail view, {which} frame (label/episode.py fullres_stack)",
                ", ".join(f"{v} {desc(*imgs[v][k].size)}" for v in here), f"{desc(*g.size)}: " + ", ".join(outs))


# ---------------------------------------------------------------- the tests

@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    work = tmp_path_factory.mktemp("corpus")
    have = encoders()
    names = [n for n, c in CASES.items() if all(e in have for e in c.need)]

    def one(n):
        try:
            return run_case(n, work / n)
        except Exception:
            r = Run(n)
            r.failures.append(f"{n}: the pipeline raised\n{traceback.format_exc()[-1500:]}")
            return r
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ex:
        return dict(zip(names, ex.map(one, names)))


# cases that still break a rule, each with where; a fix removes its case (strict, so a case that passes must go)
KNOWN = {}


@pytest.mark.parametrize("name", list(CASES))
def test_every_transform_keeps_the_picture_upright_whole_and_on_time(corpus, name, request):
    if name not in corpus:
        pytest.skip(f"this ffmpeg has no {', '.join(CASES[name].need)} encoder")
    if name in KNOWN:
        request.node.add_marker(pytest.mark.xfail(reason=KNOWN[name], strict=True))
    assert not corpus[name].failures, "\n".join(corpus[name].failures)


def test_the_table_is_what_the_code_does(corpus):
    rows = [r for n in CASES if n in corpus for r in corpus[n].rows]
    if os.environ.get("TRANSFORM_TABLE") == "write":
        TABLE.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    want = [r for r in json.loads(TABLE.read_text()) if r["case"] in corpus]
    key = lambda r: (r["case"], r["camera"], r["place"])
    got_by, want_by = {key(r): r for r in rows}, {key(r): r for r in want}
    diff = [f"{k}: table {want_by.get(k)}, measured {got_by.get(k)}" for k in sorted(set(got_by) | set(want_by))
            if got_by.get(k) != want_by.get(k)]
    assert not diff, "the table no longer says what the code does (TRANSFORM_TABLE=write rewrites it):\n" + \
        "\n".join(diff[:20])
