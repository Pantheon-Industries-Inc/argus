"""The same source packs to the same bytes on any machine and in any process. A packed file's bytes are part of an
episode's identity (label/evidence_access.py source_proof), and CI's test_ownership_context failed when one upload,
prepared twice in one process on x86, packed its camera to two different videos. swscale and x86 x264 each take SIMD
code that rounds differently from their C code, and which code runs is process state; Matroska wrote random UIDs."""
import ctypes
import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from prepare import formats


def _pictures(n=24, h=72, w=96, seed=0):
    """A moving, noisy clip: real content, whose conversion and encoding exercise the code that rounds."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return [np.roll(base, 3 * k, axis=1) // 2 + rng.integers(0, 40, (h, w, 3), dtype=np.uint8) for k in range(n)]


def _pack(path, pictures):
    w = formats.FrameWriter(path, "", blank=True)
    for k, a in enumerate(pictures):
        w.add_image(k / 30, None if k == 5 else Image.fromarray(a))      # one placeholder frame
    w.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _avutil():
    """PyAV's own libavutil, already loaded in this process."""
    import av
    root = Path(av.__file__).parent
    found = sorted(root.glob(".dylibs/libavutil*.dylib")) + sorted(root.parent.glob("av.libs/libavutil*.so*"))
    assert found, "PyAV's libavutil was not found"
    return ctypes.CDLL(str(found[0]))


@contextmanager
def _ffmpeg_cpu_flags(flags):
    """FFmpeg's process-wide CPU flags forced (0: none, so its C code runs), restored to detection after."""
    lib = _avutil()
    lib.av_force_cpu_flags(flags)
    try:
        yield
    finally:
        lib.av_force_cpu_flags(-1)


def test_frames_pack_to_the_same_bytes_whatever_the_ffmpeg_cpu_flags(tmp_path):
    pictures = _pictures()
    first = _pack(tmp_path / "a.mp4", pictures)
    with _ffmpeg_cpu_flags(0):
        assert _pack(tmp_path / "b.mp4", pictures) == first
    assert _pack(tmp_path / "c.mp4", pictures) == first


def test_frames_pack_to_the_same_bytes_whatever_code_x264_runs(tmp_path, monkeypatch):
    """x264 on its C code alone (asm=0) packs what it packs on this CPU's SIMD code."""
    pictures = _pictures()
    first = _pack(tmp_path / "a.mp4", pictures)
    params = formats.X264_OPTIONS["x264-params"]
    monkeypatch.setitem(formats.X264_OPTIONS, "x264-params", params + ":asm=0")
    assert _pack(tmp_path / "b.mp4", pictures) == first


def test_frames_pack_to_the_same_bytes_from_another_thread_beside_another_pack(tmp_path):
    pictures, other = _pictures(), _pictures(seed=1)
    first = _pack(tmp_path / "a.mp4", pictures)
    got = {}
    workers = [threading.Thread(target=lambda: got.__setitem__("b", _pack(tmp_path / "b.mp4", pictures))),
               threading.Thread(target=lambda: got.__setitem__("o", _pack(tmp_path / "o.mp4", other)))]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    assert got["b"] == first and got["o"] != first


def test_depth_packs_to_the_same_bytes_every_time(tmp_path):
    rng = np.random.default_rng(0)
    depth = [rng.integers(0, 4000, (48, 64), dtype=np.uint16) for _ in range(6)]
    hashes = []
    for k in range(2):
        w = formats.DepthWriter(tmp_path / f"d{k}.mkv")
        for i, a in enumerate(depth):
            w.add(i / 30, a, 0.001)
        w.close()
        hashes.append(hashlib.sha256((tmp_path / f"d{k}.mkv").read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


def _upload(root):
    """An HDF5 upload with a moving colour camera and a depth camera, two recordings."""
    root.mkdir()
    rng = np.random.default_rng(3)
    with h5py.File(root / "a.h5", "w") as h:
        for k, g in enumerate(("demo_0", "demo_1")):
            h[f"data/{g}/obs/front"] = np.stack(_pictures(n=12, h=36, w=64, seed=k))
            h[f"data/{g}/obs/front_depth"] = rng.integers(0, 4000, (12, 64, 80), dtype=np.uint16)
    return root


def _packed(out, report):
    """{episode name: {file name: sha256}} of every file an episode's manifests point to."""
    got = {}
    for row in report["episodes"]:
        ep = out / row["episode_id"]
        files = {}
        for manifest in ("sources.json", "depth.json"):
            if (ep / manifest).exists():
                for entry in json.loads((ep / manifest).read_text()).values():
                    if isinstance(entry, dict) and entry.get("packed"):
                        p = Path(entry["packed"])
                        p = p if p.is_absolute() else ep / p
                        files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        got[row["name"]] = files
    return got


def test_one_upload_prepared_twice_in_one_process_packs_the_same_bytes(tmp_path):
    """What CI's flaky test does, prepare one upload twice in one process, the second time with FFmpeg's CPU flags
    forced off: every packed camera and depth file is the same."""
    original = _upload(tmp_path / "original")
    first = formats.convert(original, "ego_head", tmp_path / "one", "packing", 900)
    with _ffmpeg_cpu_flags(0):
        second = formats.convert(original, "ego_head", tmp_path / "two", "packing", 900)
    assert not first["failed"] and len(first["episodes"]) == 2
    packed = _packed(tmp_path / "one", first)
    assert all(packed.values()) and packed == _packed(tmp_path / "two", second)
    if not any(name.endswith(".mkv") for files in packed.values() for name in files):
        pytest.fail(f"the fixture packed no depth: {packed}")
