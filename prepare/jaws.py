"""Conditional pixel jaw-spacing evidence from a declared two-row marker camera.

Marker response, row calibration and symmetric pairing adapted from Pantheon's
UMI harness, Argus commit f175f6f.
This fixed 640x480 marker profile is not a generic handheld-camera recognizer.
No inference of contact, force, grasp success or object thickness is made.
"""
from __future__ import annotations

import fcntl
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy import ndimage

from label.atomic import write_atomic
from prepare.isolation import run_while_progressing

PROFILE = 'orange_pair_rows_640x480_v1'
W, H = 640, 480
TOP_Y, BOT_Y = (225, 290), (335, 405)
AXIS_X = (295, 350)
BAND = 5
DY = np.arange(-3, 9)
S_MIN, S_MAX = 60, 330
OPEN_K = 19
ROW_TOL = 0.08
GAP_S = 0.5
EVENT_MIN = 0.08
EVENT_WITHIN_S = 0.7
CAL_FRAMES = 48
MAX_FRAMES = 36000
MIN_COVERAGE = 0.8
ALGORITHM = 'orange-pair-native-clock-v1'

def response(img):
    """Compact-orange response over the two bands (float32, full-frame coordinates, zero elsewhere). The opening uses
    a square window, which scipy computes as two fast one-dimensional passes."""
    y0, y1 = TOP_Y[0] - BAND - 10, BOT_Y[1] + BAND + 10
    crop = img[y0:y1].astype(np.float32)
    b, g, r = crop[..., 0], crop[..., 1], crop[..., 2]
    o = np.clip(r - np.maximum(g, b) * 0.9, 0, None) * (r > 60)
    o = ndimage.gaussian_filter(o, 0.8)
    top = o - ndimage.grey_opening(o, size=(OPEN_K, OPEN_K))
    out = np.zeros((H, W), np.float32)
    out[y0:y1] = top
    return out


def _peaks(resp, thr):
    m = (resp == ndimage.maximum_filter(resp, size=9)) & (resp > thr)
    ys, xs = np.nonzero(m)
    return np.stack([xs, ys], 1)


def calibrate(frames) -> dict | None:
    """The clip's two dot rows, its symmetry axis and its response threshold, from frames spread across the clip
    (analyse passes about CAL_FRAMES)."""
    R = [response(im) for im in frames]
    pos = np.concatenate([r[r > 0] for r in R if (r > 0).any()]) if any((r > 0).any() for r in R) else np.array([])
    if pos.size < 50:
        return None
    thr = float(np.percentile(pos, 99.0))
    P = [_peaks(r, thr) for r in R]
    allp = np.vstack([p for p in P if len(p)]) if any(len(p) for p in P) else np.zeros((0, 2))
    rows = []
    for lo, hi in (TOP_Y, BOT_Y):
        h, e = np.histogram(allp[:, 1], bins=np.arange(lo, hi + 4, 4))
        if h.max() < 5:
            return None
        rows.append(float(e[np.argmax(h)] + 2))
    mids = []
    for p in P:
        for y in rows:
            q = p[np.abs(p[:, 1] - y) <= BAND + 2]
            for i in range(len(q)):
                for j in range(i + 1, len(q)):
                    if S_MIN <= abs(q[i, 0] - q[j, 0]) <= S_MAX:
                        mids.append((q[i, 0] + q[j, 0]) / 2)
    mids = np.array([m for m in mids if AXIS_X[0] <= m <= AXIS_X[1]])
    if len(mids) < 5:
        return None
    h, e = np.histogram(mids, bins=np.arange(AXIS_X[0], AXIS_X[1] + 3, 3))
    c0 = e[np.argmax(h)] + 1.5
    return {"rows": rows, "cx": float(np.median(mids[np.abs(mids - c0) <= 4])), "thr": thr}


def _row_pair(respd, y, cal):
    """The symmetric pair with the largest summed response on one row: (spacing px, score) or None."""
    ss = np.arange(S_MIN, S_MAX + 1, 1.0)
    xl, xr = np.round(cal["cx"] - ss / 2).astype(int), np.round(cal["cx"] + ss / 2).astype(int)
    keep = (xl >= 0) & (xr < W)
    ss, xl, xr = ss[keep], xl[keep], xr[keep]
    ys = np.clip(np.round(y + DY).astype(int), 0, H - 1)
    v0, v1 = respd[ys[:, None], xl[None, :]], respd[ys[:, None], xr[None, :]]
    tot = np.where(np.minimum(v0, v1) >= 0.25 * cal["thr"], v0 + v1, 0)
    best = tot.max(0)
    if best.max() <= 0:
        return None
    i = int(best.argmax())
    return float(ss[i]), float(best[i])


def _clock(times_s):
    try:
        times = np.asarray(times_s, dtype=float)
    except (ValueError, TypeError):
        return None
    return times if (times.ndim == 1 and len(times) > 1 and np.isfinite(times).all()
                     and np.all(np.diff(times) > 0)) else None


def eligible(declaration) -> bool:
    """Only a source declaration or independently verified camera geometry opts in."""
    return (isinstance(declaration, dict) and declaration.get('profile') == PROFILE
            and declaration.get('basis') in ('declared', 'verified')
            and isinstance(declaration.get('evidence'), str) and bool(declaration['evidence'].strip()))


def _abstain(reason, **details):
    return {'status': 'abstained', 'reason': reason, 'events': [], **details}


def read_frames(video, *, frame_start, frame_count):
    """Decode only the exact frame-index window, retaining native frame order.

    No FPS filter, frame duplication or geometry scaling is allowed. ffprobe
    validates the declared profile's encoded dimensions before image decoding.
    """
    info = json.loads(subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height', '-of', 'json', str(video)], check=True, capture_output=True,
        text=True, timeout=30).stdout)['streams'][0]
    if (info['width'], info['height']) != (W, H):
        raise ValueError('unsupported_image_geometry')
    vf = f'trim=start_frame={frame_start}:end_frame={frame_start + frame_count}'
    process = subprocess.Popen([
        'ffmpeg', '-nostdin', '-v', 'error', '-noautorotate', '-i', str(video),
        '-map', '0:v:0', '-vf', vf, '-fps_mode', 'passthrough',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, bufsize=W * H * 3 * 4)
    size = W * H * 3
    try:
        while True:
            data = process.stdout.read(size)
            if not data:
                break
            if len(data) != size:
                raise ValueError('partial_decoded_frame')
            yield np.frombuffer(data, np.uint8).reshape(H, W, 3)
        if process.wait() != 0:
            raise ValueError('video_decode_failed')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def opening(sp: np.ndarray, times_s):
    """Filter row agreement and fill short gaps using the caller's native clock."""
    times = np.asarray(times_s, dtype=float)
    top, bot = sp[:, 0], sp[:, 1]
    both = np.isfinite(top) & np.isfinite(bot)
    out = np.full(len(sp), np.nan)
    observed, filled = np.zeros(len(sp), bool), np.zeros(len(sp), bool)
    if both.sum() < 10:
        return out, observed, filled
    if np.ptp(top[both]) < 1:
        predicted = np.full(len(sp), np.median(bot[both]))
    else:
        a, b = np.polyfit(top[both], bot[both], 1)
        predicted = a * top + b
    observed = both & (np.abs(bot - predicted) <= ROW_TOL * np.abs(predicted))
    out[observed] = bot[observed]
    # Keep smoothing inside contiguous measured runs and within a short time window.
    smooth = out.copy()
    for i in np.flatnonzero(observed):
        lo, hi = max(0, i - 2), min(len(out), i + 3)
        window = out[lo:hi]
        if np.isfinite(window).all() and times[hi - 1] - times[lo] <= .2:
            smooth[i] = np.median(window)
    good = np.flatnonzero(observed)
    for left, right in zip(good, good[1:]):
        if right - left > 1 and times[right] - times[left] <= GAP_S:
            smooth[left + 1:right] = np.interp(times[left + 1:right], times[[left, right]], smooth[[left, right]])
            filled[left + 1:right] = True
    return smooth, observed, filled


def events(values: np.ndarray, times_s) -> list[dict]:
    """Report measured spacing changes with native start/end times, never outcomes."""
    values, times = np.asarray(values, float), np.asarray(times_s, float)
    finite = np.isfinite(values)
    if finite.sum() < 10:
        return []
    step = EVENT_MIN * float(np.nanpercentile(values, 3))
    out, i, count = [], 0, len(values)
    while i < count - 1:
        if not finite[i]:
            i += 1
            continue
        stop = i + 1
        while (stop < count and finite[stop] and times[stop] - times[stop - 1] <= GAP_S
               and times[stop] - times[i] <= EVENT_WITHIN_S):
            stop += 1
        segment = values[i:stop]
        low, high = int(np.argmin(segment)), int(np.argmax(segment))
        if values[i] - segment[low] >= step:
            kind, end = 'close', i + low
        elif segment[high] - values[i] >= step:
            kind, end = 'open', i + high
        else:
            i += 1
            continue
        # Extend only through contiguous observations with an actual short settling interval.
        while end + 1 < count:
            later = int(np.searchsorted(times, times[end] + .1, side='left'))
            if (later >= count or not finite[end:later + 1].all()
                    or np.max(np.diff(times[end:later + 1])) > GAP_S):
                break
            moved = values[end] - values[later] if kind == 'close' else values[later] - values[end]
            if moved <= step / 4:
                break
            end = later
        out.append({'kind': kind, 'start_s': float(times[i]), 'end_s': float(times[end]),
                    'from_px': round(float(values[i]), 1), 'to_px': round(float(values[end]), 1)})
        i = end + 1
    return out


def analyse_frames(frame_factory, times_s, declaration):
    """Measure an exact declared window. At most 48 calibration images are retained."""
    if not eligible(declaration):
        return _abstain('unsupported_geometry')
    times = _clock(times_s)
    if times is None:
        return _abstain('invalid_native_clock')
    if len(times) > MAX_FRAMES:
        return _abstain('window_too_large')
    indices = set(np.linspace(0, len(times) - 1, min(CAL_FRAMES, len(times)), dtype=int))
    sample, count = [], 0
    frames = frame_factory()
    try:
        for i, image in enumerate(frames):
            count = i + 1
            if i >= len(times):
                break
            if image.shape != (H, W, 3):
                return _abstain('unsupported_image_geometry')
            if i in indices:
                sample.append(image.copy())
    finally:
        if hasattr(frames, 'close'):
            frames.close()
    if count != len(times):
        return _abstain('frame_count_mismatch', expected_frames=len(times), decoded_frames=count)
    cal = calibrate(sample)
    del sample
    if cal is None:
        return _abstain('markers_not_recognized', quality={'frames': count, 'observed_frames': 0})
    # A second streaming pass applies clip-wide calibration. Only scalar pair
    # measurements survive, so memory is bounded independently of image count.
    frames = frame_factory()
    spacing = []
    try:
        for i, image in enumerate(frames):
            if i >= len(times) or image.shape != (H, W, 3):
                return _abstain('frame_count_mismatch')
            resp = ndimage.maximum_filter(response(image), size=3)
            pairs = [_row_pair(resp, y, cal) for y in cal['rows']]
            spacing.append([p[0] if p else np.nan for p in pairs])
    finally:
        if hasattr(frames, 'close'):
            frames.close()
    if len(spacing) != len(times):
        return _abstain('frame_count_mismatch', expected_frames=len(times), decoded_frames=len(spacing))
    spacing = np.asarray(spacing)
    measured, observed, filled = opening(spacing, times)
    coverage = float(observed.mean())
    quality = {'frames': len(times), 'top_pair_frames': int(np.isfinite(spacing[:, 0]).sum()),
               'bottom_pair_frames': int(np.isfinite(spacing[:, 1]).sum()),
               'observed_frames': int(observed.sum()), 'interpolated_frames': int(filled.sum()),
               'observed_fraction': coverage, 'usable_fraction': float(np.isfinite(measured).mean()),
               'largest_native_gap_s': float(np.max(np.diff(times))),
               'minimum_observed_fraction': MIN_COVERAGE,
               'native_clock_gaps': [{'start_s': float(times[i]), 'end_s': float(times[i + 1])}
                                     for i in np.flatnonzero(np.diff(times) > GAP_S)]}
    if observed.sum() < 10 or coverage < MIN_COVERAGE:
        return _abstain('insufficient_marker_coverage', quality=quality, calibration=cal)
    moves = events(measured, times)
    for event in moves:
        start, end = np.searchsorted(times, [event['start_s'], event['end_s']])
        event['observed_samples'] = int(observed[start:end + 1].sum())
        event['interpolated_samples'] = int(filled[start:end + 1].sum())
    return {'status': 'measured', 'profile': PROFILE, 'units': 'pixels', 'times_s': times.tolist(),
            'spacing_px': [round(float(v), 1) if np.isfinite(v) else None for v in measured],
            'observed': observed.tolist(), 'interpolated': filled.tolist(),
            'events': moves, 'quality': quality, 'calibration': cal,
            'limitations': ['Pixel spacing is relative to this camera geometry.',
                            'Occluded or rejected markers leave unmeasured intervals.',
                            'Jaw motion alone does not establish contact or a task outcome.']}


def analyse(video, *, times_s, camera, declaration, output_dir, source_proof, frame_start=0,
            protected_roots=(), read_only=False):
    """Return output-cached evidence for an exact native-clock source window.

    source_proof is the caller's existing verified source receipt. Source stat and
    path identity also invalidate same-path replacement. output_dir must be outside
    the source video's directory and all caller-supplied prepared roots.
    """
    if not eligible(declaration):
        return _abstain('unsupported_geometry')
    times = _clock(times_s)
    if times is None:
        return _abstain('invalid_native_clock')
    if len(times) > MAX_FRAMES:
        return _abstain('window_too_large')
    if not source_proof or not camera or type(frame_start) is not int or frame_start < 0:
        return _abstain('missing_source_window_identity')
    video, output_dir = Path(video).resolve(), Path(output_dir).resolve()
    if any(output_dir.is_relative_to(Path(root).resolve()) for root in (video.parent, *protected_roots)):
        raise ValueError('Jaw output must be outside the source directory.')
    try:
        stat = video.stat()
    except OSError:
        return _abstain('source_unavailable')
    identity = {'algorithm': ALGORITHM, 'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'source_proof': source_proof, 'source_location_sha256': hashlib.sha256(str(video).encode()).hexdigest(),
                'source_stat': [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino],
                'camera': camera, 'declaration': declaration, 'frame_start': frame_start,
                'frame_count': len(times), 'times_sha256': hashlib.sha256(times.astype('<f8').tobytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not read_only:
        output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{key}.json'
    with (nullcontext() if read_only else (output_dir / f'{key}.lock').open('a')) as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            try:
                cached = json.loads(path.read_text())
                if isinstance(cached, dict) and cached.get('identity') == identity and cached.get('cache_key') == key:
                    current = video.stat()
                    if [current.st_size, current.st_mtime_ns, current.st_ctime_ns, current.st_ino] == identity['source_stat']:
                        return cached
                    return _abstain('source_changed_during_analysis')
            except (OSError, ValueError):
                pass
        try:
            result = analyse_frames(lambda: read_frames(video, frame_start=frame_start, frame_count=len(times)),
                                    times, declaration)
        except (ValueError, OSError, subprocess.SubprocessError):
            # Exception messages can name private paths; retain a bounded public reason.
            return _abstain('video_decode_failed')
        try:
            after = video.stat()
        except OSError:
            return _abstain('source_changed_during_analysis')
        if [after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino] != identity['source_stat']:
            return _abstain('source_changed_during_analysis')
        result.update(cache_key=key, identity=identity)
        if not read_only:
            write_atomic(path, result, allow_nan=False)
        return result


def for_episode(ep: dict, output_dir: Path | None, *, read_only=False) -> dict:
    """Measure explicitly qualified cameras outside immutable prepared/source folders."""
    ctx = ep['context']
    declarations = {view: entry.get('jaw_geometry') for view, entry in (ctx.get('cameras') or {}).items()
                    if isinstance(entry, dict) and entry.get('jaw_geometry')}
    if not declarations:
        return {}
    from label.evidence_access import source_proof
    result = {}
    for view, declaration in declarations.items():
        source = (ep.get('sources') or {}).get(view)
        times = (ep.get('times') or {}).get(view)
        pts = (ep.get('times') or {}).get(view + '_pts')
        if not eligible(declaration):
            result[view] = _abstain('unsupported_geometry')
            continue
        if (output_dir is None or not source or times is None or pts is None
                or view in (ctx.get('camera_clock') or {})
                or len(times) != source['n_frames'] or len(pts) != len(times)):
            result[view] = _abstain('unverified_source_window_or_native_clock')
            continue
        if len(times) > MAX_FRAMES:
            result[view] = _abstain('window_too_large')
            continue
        path = Path(source['packed'])
        if not path.is_absolute():
            path = Path(ep['dir']) / path
        try:
            # ffprobe reports actual decoded frame PTS. Packet counts or nominal FPS
            # cannot establish a frame-index window for variable-rate sources.
            # it decodes every frame, so it is bound by its progress (each frame it reports), not by a fixed time
            probe = run_while_progressing(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_frames', '-show_entries', 'frame=pts', '-of', 'json', str(path)], check=True, text=True)
            decoded = [frame.get('pts') for frame in json.loads(probe.stdout)['frames']]
            starts = [index for index, value in enumerate(decoded) if value == int(pts[0])]
            if len(starts) != 1 or decoded[starts[0]:starts[0] + len(pts)] != [int(value) for value in pts]:
                result[view] = _abstain('source_frame_window_mismatch')
                continue
            roots = [Path(ep['dir'])]
            if ctx.get('retained_evidence_root'):
                roots.append(Path(ctx['retained_evidence_root']))
            proof = source_proof(ctx, ep['dir'])
            record = analyse(path, times_s=times, camera=view, declaration=declaration,
                output_dir=output_dir, source_proof=proof,
                frame_start=starts[0], protected_roots=roots, read_only=read_only)
            result[view] = {key: value for key, value in record.items() if key != 'identity'}
            result[view]['source_proof'] = proof
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            result[view] = _abstain('source_frame_window_unavailable')
    return result


def prompt(records: dict) -> str:
    """Qualified pixel spacing observations with no contact or outcome inference."""
    if not records:
        return ''
    lines = ['\nJAW CAMERA EVIDENCE. These are relative marker-spacing changes in pixels. '
             'They do not establish pressure, physical jaw width, contact, or grasp success. '
             'Judge the activity and outcome from the RGB scene.']
    for view, record in records.items():
        if record.get('status') != 'measured':
            lines.append(f"{view}: measurement unavailable ({record.get('reason')}).")
            continue
        quality = record.get('quality') or {}
        lines.append(f"{view}: directly observed fraction {quality.get('observed_fraction')}; "
                     'short interpolated intervals remain separately qualified in the measurement receipt.')
        for event in record.get('events') or []:
            lines.append(f"{view}: marker spacing {event['kind']}, {event['start_s']:.2f} to "
                         f"{event['end_s']:.2f} s, {event['from_px']:.1f} to {event['to_px']:.1f} px, "
                         f"{event['observed_samples']} observed samples, "
                         f"{event['interpolated_samples']} interpolated samples.")
    return '\n'.join(lines) + '\n'
