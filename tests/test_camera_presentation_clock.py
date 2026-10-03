"""Distinct camera frames remain visible when their recorded timestamps repeat."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from board import clips, sensors
from label import episode, frames
from prepare import formats


def tied_recording(tmp_path, pattern, n=60, topics=None):
    from mcap.reader import make_reader
    from mcap.writer import Writer
    from test_formats import _camera_mcap
    source = tmp_path / "source.mcap"
    topics = topics or ["/top/image/compressed", "/left/image/compressed"]
    _camera_mcap(source, topics, n=n)
    with source.open("rb") as stream:
        rows = list(make_reader(stream).iter_messages())
    upload = tmp_path / "upload"
    upload.mkdir()
    path = upload / "recording.mcap"
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        ids = {}
        for index, (schema, channel, message) in enumerate(rows):
            if channel.topic not in ids:
                sid = writer.register_schema(name=schema.name, encoding=schema.encoding, data=schema.data)
                ids[channel.topic] = writer.register_channel(topic=channel.topic, message_encoding=channel.message_encoding,
                                                            schema_id=sid)
            k = index // len(topics)
            offset = k / 30 if pattern == "unique" else (k // 2) / 30 if pattern == "paired" else 0.0
            stamp = 1790000000000000000 + int(round(offset * 1e9))
            writer.add_message(ids[channel.topic], log_time=stamp, publish_time=stamp, data=message.data)
        writer.finish()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = formats.convert(upload, "teleop_arms", tmp_path / "episodes", "test", 900)
    assert not report["failed"] and len(report["episodes"]) == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    ep = tmp_path / "episodes" / report["episodes"][0]["episode_id"]
    return ep


@pytest.mark.parametrize("pattern", ["paired", "all"])
def test_tied_capture_frames_use_a_qualified_clock_without_changing_raw_arrays(tmp_path, pattern, monkeypatch):
    ep = tied_recording(tmp_path, pattern)
    context = json.loads((ep / "context.json").read_text())
    assert context["presentation_times"] == "presentation_times.npz"
    with np.load(ep / context["recorded_camera_ns"]) as ns:
        assert ns["exo"].dtype == np.int64 and len(ns["exo"]) == 60
        assert ns["exo"][0] == 1790000000000000000
    with np.load(ep / "times.npz") as raw, np.load(ep / context["presentation_times"]) as shown:
        expected = np.repeat(np.arange(30) / 30, 2) if pattern == "paired" else np.zeros(60)
        np.testing.assert_allclose(raw["exo"], expected, atol=1e-6)
        assert len(raw["exo"]) == len(raw["exo_pts"]) == 60
        assert (np.diff(raw["exo_pts"]) > 0).all()
        assert (np.diff(shown["exo"]) > 0).all()
        assert shown["exo"][-1] > 0
        original = {key: raw[key].copy() for key in raw.files}
    import av
    source = json.loads((ep / "sources.json").read_text())
    expected = {}
    for view, item in source.items():
        with av.open(item["packed"]) as video:
            expected[view] = [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
    decoded = []
    original_decode = episode._decode_view

    def trace(unit, view, ks, *args, **kwargs):
        got = original_decode(unit, view, ks, *args, **kwargs)
        mapping = unit["kmap"].get(view)
        for k, image in got.items():
            own = int(mapping[k]) if mapping is not None else k
            np.testing.assert_array_equal(np.asarray(image), expected[view][own])
            decoded.append((view, k, own))
        return got

    monkeypatch.setattr(episode, "_decode_view", trace)
    request = episode.build_request(ep)
    assert {(view, k) for view, k, _ in decoded} == {(view, k) for view in source for k in request["plan"]["ks"]}
    assert np.all(np.diff(request["timesteps"]) > 0)
    assert "assumed presentation" in request["prompt"]
    assert "The times are exact: use them, do not invent your own." not in request["prompt"]
    assert "headed with its exact time" not in request["prompt"]
    loaded = episode.load(ep)
    np.testing.assert_allclose(sensors.clip_times(ep, context, 60), loaded["times"]["exo"])
    jobs = list(clips.episode_jobs(ep, tmp_path / "clips", True))
    assert len(jobs) == 2
    for job in jobs:
        assert clips.extract_one(*job[:4], clips.find_ffmpeg(), 1, *job[4:12]) is None
        assert clips.clip_frames(job[3]) == 60
        view = next(v for v, s in source.items() if s["packed"] == str(job[0]))
        with av.open(str(job[3])) as video:
            board_frames = list(video.decode(video=0))
            starts = np.array([float(frame.pts * frame.time_base) for frame in board_frames])
        for index, frame in enumerate(board_frames):
            np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), expected[view][index])
        shown_at = np.searchsorted(starts, loaded["times"]["exo"], side="right") - 1
        np.testing.assert_array_equal(shown_at, np.arange(60))
    with np.load(ep / "times.npz") as raw:
        for key, value in original.items():
            np.testing.assert_array_equal(raw[key], value)
    assert "fallback" in context["camera_clock"]["exo"]["what"] if pattern == "all" else "interval" in context["camera_clock"]["exo"]["what"]


def test_a_unique_clock_keeps_the_existing_request_and_has_no_presentation_sidecar(tmp_path):
    ep = tied_recording(tmp_path, "unique")
    context = json.loads((ep / "context.json").read_text())
    assert "presentation_times" not in context and "camera_clock" not in context
    assert not (ep / "presentation_times.npz").exists()
    assert "The times are exact: use them, do not invent your own." in episode.build_request(ep)["prompt"]


def test_an_unmarked_old_tied_episode_keeps_its_label_clock(tmp_path):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.repeat(np.arange(45) / 30, 2))
    ctx = json.loads((ep / "context.json").read_text())
    ctx.pop("presentation_times", None)
    ctx.pop("camera_clock", None)
    ctx.pop("reader_issues", None)
    ctx["fps"] = 30
    ctx["duration_s"] = 1.467
    for signal in ctx.get("signals") or []:
        signal.pop("aligned_by", None)
        signal.pop("camera_aligned_by", None)
    (ep / "context.json").write_text(json.dumps(ctx))
    (ep / "presentation_times.npz").unlink(missing_ok=True)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ep.iterdir() if p.is_file()}
    request = episode.build_request(ep)
    assert request["timesteps"] == pytest.approx([0.0, 0.733, 1.467], abs=0.001)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ep.iterdir() if p.is_file()}


def test_a_tied_camera_clock_never_becomes_measured_state_timing(tmp_path):
    from test_reader_tables import _clip
    media = tmp_path / "exo.mp4"
    _clip(media, 60)
    pr = formats.probe(media)
    raw = np.repeat(np.arange(30) / 30, 2)
    state = np.arange(60 * 14, dtype=np.float32).reshape(60, 14) / 1000
    context = {"episode_id": "episode_a", "profile": "teleop_arms", "state_kind": "joints", "fps": 30,
               "n_state_frames": 60, "cameras": {"exo": {"key": "top", "name": "top"}}}
    source = {"exo": {"packed": str(media), "base_s": 0, "n_frames": 60, "camera_key": "top"}}
    ep = tmp_path / "episode_a"
    context = formats.finish_episode(ep, context, source, state=state, action=state + 1,
                                     times={"exo": raw, "exo_pts": pr["pts"]})
    assert context["state_kind"] == "none" and context["state_why"] == "assumed_clock"
    with np.load(ep / "state.npz") as original:
        np.testing.assert_array_equal(original["state"], state)
        np.testing.assert_array_equal(original["action"], state + 1)
    assert any(s["name"] == "recorded state" and s["aligned_by"] == "assumed camera clock" for s in context["signals"])
    request = episode.build_request(ep)
    assert "RECORDED MOTION" not in request["prompt"]
    assert "assumed camera" in request["prompt"]


def test_the_last_singleton_has_a_positive_tail_and_all_ties_name_their_fallback():
    from prepare.camera_clock import presentation_clock
    placed, note = presentation_clock(np.array([0, 0, 0.2]))
    np.testing.assert_array_equal(placed, [0, 0.1, 0.2])
    assert note["resolution_s"] == 0.2
    placed, note = presentation_clock(np.zeros(3), nominal_fps=25)
    np.testing.assert_array_equal(placed, [0, 0.04, 0.08])
    assert note["nominal_fps"] == 25 and note["cadence_source"] == "declared nominal cadence"
    placed, note = presentation_clock(np.zeros(3))
    assert placed[-1] > 0 and note["cadence_source"] == "unknown cadence fallback"


def test_contacts_inherit_the_camera_assumption_without_claiming_a_common_start():
    from label.contacts import mark_aligned
    from checks.contacts import check
    contact = {"id": "c1", "signals": ["force"], "start_s": 0.1, "end_s": 0.4, "peak_s": 0.2}
    marked = mark_aligned([contact], {"force": {"aligned_by": "coarse clock", "camera_aligned_by": "assumed camera clock"}})
    assert marked[0]["aligned_by"] == "assumed camera clock"
    assert "assumed camera presentation clock" in episode._contact_line(marked[0])
    assert "both starts" not in episode._contact_line(marked[0])
    checks = check({"contacts": [{"id": "c1", "touch_seen": "no"}]}, marked, {}, 30)
    assert checks["placed_on_assumed_camera_clock"] == ["c1"]
    assert "placed_from_both_starts" not in checks
    assert not checks["notes"]


def test_parts_and_trim_keep_source_pixels_and_recorded_clocks(tmp_path, monkeypatch):
    from label import pieces
    ep = tied_recording(tmp_path, 'paired')
    original = episode.load(ep)
    raw = {k: v.copy() for k, v in original['recorded_times'].items()}
    monkeypatch.setattr(pieces, 'choose_cuts', lambda *args: [{'frame': 20}, {'frame': 40}])
    parts = pieces.write_pieces(ep, tmp_path / 'parts')
    assert len(parts) == 3
    for i, part in enumerate(parts):
        unit = episode.load(part)
        offset = original['times']['exo'][i * 20]
        for view in episode.views(unit):
            np.testing.assert_array_equal(unit['recorded_times'][view] + offset, raw[view][i * 20:(i + 1) * 20])
            np.testing.assert_array_equal(unit['times'][view + '_pts'], raw[view + '_pts'][i * 20:(i + 1) * 20])
            got = episode._decode_view(unit, view, [0, 19])
            want = episode._decode_view(original, view, [i * 20, i * 20 + 19])
            for k in (0, 19):
                np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(want[i * 20 + k]))
        assert 'assumed presentation' in episode.build_request(part)['prompt']
    with np.load(ep / 'times.npz') as z:
        for k in raw:
            np.testing.assert_array_equal(z[k], raw[k])
    formats.trim_episode(ep, 0.5)
    unit = episode.load(ep)
    assert len(unit['state']) == 30
    for view in episode.views(unit):
        np.testing.assert_array_equal(unit['recorded_times'][view], raw[view][:30])
        np.testing.assert_array_equal(unit['times'][view + '_pts'], raw[view + '_pts'][:30])


def test_raw_clock_checks_remain_fired_for_a_qualified_display_clock(tmp_path):
    from checks import capture_qc
    ep = tied_recording(tmp_path, 'all')
    result = capture_qc.assess(capture_qc.extract(ep))
    assert result['episode']['clock'] == 'assumed camera presentation'
    assert result['checks']['state_time_non_monotonic_or_duplicate']['status'] == 'fired'
    assert all(result['checks'][k]['status'] == 'not_assessed' for k in capture_qc.CAMERA_MOTION_CHECKS)


def test_a_tied_replacement_anchor_cannot_make_state_timing_measured(tmp_path):
    from test_reader_tables import _clip
    media = tmp_path / 'exo.mp4'
    _clip(media, 60)
    pr = formats.probe(media)
    state = np.arange(60 * 14, dtype=np.float32).reshape(60, 14)
    ctx = {'episode_id': 'episode_a', 'profile': 'teleop_arms', 'state_kind': 'joints', 'fps': 30,
           'n_state_frames': 60, 'cameras': {'exo': {'key': 'top'}, 'left': {'key': 'left'}}}
    src = {v: {'packed': str(media), 'base_s': 0, 'n_frames': 60, 'camera_key': v} for v in ('exo', 'left')}
    ep = tmp_path / 'episode_a'
    formats.finish_episode(ep, ctx, src, state=state,
                           times={'exo': np.arange(60) / 30, 'left': np.repeat(np.arange(30) / 15, 2),
                                  'exo_pts': pr['pts'], 'left_pts': pr['pts']})
    assert ctx['state_kind'] == 'joints'
    clips.drop_cameras(ep, ['exo'])
    context = json.loads((ep / 'context.json').read_text())
    assert context['state_kind'] == 'none' and context['state_why'] == 'assumed_clock'
    assert any(s['name'] == 'recorded state' and s['camera_aligned_by'] == 'assumed camera clock'
               for s in context['signals'])
    assert 'RECORDED MOTION' not in episode.build_request(ep)['prompt']


def test_contact_rows_use_qualified_presentation_instants_and_preserve_sensor_assumptions(tmp_path):
    from test_reader_tables import _clip
    media = tmp_path / 'exo.mp4'
    _clip(media, 60)
    pr = formats.probe(media)
    sig = formats.Signals({'left pressure': np.r_[np.zeros(30), np.linspace(1, 3, 30)][:, None]})
    sig.meta['left pressure'] = {'aligned_by': 'coarse clock', 'unknown_sensor_field': {'a': 1}, 'rest': [0], 'swing': 3}
    ctx = {'episode_id': 'episode_a', 'profile': 'teleop_arms', 'state_kind': 'none', 'fps': 30,
           'n_state_frames': 60, 'cameras': {'exo': {'key': 'top'}}}
    src = {'exo': {'packed': str(media), 'base_s': 0, 'n_frames': 60, 'camera_key': 'top'}}
    ep = tmp_path / 'episode_a'
    formats.finish_episode(ep, ctx, src, signals=sig, times={'exo': np.zeros(60), 'exo_pts': pr['pts']})
    formats.measure_contacts(tmp_path, [ep.name])
    ctx = json.loads((ep / 'context.json').read_text())
    meta = ctx['signals'][0]
    assert meta['aligned_by'] == 'coarse clock' and meta['camera_aligned_by'] == 'assumed camera clock'
    assert meta['unknown_sensor_field'] == {'a': 1}
    contact = ctx['contacts'][0]
    assert contact['start_s'] > 0.5 and contact['end_s'] > 1.5
    assert contact['aligned_by'] == 'assumed camera clock'


@pytest.mark.parametrize("pattern", ["paired", "all"])
def test_unshown_tied_cameras_keep_raw_ns_and_play_every_frame_on_the_named_clock(tmp_path, pattern):
    import av
    ep = tied_recording(tmp_path, pattern, topics=["/top/image/compressed", "/left/image/compressed",
                                                  "/thermal/image/compressed"])
    ctx = json.loads((ep / "context.json").read_text())
    unshown = ctx["unshown_cameras"][0]
    assert unshown["camera_clock"] and "assumed presentation" in unshown["why"]
    with np.load(ep / unshown["camera_times"]) as z:
        assert z["capture_ns"].dtype == np.int64 and len(z["capture_ns"]) == 60
        assert (np.diff(z["pts"]) > 0).all() and (np.diff(z["presentation"]) > 0).all()
        presentation = z["presentation"].copy()
    with av.open(unshown["packed"]) as video:
        original = [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
    job = next(j for j in clips.episode_jobs(ep, tmp_path / "clips", True) if j[-1] == "unshown1")
    assert clips.extract_one(*job[:4], clips.find_ffmpeg(), 1, *job[4:12]) is None
    with av.open(str(job[3])) as video:
        board = list(video.decode(video=0))
    assert len(board) == len(original) == 60
    for frame, expected in zip(board, original):
        np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), expected)
    starts = np.array([float(frame.pts * frame.time_base) for frame in board])
    np.testing.assert_array_equal(np.searchsorted(starts, presentation, side="right") - 1, np.arange(60))


@pytest.mark.parametrize('pattern', ['paired', 'all'])
def test_unshown_parts_select_the_original_packets_without_changing_capture_arrays(tmp_path, monkeypatch, pattern):
    import av
    from label import pieces
    ep = tied_recording(tmp_path, pattern, topics=['/top/image/compressed', '/thermal/image/compressed'])
    ctx = json.loads((ep / 'context.json').read_text())
    unshown = ctx['unshown_cameras'][0]
    clock_path = ep / unshown['camera_times']
    before = hashlib.sha256(clock_path.read_bytes()).hexdigest()
    with av.open(unshown['packed']) as video:
        original = [f.to_ndarray(format='rgb24') for f in video.decode(video=0)]
    monkeypatch.setattr(pieces, 'choose_cuts', lambda *args: [{'frame': 20}, {'frame': 40}])
    parts = pieces.write_pieces(ep, tmp_path / 'parts')
    spans = []
    for i, part in enumerate(parts):
        context = json.loads((part / 'context.json').read_text())
        entry = context['unshown_cameras'][0]
        spans.append(entry['frame_span'])
        job = next(j for j in clips.episode_jobs(part, tmp_path / 'clips', True) if j[-1] == 'unshown1')
        assert clips.extract_one(*job[:4], clips.find_ffmpeg(), 1, *job[4:12]) is None
        with av.open(str(job[3])) as video:
            got = list(video.decode(video=0))
        assert len(got) == 20
        for k, frame in enumerate(got):
            np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'), original[i * 20 + k])
        starts = np.array([float(frame.pts * frame.time_base) for frame in got])
        np.testing.assert_array_equal(np.searchsorted(starts, episode.load(part)['times']['exo'], side='right') - 1,
                                      np.arange(20))
    assert spans == [[0, 20], [20, 40], [40, 60]]
    assert hashlib.sha256(clock_path.read_bytes()).hexdigest() == before


def test_a_final_single_frame_part_keeps_its_original_packet(tmp_path, monkeypatch):
    import av
    from label import pieces
    ep = tied_recording(tmp_path, 'all', topics=['/top/image/compressed', '/thermal/image/compressed'])
    monkeypatch.setattr(pieces, 'choose_cuts', lambda *args: [{'frame': 59}])
    last = pieces.write_pieces(ep, tmp_path / 'parts')[-1]
    ctx = json.loads((last / 'context.json').read_text())
    assert ctx['duration_s'] > 0 and ctx['unshown_cameras'][0]['frame_span'] == [59, 60]
    source = episode.load(ep)
    expected = episode._decode_view(source, 'exo', [59])[59]
    for job in clips.episode_jobs(last, tmp_path / 'clips', True):
        assert clips.extract_one(*job[:4], clips.find_ffmpeg(), 1, *job[4:12]) is None
        with av.open(str(job[3])) as video:
            frames = list(video.decode(video=0))
        assert len(frames) == 1
        np.testing.assert_array_equal(frames[0].to_ndarray(format='rgb24'), np.asarray(expected))
