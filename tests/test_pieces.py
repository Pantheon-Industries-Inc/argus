"""Labelling a long recording in parts (label/pieces.py): where it is cut, what each part holds, and how the parts'
labels are stitched back into one timeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from label import episode as me
from label import harness, pieces
from prepare import formats as f
from test_formats import recorder_folder
import complete_reply


def test_only_a_recording_longer_than_the_limit_is_cut(tmp_path):
    for secs, cut in ((450.0, False), (450.001, True), (472.0, True)):
        d = tmp_path / f"episode_{secs}"
        d.mkdir()
        (d / "context.json").write_text(json.dumps({"profile": "teleop_arms", "duration_s": secs}))
        assert pieces.needs_pieces(d) is cut, secs


@pytest.mark.parametrize('variable', [False, True])
def test_fallback_camera_parts_keep_full_footage_and_unaligned_source_rows(inspect_every_field, tmp_path, variable):
    """Losing the short anchor must not omit the longer camera's tail or invent sensor alignment."""
    import base64
    import io
    from PIL import Image
    from board import clips
    from board.build import episode_seconds
    from label import policy
    from test_formats import _clip
    from test_prepare import joint_names

    upload = tmp_path / 'upload'
    upload.mkdir()
    old_video, surviving_video = upload / 'top.mp4', upload / 'wrist_left.mp4'
    _clip(old_video, 30)
    import av
    from fractions import Fraction
    ticks = np.arange(90) if not variable else np.r_[np.arange(45) * 2, 90 + np.arange(45) * 3]
    rate = 30 if variable else 15
    source_times = ticks / rate
    source_end = float((ticks[-1] + 1) / rate)
    with av.open(str(surviving_video), 'w') as container:
        stream = container.add_stream('mpeg4', rate=rate)
        stream.width, stream.height, stream.pix_fmt = 64, 36, 'yuv420p'
        for row in range(90):
            frame = av.VideoFrame.from_ndarray(np.full((36, 64, 3), row * 9 % 256, np.uint8), format='rgb24')
            frame.pts, frame.time_base = int(ticks[row]), Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    episodes = tmp_path / 'episodes'
    source = episodes / 'episode_fallback'
    state = np.tile(np.arange(30, dtype=np.float32)[:, None], (1, 14))
    action = state + 100
    signal = np.arange(30, dtype=np.int64)[:, None]
    f.video_views_episode(source, {'exo': ('top', old_video), 'left': ('wrist_left', surviving_video)},
                         'teleop_arms', 'fixture', {},
                         real={'exo': np.arange(30) / 30, 'left': source_times},
                         state=state, action=action, state_names=joint_names(), signals={'force': signal})
    context = json.loads((source / 'context.json').read_text())
    native_values = np.stack([np.arange(30, dtype=np.uint64) + np.uint64(2**60),
                              np.arange(30, dtype=np.uint64) + np.uint64(2**60 + 1)], axis=1)
    present = np.ones(30, dtype=bool)
    present[7] = False
    native_frames = np.arange(30, dtype=np.int64) * 2 + 7
    signals = f.Signals({'force': signal})
    signals.native_table_fields = [{'name': 'opaque_counters', 'values': native_values[present],
                                   'source_rows': np.flatnonzero(present), 'source_row_count': 30,
                                   'source_shape': [2], 'metadata': {}, 'bookkeeping': False}]
    signals.native_table_frames = native_frames
    f.write_signals(source, context, signals)
    assert context['duration_s'] == 1.0
    context['annotation_policy'] = {**policy.resolve(context), 'part_max_s': 1.0}
    context.pop('real_times', None)
    (source / 'context.json').write_text(json.dumps(context))
    (source / 'times.npz').unlink()
    old_video.write_bytes(b'fixture camera cannot decode')
    # A timestampless survivor cannot establish its origin on an existing label clock.
    labelled_context = {**context, 'clock_zero_s': 1.25, 'clock_start_s': 12.0}
    (source / 'context.json').write_text(json.dumps(labelled_context))
    saved_context = (source / 'context.json').read_bytes()
    saved_sources = (source / 'sources.json').read_bytes()
    with pytest.raises(ValueError, match='cannot preserve the existing label clock'):
        clips.drop_cameras(source, ['exo'], keep_clock=True)
    assert (source / 'context.json').read_bytes() == saved_context
    assert (source / 'sources.json').read_bytes() == saved_sources
    assert not (source / 'surviving_video_times.npz').exists()
    (source / 'context.json').write_text(json.dumps(context))
    main, issues = clips.drop_cameras(source, ['exo'])
    assert main == 'left' and any(issue['kind'] == 'state_unaligned' for issue in issues)
    whole = me.load(source)
    assert me.request_duration(whole) == pytest.approx(source_end) and whole['context']['state_unaligned']
    assert episode_seconds(whole['context'], source) == pytest.approx(source_end)
    assert np.allclose(whole['times']['left'], source_times)
    assert whole['context']['n_state_frames'] == 90

    def check_board_clip(directory, expected_frames, expected_end):
        (clip_job,) = clips.episode_jobs(directory, tmp_path / 'board_clips' / directory.name, True)
        assert clips.extract_one(*clip_job[:4], clips.find_ffmpeg(), 1, *clip_job[4:13]) is None
        with av.open(str(clip_job[3])) as encoded:
            video = encoded.streams.video[0]
            packets = [packet for packet in encoded.demux(video) if packet.size and packet.pts is not None]
            actual_end = max(float((packet.pts + packet.duration) * video.time_base) for packet in packets)
        assert len(packets) == expected_frames
        assert actual_end == pytest.approx(expected_end, abs=2e-6)
        with av.open(str(clip_job[3])) as encoded:
            images = [frame.to_ndarray(format='rgb24') for frame in encoded.decode(video=0)]
        original = me._decode_view(me.load(directory), 'left', [0, expected_frames - 1])
        for image, index in zip((images[0], images[-1]), (0, expected_frames - 1)):
            assert abs(image.mean() - np.asarray(original[index]).mean()) <= 3

    check_board_clip(source, 90, source_end)
    # Previously prepared fallback sidecars can still carry the old anchor's length.
    legacy = tmp_path / 'legacy_fallback'
    legacy.mkdir()
    legacy_context = {**whole['context'], 'duration_s': 1.0, 'n_state_frames': 30}
    legacy_context.pop('real_times', None)
    (legacy / 'context.json').write_text(json.dumps(legacy_context))
    (legacy / 'sources.json').write_text((source / 'sources.json').read_text())
    assert pieces.needs_pieces(legacy)

    job = tmp_path / 'job'
    job.mkdir()
    units = pieces.write_units(job, episodes)
    assert len(units[source.name]) == int(np.ceil(source_end))
    parts = [job / 'units' / name for name in units[source.name]]
    assert sum(me.load(part)['sources']['left']['n_frames'] for part in parts) == 90
    covered = []
    for part in parts:
        episode = me.load(part)
        count = episode['sources']['left']['n_frames']
        base = int(np.searchsorted(source_times, episode['context']['piece']['t0_s'] - .0005))
        covered.extend(range(base, base + count))
        assert episode['context']['state_unaligned']
        assert np.array_equal(episode['state'], state) and np.array_equal(episode['action'], action)
        assert np.array_equal(np.load(part / 'signals.npz')['s0'], signal)
        assert (part / 'state.npz').samefile(source / 'state.npz')
        assert (part / 'signals.npz').samefile(source / 'signals.npz')
        request = me.build_request(part, cell_w=64, inspect_evidence=True)
        assert me.request_duration(episode) <= 1.0 + 1e-9
        expected_end = (source_times[base + count] if base + count < 90 else source_end) - source_times[base]
        check_board_clip(part, count, float(expected_end))
        assert not request['plan']['state_usable'] and not request.get('contacts')
        assert request['timesteps'][0] == 0.0
        assert request['timesteps'][-1] == pytest.approx(source_times[base + count - 1] - source_times[base], abs=.001)
        assert 'state' not in request['blocks'] and 'signals' not in request['blocks']
        got = me._decode_view(episode, 'left', list(range(count)))
        want = me._decode_view(whole, 'left', list(range(base, base + count)))
        assert len(got) == count
        for row in range(count):
            assert np.array_equal(np.asarray(got[row]), np.asarray(want[base + row]))
        sent_images = [item for item in request['content'] if item['type'] == 'image_url']
        for item, row in zip(sent_images[-2:], (base, base + count - 1)):
            sent = Image.open(io.BytesIO(base64.b64decode(item['image_url']['url'].split(',', 1)[1])))
            # Compare assembled request image pixels below the caption, allowing JPEG rounding.
            assert abs(np.asarray(sent)[-16:, 16:48].mean() - np.asarray(want[row]).mean()) <= 3
        access = request['evidence_access']
        inspected = set()
        for field in access.inventory():
            if field['name'] not in ('state', 'force', 'opaque_counters'):
                continue
            receipt = access.inspect({'field_id': field['id'], 'mode': 'samples',
                                      'start_row': 0, 'end_row': 30, 'columns': [0]})
            inspected.add(field['name'])
            assert receipt['rows'] == list(range(30))
            assert 'times_s' not in receipt
            if field['name'] == 'opaque_counters':
                assert receipt['native_dtype'] == 'uint64'
                assert receipt['source_present'] == present.tolist()
                assert receipt['source_frame_indices'] == native_frames.tolist()
                assert receipt['values'] == [[int(value)] if valid else [None]
                                             for value, valid in zip(native_values[:, 0], present)]
            else:
                assert np.array_equal(np.asarray(receipt['values']).ravel(), np.arange(30))
            assert any('no verified episode alignment' in note for note in receipt['limitations'])
        assert inspected == {'state', 'force', 'opaque_counters'}
    assert covered == list(range(90))


def test_cuts_land_at_the_stillest_moment_near_each_target():
    t = np.arange(0, 1000, 0.1)
    m = np.ones_like(t)
    for still in (310.0, 690.0):                       # the only still stretches, near the thirds of 1000 s
        m[(t > still - 3) & (t < still + 3)] = 0.0
    cuts = pieces.choose_cuts(t, m, 450.0)
    assert len(cuts) == 2 and all(c["still"] for c in cuts)
    assert abs(cuts[0]["t_s"] - 310) <= 3 and abs(cuts[1]["t_s"] - 690) <= 3        # inside the still stretches
    assert pieces.choose_cuts(t[:4000], m[:4000], 450.0) == []          # 400 s fits in one request


def test_each_part_reads_the_recording_at_its_own_span(monkeypatch):
    """Every frame of the recording is in exactly one part, each part decodes the same pixels as the recording at
    the same moment on every camera, its state is the recording's, and it is told it is one part of a recording,
    with the recording's task text as context rather than its goal."""
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "upload"
        recorder_folder(root, n=90, duplicate_stamps=False)
        rep = f.convert(root, "teleop_arms", Path(t) / "eps", "test", 900)
        src = Path(t) / "eps" / rep["episodes"][0]["episode_id"]
        whole = me.load(src)
        n = int(whole["sources"]["exo"]["n_frames"])
        parts = pieces.write_pieces(src, Path(t) / "pieces")
        assert [p.name for p in parts] == [f"{src.name}__p0{i}" for i in (1, 2, 3)]
        assert json.loads((src / "context.json").read_text())["pieces"]["parts"] == [p.name for p in parts]
        eps = [me.load(p) for p in parts]
        assert sum(int(e["sources"]["exo"]["n_frames"]) for e in eps) == n
        a = 0
        for i, e in enumerate(eps, 1):
            m = int(e["sources"]["exo"]["n_frames"])
            ctx = e["context"]
            assert ctx["duration_s"] <= 1.0
            assert ctx["piece"]["t1_s"] - ctx["piece"]["t0_s"] <= 1.0
            assert ctx["piece"]["index"] == i and ctx["piece"]["count"] == 3 and ctx["piece"]["of"] == src.name
            # each part keeps the recording's instruction to check against, on what that part shows
            assert ctx["instruction"] == "Pick up the cube"
            assert "check this part against it only on what this part shows" in ctx["collection_note"]
            assert ctx["collection_note"].startswith(f"this clip is part {i} of 3 of one continuous")
            request = me.build_request(parts[i - 1])
            assert "nothing was trimmed" not in request["prompt"]
            assert f"part {i} of 3" in request["prompt"]
            assert "Source metadata may describe the full recording" in request["prompt"]
            assert np.array_equal(e["state"], whole["state"][a:a + m])
            ks = [0, m - 1]
            for v in me.views(whole):
                got = me._decode_view(e, v, ks)
                want = me._decode_view(whole, v, [a + k for k in ks])
                for k in ks:
                    assert np.array_equal(np.asarray(got[k]), np.asarray(want[a + k])), (i, v, k)
            a += m


def test_stitching_puts_the_parts_back_on_the_recordings_clock(monkeypatch):
    """Times move onto the recording's clock, local outcomes retain request scope, and a
    cut-off issue at one of our cuts is set aside with the reason, while one at the recording's real end is kept."""
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.5)
    monkeypatch.setattr(pieces, "CUT_GUARD_S", 0.2)          # the guard, scaled to a 3-second recording
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "upload"
        recorder_folder(root, n=90, duplicate_stamps=False)
        rep = f.convert(root, "teleop_arms", Path(t) / "eps", "test", 900)
        src = Path(t) / "eps" / rep["episodes"][0]["episode_id"]
        parts = pieces.write_pieces(src, Path(t) / "pieces")
        assert len(parts) == 2
        pcs = [json.loads((p / "context.json").read_text()) for p in parts]
        cut = pcs[1]["piece"]["t0_s"]

        def result(i, end):
            return {"episode_dir": str(parts[i]), "parse_ok": True, "model": "m",
                    "config": {"timesteps_s": [0.0, 1.0], "resolution_route": {"cost_usd": 0.01 * (i + 1)}},
                    "usage": {"est_cost_usd": 0.5},
                    "labels": {"task_summary": f"part {i + 1}", "timeline": [{"start_s": 0.2, "end_s": 0.9,
                                                                              "action": "reach"}],
                               "key_events": [{"t_s": 0.5, "label": "grasp"}],
                               "completion": {"task_completed": "partial"},
                               "data_issues": [{"category": "truncated_episode", "severity": "medium", "t_s": end,
                                                "issue": "the episode ends mid-task"}]}}
        out = pieces.stitch(src, [(pcs[0], result(0, cut - 0.1)), (pcs[1], result(1, pcs[1]["piece"]["t1_s"] - cut))])
        lab = out["labels"]
        assert [s["start_s"] for s in lab["timeline"]] == [0.2, round(0.2 + cut, 3)]
        assert [k["t_s"] for k in lab["key_events"]] == [0.5, round(0.5 + cut, 3)]
        assert lab['tasks'] == []
        assert [x['completion']['task_completed'] for x in lab['part_assessments']] == ['partial', 'partial']
        assert [x["part"] for x in lab["data_issues"]] == [2]                  # the recording's own end stays
        assert [x["excluded_by"] for x in lab["_excluded"]] == ["piece_cut"]   # the cut after part 1 is ours
        assert out["stitched"]["parts"] == 2 and out["usage"]["est_cost_usd"] == 1.0
        # every part's routing call is billed, and every part was checked against the recording's instruction
        assert out["config"]["resolution_route"]["cost_usd"] == 0.03 and len(out["config"]["resolution_route"]["parts"]) == 2
        assert harness.episode_cost(out) == 1.03 and out["prompt_mode"] == "given"



def test_a_recording_with_other_signals_can_be_labelled_in_parts(tmp_path, monkeypatch):
    """A LeRobot upload with one column the reader has no slot for (a base velocity) is longer than a part: each part
    loads, carrying its own rows of the signals its context lists."""
    from test_prepare import _lerobot_v21
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 0.5)
    n = 45
    root = tmp_path / "mobile"
    _lerobot_v21(root, n=n, extra={"observation.velocity": list(np.linspace(0, 1, n)[:, None] * [1.0, 0.0])})
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    src = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    assert json.loads((src / "context.json").read_text())["signals"]
    parts = pieces.write_pieces(src, tmp_path / "pieces")
    assert len(parts) > 1
    for p in parts:
        e = me.load(p)
        assert len(next(iter(e["signals"].values()))) == int(e["context"]["n_state_frames"])


def _long_press(tmp_path, monkeypatch, measured: bool = True) -> tuple[Path, list[Path]]:
    """A 10 s head camera recording whose right glove is pressed flat by the palm from 3 s to 6 s, written in parts of
    about 1 s: the parts from 3.7 s to 5.8 s fall inside the press. Without measured, its context has no contacts (an
    upload prepared before contacts were measured)."""
    from test_touch_depth import _hdf5
    root = tmp_path / "up"
    root.mkdir()
    _hdf5(root / "kitchen_p1.hdf5", demos=1, n=300, press=(90, 180), palm=True)
    rep = f.convert(root, "ego_head", tmp_path / "eps", "touchset", 900)
    src = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    if not measured:
        ctx = json.loads((src / "context.json").read_text())
        del ctx["contacts"]
        (src / "context.json").write_text(json.dumps(ctx))
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    return src, pieces.write_pieces(src, tmp_path / "pieces")


def test_a_recording_prepared_without_contacts_gives_each_part_the_whole_recordings_contacts(tmp_path, monkeypatch):
    """A recording whose context has no contacts has them found once, on the whole recording, and clipped into each
    part, so a part inside a long press keeps the contact that finding it on the part's own slice would miss."""
    from label import contacts as lc
    src, parts = _long_press(tmp_path, monkeypatch, measured=False)
    assert "contacts" not in json.loads((src / "context.json").read_text())      # the recording's own is unchanged
    inside = [p for p in parts if 3.0 < me.load(p)["context"]["piece"]["t0_s"]
              and me.load(p)["context"]["piece"]["t1_s"] < 6.0]
    assert inside
    records = {}
    for p in inside:
        e = me.load(p)
        assert [c["id"] for c in e["context"]["contacts"]] == ["c1"]
        assert lc.find(e["signals"], e["signal_meta"], np.arange(len(e["state"])) / 30.0) == []   # its slice alone
        r = me.build_request(p)
        assert "contacts" in r["blocks"] and "CONTACTS:" in r["prompt"] and r["contact_views"]["shown"] == ["c1"]
        records[p] = {"contacts": r["contacts"], "contact_views": {"shown": ["c1"], "strips": {}}}
    # the stitched record lists the contact the parts were asked about, on the recording's own clock
    out = pieces.stitch(src, [(json.loads((p / "context.json").read_text()),
                               {"episode_dir": str(p), "parse_ok": True, "model": "m", "usage": {},
                                "config": {"timesteps_s": [0.0]}, "labels": {"task_summary": p.name},
                                **records.get(p, {})}) for p in parts])
    (c,) = out["contacts"]
    assert c["id"] == "c1" and abs(c["start_s"] - 3.0) < 0.05 and abs(c["end_s"] - 6.0) < 0.05


def test_a_part_inside_a_long_press_keeps_the_contact_the_whole_recording_shows(tmp_path, monkeypatch):
    """Touch is judged once, on the whole recording: a part inside a long press has no rest of its own, so its slice
    alone does not read as touch, and the part takes the recording's verdict from its context."""
    from label import signals as sg
    src, parts = _long_press(tmp_path, monkeypatch)
    assert len(parts) >= 3
    assert all("touch" not in s for s in json.loads((src / "context.json").read_text())["signals"])
    inside = []
    for p in parts:
        e = me.load(p)
        assert {s["name"]: s["touch"] for s in e["context"]["signals"]} == {"right_pressure": True,
                                                                            "right_hand_landmarks": False}
        m = e["signal_meta"]["right_pressure"]
        a = np.asarray(e["signals"]["right_pressure"], dtype=np.float64)
        if e["context"]["contacts"] and not sg.is_touch("right_pressure", a, m.get("rest"), m.get("swing")):
            inside.append(p)
    assert len(inside) >= 2                          # parts the contact covers whose own numbers do not read as touch
    for p in inside:
        r = me.build_request(p)
        assert "contacts" in r["blocks"] and "CONTACTS:" in r["prompt"] and r["contact_views"]["shown"] == ["c1"]


def test_a_stitched_record_keeps_the_contacts_its_parts_showed_and_every_parts_blocks(tmp_path, monkeypatch):
    """The stitched record lists the recording's contacts some part showed, not the ones no part was asked about, and
    its config names every block, output field and check any part had, in the order they first appear."""
    src, parts = _long_press(tmp_path, monkeypatch)
    ctx = json.loads((src / "context.json").read_text())
    ctx["contacts"].append({**ctx["contacts"][0], "id": "c9", "signals": ["right_hand_landmarks"]})
    (src / "context.json").write_text(json.dumps(ctx))
    pcs = [json.loads((p / "context.json").read_text()) for p in parts]

    def result(i):
        shown = i == 1
        blocks = ["collection_note", "no_state", "signals"] + (["contacts"] if shown else [])
        return {"episode_dir": str(parts[i]), "parse_ok": True, "model": "m", "usage": {},
                "config": {"timesteps_s": [0.0], "prompt_blocks": blocks,
                           "schema_fields": ["contacts", "contacts_missing"] if shown else [],
                           "checks_implied": ["sensor_checks"] + (["contact_checks"] if shown else [])},
                "labels": {"task_summary": f"part {i + 1}"},
                **({"contacts": [ctx["contacts"][0]], "contact_views": {"shown": ["c1"], "strips": {}}}
                   if shown else {})}
    out = pieces.stitch(src, [(pc, result(i)) for i, pc in enumerate(pcs)])
    assert [c["id"] for c in out["contacts"]] == ["c1"]
    assert out["config"]["prompt_blocks"] == ["collection_note", "no_state", "signals", "contacts"]
    assert out["config"]["schema_fields"] == ["contacts", "contacts_missing"]
    assert out["config"]["checks_implied"] == ["sensor_checks", "contact_checks"]
    quiet = []
    for i, pc in enumerate(pcs):
        r = result(i)
        r.pop('contacts', None)
        r.pop('contact_views', None)
        r['labels']['contacts_missing'] = [{'t_s': .5, 'hand': 'right', 'object': 'cup'}]
        quiet.append((pc, r))
    stitched = pieces.stitch(src, quiet)
    assert not stitched.get('contacts')
    assert [m['t_s'] for m in stitched['labels']['contacts_missing']] == pytest.approx(
        [pc['piece']['t0_s'] + .5 for pc in pcs])


def test_a_contact_cut_by_our_cuts_asks_each_part_only_for_the_strips_it_shows_and_stitches_them_back(tmp_path,
                                                                                                    monkeypatch):
    """The press from 3 s to 6 s crosses several cuts: the part where it begins shows only its begin strip, the part
    where it ends only its end strip, and the parts inside it neither, so none of them is asked for a frame of a strip
    it does not show. The stitched answer takes first_touch_frame from the part that showed the begin strip and
    last_touch_frame from the part that showed the end strip, each with that strip's times on the recording's clock."""
    src, parts = _long_press(tmp_path, monkeypatch)
    results, kinds = [], []
    for i, p in enumerate(parts):
        r = me.build_request(p)
        strips = (r.get("contact_views") or {}).get("strips", {}).get("c1")
        kinds.append(sorted(strips) if strips is not None else None)
        if strips is not None:
            assert ("first_touch_frame" in r["prompt"]) == ("begin" in strips)
            assert ("last_touch_frame" in r["prompt"]) == ("end" in strips)
        answer = {"id": "c1", "touch_seen": "yes", "first_touch_frame": 10 + i, "last_touch_frame": 20 + i}
        results.append({"episode_dir": str(p), "parse_ok": True, "model": "m", "usage": {},
                        "config": {"timesteps_s": [0.0]}, "labels": {"task_summary": p.name, "contacts": [answer]},
                        **({"contacts": r["contacts"], "contact_views": r["contact_views"]} if strips is not None
                           else {})})
    shown = [k for k in kinds if k is not None]
    assert shown[0] == ["begin"] and shown[-1] == ["end"] and [] in shown[1:-1]
    first, last = kinds.index(["begin"]), kinds.index(["end"])
    pcs = [json.loads((p / "context.json").read_text()) for p in parts]
    out = pieces.stitch(src, list(zip(pcs, results)))
    (c,) = out["labels"]["contacts"]
    assert c["first_touch_frame"] == 10 + first and c["last_touch_frame"] == 20 + last
    strips = out["contact_views"]["strips"]["c1"]
    shifted = lambda i, kind: [round(x + pcs[i]["piece"]["t0_s"], 3)
                               for x in results[i]["contact_views"]["strips"]["c1"][kind]]
    assert strips["begin"] == shifted(first, "begin") and strips["end"] == shifted(last, "end")
    assert abs(strips["begin"][2] - 3.0) < 0.05 and abs(strips["end"][1] - 6.0) < 0.05


def test_a_part_of_a_long_head_camera_recording_gets_its_subtasks_on_its_own_clock(tmp_path, monkeypatch):
    """The dataset's timed subtasks are on the recording's clock; a part's frames start at 0 at its cut."""
    from prepare import formats
    from test_formats import _clip
    (tmp_path / "v").mkdir()
    _clip(tmp_path / "v" / "a.mp4", 90)                      # 3 s at 30 fps
    ep = tmp_path / "eps" / "episode_a"
    subs = [{"t0": 0.0, "t1": 1.4, "label": "pick up the cup"},
            {"t0": 1.2, "t1": 2.4, "label": "transfer the cup", "ok": False, "publisher_note": "handoff attempt"},
            {"t0": 1.6, "t1": 3.0, "label": "wipe the table"}]
    formats.video_views_episode(ep, {"exo": ("a", tmp_path / "v" / "a.mp4")}, "ego_head", "mine",
                                {"instruction": "clean up", "annotation_subtasks": subs})
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.6)
    parts = pieces.write_pieces(ep, tmp_path / "pieces")
    assert len(parts) == 2
    e = me.load(parts[1])
    t0 = e["context"]["piece"]["t0_s"]
    req = me.build_request(parts[1])
    block = req["prompt"].split("THE DATASET'S ANNOTATION FOR THIS EPISODE")[1]
    # the part's own clock runs 0..(t1 - t0); the second subtask is at 1.6-3.0 s of the recording
    assert f"{1.6 - t0:.1f}-{3.0 - t0:.1f}s  wipe the table" in block, block
    first = me.load(parts[0])["context"]["annotation_subtasks"]
    assert [x["label"] for x in first] == ['pick up the cup', 'transfer the cup'] + (["wipe the table"] if t0 > 1.6 else [])
    earlier = next(x for x in first if x['label'] == 'transfer the cup')
    later = next(x for x in e['context']['annotation_subtasks'] if x['label'] == 'transfer the cup')
    assert earlier['t1'] == 2.4 and later['t0'] == round(1.2 - t0, 3)
    assert earlier['source_annotation'] == later['source_annotation'] == subs[1]
    assert earlier['request_window']['continues_after'] is True
    assert later['request_window']['continues_before'] is True
    assert earlier['request_window']['visible_end_s'] == t0
    assert later['request_window']['visible_start_s'] == 0
    assert later['ok'] is False and '(marked unsuccessful)' in block
    assert 'Original recording span: 1.2-2.4s' in block
    assert 'Only the visible portion can be checked' in block
    assert 'continues before this request' in block
    assert json.loads((ep / 'context.json').read_text())['annotation_subtasks'] == subs


def test_stitch_run_carries_every_short_episodes_reply_whatever_came_back(tmp_path):
    """A short episode's reply is carried to the board as it came: parsed, not parsed, or cut off at the output
    limit (failed_<episode>.json), so the board shows the episode either way."""
    src = tmp_path / "job" / "run" / "out"
    src.mkdir(parents=True)
    (src / "episode_000000.json").write_text(json.dumps({"parse_ok": True}))
    (src / "episode_000001.json").write_text(json.dumps({"parse_ok": False}))
    (src / "failed_episode_000002.json").write_text(json.dumps({"finish_reason": "length"}))
    out = tmp_path / "final" / "out"
    pieces.stitch_run(tmp_path / "job", tmp_path / "eps", {}, out)
    assert sorted(p.name for p in out.iterdir()) == ["episode_000000.json", "episode_000001.json",
                                                     "failed_episode_000002.json"]


def _three_parts(t: Path, monkeypatch) -> tuple[Path, Path, list[Path]]:
    """A recording cut into three parts, written where a job keeps them: job/pieces."""
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    root = t / "upload"
    recorder_folder(root, n=90, duplicate_stamps=False)
    rep = f.convert(root, "teleop_arms", t / "eps", "test", 900)
    src = t / "eps" / rep["episodes"][0]["episode_id"]
    job = t / "job"
    parts = pieces.write_pieces(src, job / "pieces")
    assert len(parts) == 3
    (job / "run" / "out").mkdir(parents=True)
    return job, src, parts


def _part_result(p: Path, action: str) -> dict:
    return {"episode_dir": str(p), "parse_ok": True, "model": "m", "config": {"timesteps_s": [0.0, 0.5]},
            "usage": {"est_cost_usd": 0.5}, "decode_failed": [],
            "labels": {"task_summary": action, "timeline": [{"start_s": 0.1, "end_s": 0.6, "action": action}],
                       "completion": {"task_completed": "success"}}}


def test_camera_content_consensus_keeps_part_source_proof_and_abstains_at_gaps(tmp_path, monkeypatch):
    import copy
    from label.evidence_access import source_proof
    from board import dictionary_projection, public_export, static, to_board
    _job, source, folders = _three_parts(tmp_path, monkeypatch)
    contexts = [json.loads((p / 'context.json').read_text()) for p in folders]
    records = []
    for folder, ctx in zip(folders, contexts):
        record = _part_result(folder, 'observe the supplied views')
        record['config'].update(views=['exo', 'left', 'right'], camera_observation_views=['exo', 'left', 'right'],
                                schema_fields=['camera_observations'])
        record['input_identity'] = {'source_proof': source_proof(ctx, folder)}
        record['camera_observations'] = harness.bind_camera_observations([
            {'view': v, 'content': 'scene' if v == 'right' else 'sensor',
             'evidence': 'Visible workspace.' if v == 'right' else 'Visible sensor surface.'}
            for v in ('exo', 'left', 'right')], record)
        records.append(record)
    joined = pieces.stitch(source, list(zip(contexts, records)))
    assert len(joined['camera_observations']['parts']) == 3
    assert joined['camera_observations']['observations'][-1]['view'] == 'right'
    ctx = json.loads((source / 'context.json').read_text())
    saved = to_board.convert(joined, 'native parts')
    projected = dictionary_projection.apply(copy.deepcopy(saved), ctx, source)
    assert static.shown_cams(projected)[0] == 'right'
    assert public_export.project(projected)['camera_observations']['source_current'] is True

    conflicting = copy.deepcopy(records)
    conflicting[1]['camera_observations']['observations'][-1]['content'] = 'sensor'
    mixed = pieces.stitch(source, list(zip(contexts, conflicting)))
    assert not mixed['camera_observations']['observations']
    assert mixed['camera_observations']['withheld_observations']
    projected = dictionary_projection.apply(to_board.convert(mixed, 'conflicting parts'), ctx, source)
    assert not projected['camera_observations']['observations']
    assert static.shown_cams(projected) == ['exo', 'left', 'right']
    incomplete = copy.deepcopy(records)
    incomplete[1]['camera_observations']['observations'].pop(0)
    joined_partial = pieces.stitch(source, list(zip(contexts, incomplete)))
    assert not joined_partial['camera_observations']['observations']
    assert joined_partial['labels']['timeline'] and joined_partial['camera_observations']['parts']
    # A retained camera with no supplied images is not added to the observation contract.
    subset = copy.deepcopy(records)
    for record in subset:
        record['config']['camera_observation_views'] = ['left', 'right']
        record['camera_observations'] = harness.bind_camera_observations(
            record['camera_observations']['observations'][1:], record)
    supplied_only = pieces.stitch(source, list(zip(contexts, subset)))
    assert supplied_only['camera_observations']['views'] == ['left', 'right']
    assert len(supplied_only['camera_observations']['observations']) == 2
    missing = [{'part': 2, 't0_s': contexts[1]['piece']['t0_s'],
                't1_s': contexts[1]['piece']['t1_s'], 'why': 'no parsed labels'}]
    gapped = pieces.stitch(source, [(contexts[i], records[i]) for i in (0, 2)], missing)
    assert not gapped['camera_observations']['observations']
    # A later change to a child's ownership context cannot promote its old consensus.
    child_path = folders[1] / 'context.json'
    changed = json.loads(child_path.read_text())
    changed['source']['view_note'] = 'Changed native ownership claim'
    child_path.write_text(json.dumps(changed))
    stale = dictionary_projection.apply(copy.deepcopy(saved), ctx, source)
    assert stale['camera_observations']['source_current'] is False
    assert not stale['camera_observations']['observations']
    # Even a fresh child proof cannot assign a parent's view ID to a different native camera.
    changed = copy.deepcopy(contexts[1])
    child_path.write_text(json.dumps(changed))
    source_path = folders[1] / 'sources.json'
    sources = json.loads(source_path.read_text())
    sources['right']['packed'], sources['left']['packed'] = sources['left']['packed'], sources['right']['packed']
    source_path.write_text(json.dumps(sources))
    reassigned = copy.deepcopy(records)
    reassigned[1]['input_identity'] = {'source_proof': source_proof(changed, folders[1])}
    reassigned[1]['camera_observations'] = harness.bind_camera_observations(
        records[1]['camera_observations']['observations'], reassigned[1])
    assert not pieces.stitch(source, list(zip(contexts, reassigned)))['camera_observations']['observations']


@pytest.mark.parametrize('first', ['refused', 'claimed', 'unknown', 'newer_claimed', 'newer_unknown',
                                  'contradicted_refusal', 'paid_failed', 'paid_unknown_cost', 'paid_reply',
                                  'ambiguous_corrupt', 'ambiguous_index'])
def test_split_refusals_keep_each_dispatch_receipt_in_native_board_download(tmp_path, monkeypatch, first):
    import shutil
    import subprocess
    from board import build as board_build, rules, serve
    from label.dictionary_stage import label_spend, label_spend_complete
    from label.evidence_access import source_proof

    job, source, parts = _three_parts(tmp_path, monkeypatch)
    context = json.loads((source / 'context.json').read_text())
    (job / 'pieces' / ('.' + source.name + '.source_proof.json')).write_text(
        json.dumps(source_proof(context, source)))
    monkeypatch.setattr(harness, 'call_model_once', lambda *a, **k: pytest.fail('unexpected dispatch'))
    requests = []
    out = job / 'run' / 'out'
    for part in parts:
        request = harness.label_episode(part, job / 'dry' / (part.name + '.json'),
            model=None, reasoning=None, max_tokens=100, timeout=10, cell_w=128,
            api_key='dry-run', dry_run=True)
        requests.append(request)
        assert harness.record_budget_refusal(out / (part.name + '.json'), request)
    refused = out / ('noreply_' + parts[0].name + '.json')
    ambiguous = first.startswith('ambiguous_')
    unknown = ambiguous or first in ('claimed', 'unknown', 'newer_claimed', 'newer_unknown', 'paid_unknown_cost')
    if ambiguous or first in ('claimed', 'unknown', 'newer_claimed', 'newer_unknown', 'contradicted_refusal'):
        record = json.loads(refused.read_text())
        record.update(final_dispatch_outcome='not dispatched' if first == 'contradicted_refusal' else
                      'claimed' if first.endswith('claimed') else 'unverified', final_reserved_usd=6.4,
                      no_reply='TimeoutError /srv/private?token=secret')
        if first != 'contradicted_refusal':
            record.pop('no_reply_kind')
        if first.startswith('newer_') or ambiguous:
            import copy
            old = _part_result(parts[0], 'An older paid label must not leak.')
            old['usage'] = {'est_cost_usd': .02}
            old['input_identity'] = copy.deepcopy(requests[0]['input_identity'])
            old['input_identity']['source_proof'][0]['sha256'] = '0' * 64
            (out / (parts[0].name + '.json')).write_text(json.dumps(old))
            record.update(final_attempt_index=2, final_cost_history=[{'usage': {'est_cost_usd': .02}}])
        if first == 'ambiguous_index':
            record['final_attempt_index'] = True
        refused.write_text('{broken private receipt' if first == 'ambiguous_corrupt' else json.dumps(record))
    elif first in ('paid_failed', 'paid_unknown_cost', 'paid_reply'):
        record = {key: requests[0][key] for key in ('episode_dir', 'model', 'input_identity', 'annotation_policy')}
        record['usage'] = {'est_cost_usd': .02}
        if first in ('paid_failed', 'paid_unknown_cost'):
            record.update(finish_reason='length', content_tail='An incomplete recorded response')
            if first == 'paid_unknown_cost':
                record.update(usage={'est_cost_usd': None}, final_dispatch_outcome='unverified', final_reserved_usd=6.4)
            target = out / ('failed_' + parts[0].name + '.json')
        else:
            record.update(parse_ok=True, labels=complete_reply.labels('Reach for the object.'))
            target = out / (parts[0].name + '.json')
        refused.unlink()
        target.write_text(json.dumps(record))
    before = {path: path.read_bytes() for path in out.glob('*.json')}
    expected_cost = 1e300 if ambiguous else 6.42 if first.startswith('newer_') else 6.4 if unknown or first == 'contradicted_refusal' else .02 if first.startswith('paid_') else 0
    assert label_spend(job) == expected_cost
    assert label_spend_complete(job) is (not unknown)
    final = job / 'run_final'
    pieces.stitch_run(job, source.parent, {source.name: [part.name for part in parts]}, final / 'out')
    assert before == {path: path.read_bytes() for path in out.glob('*.json')}
    joined = json.loads((final / 'out' / (source.name + '.json')).read_text())
    assert joined['parse_ok'] is (first == 'paid_reply')
    assert 'An older paid label must not leak.' not in json.dumps(joined)
    assert 'final_dispatch_outcome' not in joined and 'no_reply_kind' not in joined
    missing = joined['stitched']['missing']
    skipped = [row for row in missing if row.get('no_reply_kind') == 'spend_cap']
    assert len(skipped) == (3 if first == 'refused' else 2)
    assert all(row['final_dispatch_outcome'] == 'not dispatched' for row in skipped)
    assert all(row['no_reply_kind'] == 'spend_cap' for row in joined['config']['pieces'][1:])
    if first in ('claimed', 'unknown', 'newer_claimed', 'newer_unknown'):
        assert missing[0]['final_dispatch_outcome'] == ('claimed' if first.endswith('claimed') else 'unverified')
        # the part's TimeoutError is our own client giving up, named apart from a provider failure on the board
        assert missing[0]['no_reply_kind'] == 'client_timeout'
    elif ambiguous or first in ('paid_failed', 'contradicted_refusal'):
        assert 'final_dispatch_outcome' not in missing[0] and 'no_reply_kind' not in missing[0]
    if first in ('paid_failed', 'paid_unknown_cost'):
        assert missing[0]['raw_head'] == 'An incomplete recorded response'
        assert missing[0]['finish_reason'] == 'length'
    info = {'run_id': 'fixture', 'code': 'fixture', 'kind': 'review'}
    entry = rules.own_data_entry('fixture', str(final), str(source.parent), 'teleop_arms')
    native = board_build.episode_file(entry, {}, source.name + '.json', source.name, joined,
                                    info, source.parent, context)
    public = serve.public_label(native)
    public_skipped = [row for row in public['_stitched']['missing'] if row.get('no_reply_kind') == 'spend_cap']
    assert public_skipped == [row for row in public['_stitched']['missing'] if row.get('final_dispatch_outcome') == 'not dispatched']
    assert len(public_skipped) == len(skipped)
    if first != 'paid_reply':
        assert public['_label_failed']['status'] == 'no_part'
        assert len([row for row in public['_label_failed']['parts'] if row.get('no_reply_kind') == 'spend_cap']) == len(skipped)
    assert 'token=secret' not in json.dumps(public) and 'client_timeout' not in json.dumps(public)
    assert label_spend(job) == expected_cost
    node = shutil.which('node')
    if not node:
        pytest.skip('no node')
    program = """
const {piece} = require('./tests/page_functions')('./board/serve.py');
const render = new Function('famName','famList',piece('esc')+piece('fmtDur')+piece('fmtT')
  +piece('partFailureReason')+piece('cmpFailHtml')+piece('readerIssueRows')+piece('readerIssueEntries')
  +'return {cmpFailHtml,readerIssueRows};')(x=>x,()=> 'labelling');
const d = JSON.parse(require('fs').readFileSync(0,'utf8'));
console.log(JSON.stringify({card:d._label_failed ? render.cmpFailHtml(d._label_failed,'Astra',d._usage,true) : '',
                           gaps:render.readerIssueRows(d)}));
"""
    completed = subprocess.run([node, '-e', program], cwd=Path(__file__).resolve().parent.parent,
        input=json.dumps(public), capture_output=True, text=True, check=True)
    rendered = json.loads(completed.stdout)
    block = rendered['card'] if first != 'paid_reply' else ''.join(rendered['gaps'])
    assert block.count('Annotation not sent') == len(skipped)
    assert 'remaining spend cap' in block
    if first in ('claimed', 'unknown', 'newer_claimed', 'newer_unknown'):
        assert 'Response outcome unknown' in block
    if first in ('paid_failed', 'paid_unknown_cost'):
        assert 'Response cut off' in block and 'Response outcome unknown' not in block
    if first != 'paid_reply':
        assert 'The call cost' not in block
    assert 'token=secret' not in completed.stdout and 'not called' not in completed.stdout


def test_stitch_keeps_distinct_untimed_issues_and_failed_rgb_receipts(tmp_path, monkeypatch):
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    out = job / 'run' / 'out'
    left = _part_result(parts[0], 'reach')
    left['labels']['data_issues'] = [
        {'category': 'sensor_problem', 'severity': 'medium', 'issue': 'left wrist is noisy'},
        {'category': 'sensor_problem', 'severity': 'medium', 'issue': 'right wrist is noisy'}]
    last = _part_result(parts[2], 'place')
    last['labels']['data_issues'] = [{'category': 'sensor_problem', 'severity': 'medium',
                                      'issue': 'left wrist is noisy'}]
    (out / f'{parts[0].name}.json').write_text(json.dumps(left))
    (out / f'{parts[2].name}.json').write_text(json.dumps(last))
    failed = {'episode_dir': str(parts[1]), 'finish_reason': 'length', 'generation_id': 'paid-2',
              'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'est_cost_usd': .3},
              'decode_failed': [{'view': 'exo', 't0_s': 1.1, 't1_s': 1.2}]}
    (out / f'failed_{parts[1].name}.json').write_text(json.dumps(failed))
    pieces.stitch_run(job, src.parent, {src.name: [p.name for p in parts]}, tmp_path / 'final')
    result = json.loads((tmp_path / 'final' / f'{src.name}.json').read_text())
    assert [(x['issue'], x.get('parts')) for x in result['labels']['data_issues']] == [
        ('left wrist is noisy', [1, 3]), ('right wrist is noisy', None)]
    assert result['usage']['est_cost_usd'] == 1.3
    assert result['usage']['prompt_tokens'] == 100
    assert result['decode_failed'] == failed['decode_failed']
    assert result['config']['pieces'][1]['generation_id'] == 'paid-2'


def test_a_long_recording_with_a_part_that_failed_is_stitched_from_the_rest_with_the_gap_marked(tmp_path, monkeypatch):
    """Part 2's reply did not parse: the recording is stitched from parts 1 and 3, on its own clock, and the record
    names part 2 with its span and why, which the board flags as a data issue at that span."""
    from board import build as board_build
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    out = job / "run" / "out"
    (out / f"{parts[0].name}.json").write_text(json.dumps(_part_result(parts[0], "reach")))
    (out / f"{parts[1].name}.json").write_text(json.dumps({"episode_dir": str(parts[1]), "parse_ok": False,
                                                           "labels": {"_raw": "{oops", "_parse_error": "x"},
                                                           "evidence_inspection": {"version": 1, "cost_usd": .12, "inspections": []}}))
    (out / f"{parts[2].name}.json").write_text(json.dumps(_part_result(parts[2], "place")))
    res = pieces.stitch_run(job, src.parent, {src.name: [p.name for p in parts]}, tmp_path / "final")
    assert res["stitched"] == 1 and res["incomplete"] == [src.name]
    r = json.loads((tmp_path / "final" / f"{src.name}.json").read_text())
    assert r["evidence_inspection"]["cost_usd"] == .12
    assert r["evidence_inspection"]["parts"][0]["part"] == 2
    pcs = [json.loads((p / "context.json").read_text())["piece"] for p in parts]
    assert r["parse_ok"] is True and [s["action"] for s in r["labels"]["timeline"]] == ["reach", "place"]
    assert r["labels"]["timeline"][1]["start_s"] == round(0.1 + pcs[2]["t0_s"], 3)
    assert r["stitched"]["parts"] == 3 and r["stitched"]["cuts_s"] == [pcs[1]["t0_s"], pcs[2]["t0_s"]]
    (gap,) = r["stitched"]["missing"]
    # with the start of the part's own reply, which the board shows (it never reached the board before)
    assert gap == {"part": 2, "t0_s": pcs[1]["t0_s"], "t1_s": pcs[1]["t1_s"], "why": "the model's reply did not parse (x)",
                   "raw_head": "{oops"}
    assert r['labels']['tasks'] == []
    assert [x['task_summary'] for x in r['labels']['part_assessments']] == ['reach', 'place']
    assert r['labels']['completion']['task_completed'] == 'unclear'
    assert 'part 2 has no labels' in r['labels']['completion']['reason']
    assert r['config']['n_frames_sent'] is None and r['config']['n_timesteps'] is None
    d = {}
    board_build.add_reader_issues(d, {}, r)
    (iss,) = d["dataset_checks"]["reader_issues"]
    assert iss["kind"] == "part_not_labelled" and iss["t0_s"] == gap["t0_s"] and iss["t1_s"] == gap["t1_s"]
    assert iss["family"] == "label-failed" and "Part 2 of 3" in iss["what"] and "did not parse" in iss["what"]
    # the board's issue carries the start of the part's reply, which the page shows under it (tests/reader_issues.js)
    assert iss["reply_head"] == "{oops"


def test_a_long_recording_none_of_whose_parts_parsed_is_still_on_the_board(tmp_path, monkeypatch):
    """Every part failed (one cut off, one not parsing, one never answered): the recording is still written, as a reply
    that gave no labels, so the board shows its footage, checks and sensors and says why."""
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    out = job / "run" / "out"
    (out / f"failed_{parts[0].name}.json").write_text(json.dumps({"episode_dir": str(parts[0]),
                                                                  "finish_reason": "length", "content_tail": "...",
                                                                  "usage": {"completion_tokens": 64000}}))
    (out / f"{parts[1].name}.json").write_text(json.dumps({"episode_dir": str(parts[1]), "parse_ok": False,
                                                           "labels": {"_raw": "{", "_parse_error": "x"}}))
    (out / f"noreply_{parts[2].name}.json").write_text(json.dumps({"no_reply": "decoder failed after inspection",
        "evidence_inspection": {"version": 1, "cost_usd": .07, "status": "incomplete", "inspections": []}}))
    res = pieces.stitch_run(job, src.parent, {src.name: [p.name for p in parts]}, tmp_path / "final")
    assert res["stitched"] == 0 and res["unlabelled"] == [src.name]
    r = json.loads((tmp_path / "final" / f"{src.name}.json").read_text())
    assert r["evidence_inspection"]["cost_usd"] == .07
    assert r["evidence_inspection"]["parts"][0]["part"] == 3
    assert r["parse_ok"] is False and r["episode_dir"] == str(src)
    assert [g["why"] for g in r["stitched"]["missing"]] == [
        "the model's reply was cut off at the output limit", "the model's reply did not parse (x)",
        "the model gave no reply (decoder failed after inspection)"]
    # the board says why for each part, never that the model's reply did not parse
    from board import build as board_build
    from board import to_board
    d = to_board.convert(r, "demo")
    assert d["_label_failed"]["status"] == "no_part"
    assert [g["why"] for g in d["_label_failed"]["parts"]] == [g["why"] for g in r["stitched"]["missing"]]
    board_build.add_reader_issues(d, {}, r)
    (iss,) = d["dataset_checks"]["reader_issues"]
    assert iss["kind"] == "no_part_labelled" and iss["family"] == "label-failed"
    assert ("part 1 was cut off at the output limit" in iss["what"] and "part 2 did not parse" in iss["what"]
            and "part 3 never answered" in iss["what"]), iss["what"]


def test_a_part_time_written_as_text_is_moved_onto_the_recordings_clock():
    """_shift moved only times written as numbers, so a part's step at "0.2" stayed at 0.2 s of the recording though it
    lies in a later part. A time written as text that reads as a number (number()) is shifted like any other; one that
    reads as none is left as it is."""
    lab = {"timeline": [{"start_s": "0.2", "end_s": "0.5s", "action": "a"}, {"start_s": "late", "end_s": 1}]}
    out = pieces._shift(lab, 10.0)
    assert out["timeline"][0]["start_s"] == 10.2 and out["timeline"][0]["end_s"] == 10.5
    assert out["timeline"][1] == {"start_s": "late", "end_s": 11.0}


def test_a_part_time_outside_its_own_part_is_kept_and_flagged(tmp_path, monkeypatch):
    """A step the model timed before its own part began (part 2 says -0.8 s) was moved onto the recording's clock into
    the previous part's footage with no flag, and so was a key event past its part's end. They are kept as given, the
    stitched record counts them per part, and the board flags them with the part and its span. A time within half a
    second of its part's edges is a rounding."""
    from board import build as board_build
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    out = job / "run" / "out"
    for i, p in enumerate(parts):
        r = _part_result(p, f"step {i + 1}")
        if i == 1:
            r["labels"]["timeline"].append({"start_s": -0.8, "end_s": 0.2, "action": "early"})
            r["labels"]["key_events"] = [{"t_s": 0.5, "label": "inside"}, {"t_s": 9.0, "label": "late"}]
        if i == 2:
            r["labels"]["timeline"].append({"start_s": -0.3, "end_s": 0.2, "action": "rounding"})
        (out / f"{p.name}.json").write_text(json.dumps(r))
    pieces.stitch_run(job, src.parent, {src.name: [p.name for p in parts]}, tmp_path / "final")
    r = json.loads((tmp_path / "final" / f"{src.name}.json").read_text())
    pcs = [json.loads((p / "context.json").read_text())["piece"] for p in parts]
    early = next(s for s in r["labels"]["timeline"] if s["action"] == "early")
    assert early["start_s"] == round(pcs[1]["t0_s"] - 0.8, 3) < pcs[1]["t0_s"]
    assert r["stitched"]["outside_part"] == [{"part": 2, "t0_s": pcs[1]["t0_s"], "t1_s": pcs[1]["t1_s"], "steps": 1,
                                              "key_events": 1}]
    d = {"event_labels": [], "key_events": []}
    (iss,) = board_build.steps_outside(d, r)
    assert iss["kind"] == "model_steps_outside_episode"
    assert (f"1 step and 1 key event of part 2 lie outside that part's footage, from {pcs[1]['t0_s']:.1f} s to "
            f"{pcs[1]['t1_s']:.1f} s") in iss["what"], iss["what"]


def test_a_part_whose_output_does_not_read_is_a_gap_and_one_that_breaks_the_format_is_stitched(tmp_path, monkeypatch):
    """One part output that was not valid JSON crashed stitch_run, and one parsed part with a field of the wrong type
    (a string in scene.objects, a key event as text, a time as text) crashed the stitch, losing the whole recording.
    The unreadable part is a gap, named with why, as is a part the spend cap stopped; the part that breaks the format is
    stitched with what it gives, and what it broke is carried to the board."""
    from board import build as board_build
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    out = job / "run" / "out"
    bad = _part_result(parts[0], "reach")
    bad["labels"]["scene"] = {"objects": ["cup", {"name": "plate"}]}
    bad["labels"]["key_events"] = ["grasped", {"t_s": "0.2", "label": "lifted"}]
    bad["labels"]["timeline"].append({"start_s": "late", "end_s": 0.9, "action": "untimed"})
    (out / f"{parts[0].name}.json").write_text(json.dumps(bad))
    (out / f"{parts[1].name}.json").write_text('{"episode_dir": "x", "parse_ok": tr')
    (out / f"noreply_{parts[2].name}.json").write_text(json.dumps({"episode_dir": str(parts[2]), "parse_ok": False,
                                                                   "no_reply": "spend cap $1.00 reached"}))
    res = pieces.stitch_run(job, src.parent, {src.name: [p.name for p in parts]}, tmp_path / "final")
    assert res["stitched"] == 1 and res["incomplete"] == [src.name]
    r = json.loads((tmp_path / "final" / f"{src.name}.json").read_text())
    assert [s["action"] for s in r["labels"]["timeline"]] == ["reach", "untimed"]
    assert [k["label"] for k in r["labels"]["key_events"]] == ["lifted"]
    assert [(o["name"], o["source_piece"]["index"]) for o in r["labels"]["scene"]["objects"]] == [("plate", 1)]
    assert sorted((x["field"], x["part"]) for x in r["labels"]["_dropped"]) == [("key_events", 1), ("scene.objects", 1)]
    whys = {g["part"]: g["why"] for g in r["stitched"]["missing"]}
    assert whys[2].startswith("its output file does not read (JSONDecodeError")
    assert whys[3] == "the model gave no reply (spend cap $1.00 reached)"
    d = {}
    board_build.add_reader_issues(d, {}, r)
    kinds = [x["kind"] for x in d["dataset_checks"]["reader_issues"]]
    assert kinds == ["part_not_labelled", "part_not_labelled", "model_reply_fields_dropped"]
    assert "spend cap $1.00 reached" in d["dataset_checks"]["reader_issues"][1]["what"]
    # the short episodes' records of no reply reach the board as well
    (out / "noreply_episode_000009.json").write_text(json.dumps({"no_reply": "spend cap $1.00 reached"}))
    pieces.stitch_run(job, src.parent, {src.name: [p.name for p in parts]}, tmp_path / "final2")
    assert (tmp_path / "final2" / "noreply_episode_000009.json").exists()


def test_a_dataset_label_with_a_time_that_is_no_number_never_stops_a_long_recording_being_cut(tmp_path, monkeypatch):
    """A dataset label with a string start crashed the cutting of a long recording, one with no start became a label
    at 0 s of the first part, and one whose time was not a number silently left every part's prompt. A time written as
    text that reads as a number is placed as that number; a label with no time the board can read is kept, untimed and
    named, in every part, as a short episode's prompt names it."""
    from prepare import formats
    from test_formats import _clip
    (tmp_path / "v").mkdir()
    _clip(tmp_path / "v" / "a.mp4", 90)                      # 3 s at 30 fps
    ep = tmp_path / "eps" / "episode_a"
    subs = [{"t0": "abc", "t1": 1.0, "label": "string start"}, {"t0": None, "label": "no start"},
            {"t0": "2.0", "t1": "2.5", "label": "times as text"}, {"t0": float("nan"), "t1": 2.0, "label": "nan start"},
            {"t0": 0.2, "t1": 1.0, "label": "timed"}]
    formats.video_views_episode(ep, {"exo": ("a", tmp_path / "v" / "a.mp4")}, "ego_head", "mine",
                                {"instruction": "clean up", "annotation_subtasks": subs})
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.6)
    parts = pieces.write_pieces(ep, tmp_path / "pieces")
    assert len(parts) == 2
    t0 = me.load(parts[1])["context"]["piece"]["t0_s"]
    for i, part in enumerate(parts):
        got = me.load(part)["context"]["annotation_subtasks"]
        untimed = [x["label"] for x in got if x["t0"] is None]
        assert untimed == ["string start", "no start", "nan start"], (i, got)
        block = me.build_request(part)["prompt"].split("THE DATASET'S ANNOTATION FOR THIS EPISODE")[1]
        assert "no time  string start" in block and "no time  no start" in block and "no time  nan start" in block
    second = {x["label"]: x for x in me.load(parts[1])["context"]["annotation_subtasks"]}
    assert second["times as text"]["t0"] == round(2.0 - t0, 3) and second["times as text"]["t1"] == round(2.5 - t0, 3)
    first = [x["label"] for x in me.load(parts[0])["context"]["annotation_subtasks"] if x["t0"] is not None]
    assert first == ["timed"] + (["times as text"] if t0 > 2.0 else [])


def test_a_dataset_label_time_written_as_text_reads_the_same_in_the_prompt_the_parts_and_the_board(tmp_path):
    """The parts of a long recording read a dataset label's time written as text ("2.0") as that number and the board
    did too, but a short episode's prompt listed the same label with no time, and a time written "1.5s" read on no
    path. One rule (label/episode.py number) now reads them alike: text that reads as a number is that time, and
    anything else (a word, true or false) stays untimed everywhere."""
    from board import build as board_build
    from test_formats import _clip
    (tmp_path / "v").mkdir()
    _clip(tmp_path / "v" / "a.mp4", 90)
    ep = tmp_path / "eps" / "episode_a"
    subs = [{"t0": "2.0", "t1": "2.5", "label": "times as text",
             'request_window': {'source_start_s': 80, 'source_end_s': 90},
             'source_annotation': {'publisher_frame_note': 'native metadata, not an API piece'}},
            {"t0": "1.5s", "label": "time with its unit"},
            {"t0": True, "t1": 2.0, "label": "a flag, no time"}, {"t0": 0.2, "t1": 1.0, "label": "timed"}]
    f.video_views_episode(ep, {"exo": ("a", tmp_path / "v" / "a.mp4")}, "ego_head", "mine",
                          {"instruction": "clean up", "annotation_subtasks": subs})
    block = me.build_request(ep)["prompt"].split("THE DATASET'S ANNOTATION FOR THIS EPISODE")[1]
    assert "2.0-2.5s  times as text" in block and "1.5s  time with its unit" in block
    assert "no time  a flag, no time" in block and "0.2-1.0s  timed" in block
    assert 'Original recording span:' not in block
    assert json.loads((ep / 'context.json').read_text())['annotation_subtasks'][0] == subs[0]
    board = [board_build.dataset_label(x) for x in subs]
    assert [(x["t0"], x["t1"]) for x in board] == [(2.0, 2.5), (1.5, 1.5), (None, 2.0), (0.2, 1.0)]


def test_cut_plan_reserves_the_real_last_frame_tail_and_remaining_capacity():
    # The donor's 661-frame case previously chose 231s and left a 430s final request.
    t = np.arange(661, dtype=float)
    cuts = pieces.choose_cuts(t, np.zeros_like(t), 330)
    bounds = [0.] + [t[c['frame']] for c in cuts] + [661.]
    assert len(cuts) == 2
    assert max(np.diff(bounds)) <= 330
    # A single-frame tail counts, including nonzero presentation origins and uneven capture gaps.
    t = np.array([40., 40.2, 40.7, 41.2, 41.9, 42.3])
    cuts = pieces.choose_cuts(t, np.zeros_like(t), 1.3, end_s=42.8)
    bounds = [40.] + [t[c['frame']] for c in cuts] + [42.8]
    assert max(np.diff(bounds)) <= 1.3 + 1e-9
    assert all(0 < c['frame'] < len(t) for c in cuts)
    with pytest.raises(ValueError, match='frame boundary'):
        pieces.choose_cuts(np.array([0., 4.]), np.zeros(2), 2., end_s=5.)


def test_cut_artifacts_belong_to_their_explicit_source_boundary():
    start = {'category': 'starts_mid_task', 'issue': 'starts mid task', 't_s': None}
    end = {'category': 'truncated_episode', 'issue': 'the clip ends mid-task', 't_s': None}
    unknown = {'category': 'incomplete', 'issue': 'the intended stack is incomplete', 't_s': None}
    assert not pieces.is_cut_artifact(start, 1, 2, 0., 100., [100.])
    assert pieces.is_cut_artifact(start, 2, 2, 100., 200., [100.])
    assert pieces.is_cut_artifact(end, 1, 2, 0., 100., [100.])
    assert not pieces.is_cut_artifact(end, 2, 2, 100., 200., [100.])
    assert not pieces.is_cut_artifact(unknown, 1, 2, 0., 100., [100.])
    # A label emitted outside its own part cannot borrow the next part's internal edge.
    assert not pieces.is_cut_artifact({**end, 't_s': 100.}, 2, 3, 100., 200., [100., 200.])


@pytest.mark.parametrize('observed_outcome', ['failure', 'partial'])
def test_stitch_preserves_observed_incidents_with_unfinished_recovery_at_cuts(tmp_path, monkeypatch, observed_outcome):
    """A failed attempt and its recovery are one incident, even when a request cuts through it."""
    import copy
    monkeypatch.setattr(pieces, 'piece_max', lambda ctx: 1.0)
    upload = tmp_path / 'upload'
    recorder_folder(upload, n=90, duplicate_stamps=False)
    report = f.convert(upload, 'ego_head', tmp_path / 'episodes', 'fixture', 900)
    source = tmp_path / 'episodes' / report['episodes'][0]['episode_id']
    parts = pieces.write_pieces(source, tmp_path / 'pieces')
    contexts = [json.loads((part / 'context.json').read_text()) for part in parts]
    records = [_part_result(part, 'Seat the cup in the holder') for part in parts]
    assert len(parts) == 3
    for i, (context, record) in enumerate(zip(contexts, records)):
        duration = context['piece']['t1_s'] - context['piece']['t0_s']
        boundary_issue = {'category': 'incomplete_task', 'severity': 'medium',
            't_s': duration, 'end_s': duration, 'recovered': False,
            'issue': 'The clip ends before the task can finish.', 'evidence': 'head camera, final frame'}
        boundary_task = {'start_s': duration - .005, 'end_s': duration, 'task': 'Align the holder',
            'objects': ['holder'], 'outcome': 'partial', 'completed_at_s': None,
            'note': 'The clip ends while this task is still under way.'}
        record['labels'].update(operator_mistakes=[boundary_issue], tasks=[boundary_task],
                                timeline=[], key_events=[], recovery=[])
        if i == 1:
            # Incomplete structured output must not turn an independently described event into a cut artifact.
            mixed = ['The clip ends mid-task after the cup falls and breaks.',
                     'The clip ends before the operator can retrieve the cup that fell.']
            record['labels']['operator_mistakes'] += [{**boundary_issue, 'issue': issue,
                't_s': duration - .02, 'evidence': 'Head camera shows the cup on the floor.'} for issue in mixed]
            record['labels']['tasks'] += [{**boundary_task, 'task': f'Recover the cup {j}',
                'objects': ['cup'], 'outcome': 'failure', 'note': note} for j, note in enumerate(mixed)]
        if i != 1:
            failed_at = duration - .2
            record['labels']['timeline'] = [{'start_s': failed_at, 'end_s': duration,
                'hand': 'right', 'action': 'Try to seat the cup, then retrieve it',
                'object': 'cup', 'contribution': 'wasteful', 'progress': .5,
                'notes': 'The cup falls beside the holder; retrieval has not finished.'}]
            record['labels']['key_events'] = [{'t_s': failed_at, 'label': 'The cup falls beside the holder',
                'outcome': 'failure', 'note': 'Head camera shows the cup outside the intended holder.'}]
            if i == 0:
                record['labels']['key_events'] += [
                    {'t_s': duration - .02, 'label': label, 'outcome': 'failure'} for label in (
                        'The clip ends mid-task after the cup falls and breaks.',
                        'The clip ends before the operator can retrieve the cup that fell.',
                        'The clip ends before the task can finish.')]
                record['labels']['key_events'].append({'t_s': duration - .02, 'outcome': 'failure',
                    'label': 'The clip ends before the task can finish.',
                    'note': 'Head camera shows the cup falling beside the holder.'})
            record['labels']['recovery'] = [{'failure_t_s': failed_at, 'failure': 'The seating attempt fails',
                'recovered': False, 'recovered_at_s': None, 'correction': None}]
            record['labels']['operator_mistakes'].append({**boundary_issue,
                't_s': failed_at, 'issue': 'The cup falls; recovery is still under way at the end.',
                'evidence': 'Head camera shows the cup falling beside the holder.'})
            record['labels']['operator_mistakes'].append({**boundary_issue,
                't_s': failed_at, 'issue': 'The clip ends while recovery is still under way.',
                'evidence': 'Head camera shows the failed seating attempt and the cup beside the holder.'})
            record['labels']['tasks'].append({**boundary_task, 'task': 'Seat the cup',
                'start_s': .05, 'objects': ['cup'], 'outcome': observed_outcome,
                'note': 'The clip ends while recovery is still under way.'})
    original = copy.deepcopy(records)
    stitched = pieces.stitch(source, list(zip(contexts, records)))
    labels = stitched['labels']
    incidents = labels['operator_mistakes']
    assert [incident['part'] for incident in incidents if incident['issue'].startswith('The cup falls')] == [1, 3]
    assert [incident['part'] for incident in incidents if incident['issue'] == 'The clip ends before the task can finish.'] == [3]
    assert [incident['part'] for incident in incidents if incident['issue'] == 'The clip ends while recovery is still under way.'] == [1, 3]
    assert [incident['issue'] for incident in incidents if incident['part'] == 2] == mixed
    tasks = labels['tasks']
    assert [(task['source_piece']['index'], task['outcome']) for task in tasks if task['task'] == 'Seat the cup'] == [
        (1, observed_outcome), (3, observed_outcome)]
    assert [(task['source_piece']['index'], task['outcome']) for task in tasks if task['task'] == 'Align the holder'] == [
        (1, 'continues_past_cut'), (2, 'continues_past_cut'), (3, 'partial')]
    assert [task['outcome'] for task in tasks if task['task'].startswith('Recover the cup')] == ['failure', 'failure']
    assert [event['outcome'] for event in labels['key_events']] == ['failure'] * 5
    noted_event = next(event for event in labels['key_events'] if event.get('note') ==
                       'Head camera shows the cup falling beside the holder.')
    assert noted_event['label'] == 'The clip ends before the task can finish.'
    assert noted_event['source_piece']['index'] == 1
    assert len(labels['recovery']) == 2 and all(not row['recovered'] for row in labels['recovery'])
    assert [row['list'] for row in labels['_excluded']].count('operator_mistakes') == 2
    assert [row['label'] for row in labels['_excluded'] if row['list'] == 'key_events'] == [
        'The clip ends before the task can finish.']
    assert records == original


def test_blanket_continuation_joins_only_on_a_seam_verdict_that_says_same_task():
    # The UMI pilot split blanket work at 304.833s. Where the halves meet is known, but a task can also begin right at
    # a cut, so the halves join only when a check of the seam says they are one task; nothing the parts wrote decides.
    source = lambda index, lo, hi: {'index': index, 't0_s': lo, 't1_s': hi}
    cut = 304.833
    first = {'start_s': 190., 'end_s': 304.8, 'task': 'Unfold the blanket over the bed',
             'objects': ['floral fleece blanket', 'penguin plush'], 'outcome': 'partial',
             'success_predicate': 'The blanket lies flat over the whole bed.',
             'note': 'The clip ends while the task is still under way.', 'completed_at_s': None,
             'source_piece': source(1, 0., cut)}
    second = {'start_s': cut, 'end_s': 313.833, 'task': 'Settle the floral blanket over the bed',
              'objects': ['floral blanket', 'bed'], 'outcome': 'success', 'completed_at_s': 313.4,
              'success_predicate': 'The last corner is tucked in.',
              'goal_reached_at_s': 313.4, 'note': 'Continues the task from the previous part.',
              'source_piece': source(2, cut, 500.)}
    same = {cut: {'same_task': True, 'evidence': 'The same blanket is spread over the bed on both sides of the cut.'}}
    different = {'304.833': {'same_task': False, 'evidence': 'The blanket is put down and a new task starts.'}}
    unclear = {cut: {'same_task': 'unclear', 'evidence': 'The operator blocks the view.'}}
    originals = json.loads(json.dumps([first, second]))
    # without a verdict the halves stay apart, exactly as they came
    assert pieces.join_across_cuts([first, second], [cut]) == [first, second]
    joined = pieces.join_across_cuts([first, second], [cut], same)
    assert len(joined) == 1 and joined[0]['end_s'] == 313.833
    # the joined task keeps what it set out to do from the left half and how it ended from the right half
    assert joined[0]['task'] == first['task'] and joined[0]['success_predicate'] == first['success_predicate']
    assert (joined[0]['outcome'], joined[0]['completed_at_s'], joined[0]['goal_reached_at_s']) == (
        'success', 313.4, 313.4)
    assert joined[0]['joined_from'] == originals
    assert joined[0]['cut_joins'] == [{'cut_s': cut, 'by': 'seam_check', **same[cut]}]
    assert [first, second] == originals
    # A new goal right at the cut stays apart without a verdict, with a "different" verdict and with an unclear one,
    # and joins with "same": the verdict decides, never the task text.
    new_goal = {**second, 'task': 'Put the blanket in the hamper', 'note': 'Starts a new task.'}
    assert pieces.join_across_cuts([first, new_goal], [cut]) == [first, new_goal]
    apart = pieces.join_across_cuts([first, new_goal], [cut], different)
    assert [t['task'] for t in apart] == [first['task'], new_goal['task']]
    assert all(t['cut_seams'] == [{'cut_s': cut, **different['304.833']}] for t in apart)
    assert [t['task'] for t in pieces.join_across_cuts([first, new_goal], [cut], unclear)] == [
        first['task'], new_goal['task']]
    assert len(pieces.join_across_cuts([first, new_goal], [cut], same)) == 1
    # an answer that is not the boolean true joins nothing
    assert len(pieces.join_across_cuts([first, second], [cut], {cut: {'same_task': 'true'}})) == 2
    # Object names are what a part wrote, so they decide nothing either way.
    wrong_object = {**second, 'objects': ['blue blanket']}
    assert pieces.join_across_cuts([first, wrong_object], [cut]) == [first, wrong_object]
    assert len(pieces.join_across_cuts([first, wrong_object], [cut], same)) == 1
    no_owner = {k: v for k, v in second.items() if k != 'source_piece'}
    assert pieces.join_across_cuts([first, no_owner], [cut], same) == [first, no_owner]
    # A left task its part saw end, or one that ends well before the cut, is no candidate, whatever the verdict.
    assert len(pieces.join_across_cuts([{**first, 'completed_at_s': 300.}, second], [cut], same)) == 2
    assert len(pieces.join_across_cuts([{**first, 'end_s': cut - pieces.JOIN_GUARD_S - .5}, second], [cut], same)) == 2
    # Joining an earlier cut does not erase the fact that the task is still open at a later cut.
    middle = {**second, 'end_s': 500., 'outcome': 'partial', 'completed_at_s': None, 'goal_reached_at_s': None,
              'note': second['note'] + ' The clip ends while this task is still under way.'}
    both = {**same, 500.: {'same_task': True, 'evidence': 'The blanket is still being spread at 500s.'}}
    open_task = pieces.mark_cut_tasks(pieces.join_across_cuts([first, middle], [cut, 500.], both), [cut, 500.])
    assert len(open_task) == 1 and open_task[0]['outcome'] == 'continues_past_cut'
    assert len(open_task[0]['joined_from']) == 2 and open_task[0]['cut_continuation']['t_s'] == 500.
    last = {**second, 'start_s': 500., 'end_s': 530., 'completed_at_s': 529., 'goal_reached_at_s': 529.,
            'source_piece': source(3, 500., 600.)}
    all_parts = pieces.join_across_cuts([first, middle, last], [cut, 500.], both)
    assert len(all_parts) == 1 and len(all_parts[0]['joined_from']) == 3
    assert all_parts[0]['goal_reached_at_s'] == 529. and [j['cut_s'] for j in all_parts[0]['cut_joins']] == [cut, 500.]
    assert len(pieces.join_across_cuts([first, middle, last], [cut, 500.], same)) == 2
    # Missing and ambiguous parts never borrow another candidate, whatever the verdict says.
    gap = {**second, 'source_piece': source(3, cut, 500.)}
    assert pieces.join_across_cuts([first, gap], [cut], same) == [first, gap]
    ambiguous = pieces.join_across_cuts([first, second, {**second, 'task': 'Another blanket activity'}], [cut], same)
    assert len(ambiguous) == 3
    two_open = pieces.join_across_cuts([first, {**first, 'task': 'Hold the pillow'}, second], [cut], same)
    assert len(two_open) == 3


def test_an_unclear_task_ending_at_our_cut_continues_past_it_whatever_its_note_says():
    """A UMI recording's real part-1 task (fob_47b204-1da855ba-867-234), in the model's own words: its span ends at the
    part's internal cut, so its unclear outcome is the cut's, even with a fumble inside the task and a note that names
    the cut in words no phrase test knows. A partial outcome is a judgment and keeps the stricter test."""
    cut = 426.559
    task = {'start_s': 0.0, 'end_s': 426.53, 'task': 'Thread the loose colored rings onto the blue peg board.',
            'objects': ['colored rings', 'blue peg board'], 'outcome': 'unclear', 'success_predicate': 'All rings on posts.',
            'completed_at_s': None,
            'note': 'Only one continuing objective is visible in this window. Multiple ring stacks are built, but loose '
                    'rings remain and the last ring is still being aligned when this preparation part ends; the eventual '
                    'outcome is outside the supplied window.',
            'source_piece': {'index': 1, 't0_s': 0.0, 't1_s': cut}}
    labels = {'timeline': [{'start_s': 120.0, 'end_s': 124.0, 'contribution': 'wasteful',
                            'source_piece': {'index': 1, 't0_s': 0.0, 't1_s': cut}}]}
    [marked] = pieces.mark_cut_tasks([task], [cut], labels)
    assert marked['outcome'] == 'continues_past_cut' and marked['outcome_in_part'] == 'unclear'
    assert marked['cut_continuation']['t_s'] == cut
    partial = {**task, 'outcome': 'partial'}
    assert pieces.mark_cut_tasks([partial], [cut], labels) == [partial]
    away = {**task, 'end_s': 300.0}
    assert pieces.mark_cut_tasks([away], [cut], labels) == [away]
    reached = {**task, 'goal_reached_at_s': 410.0}
    assert pieces.mark_cut_tasks([reached], [cut], labels) == [reached]


def test_stitch_retains_near_cut_failures_and_complete_task_outcome_evidence(tmp_path, monkeypatch):
    _, src, parts = _three_parts(tmp_path, monkeypatch)
    pcs = [json.loads((p / 'context.json').read_text()) for p in parts]
    boundary = pcs[0]['piece']['t1_s']
    records = [_part_result(p, 'fold the blanket') for p in parts]
    records[0]['labels']['completion'] = {'task_completed': 'partial', 'completed_at_s': None,
                                          'reason': 'The clip ends while folding is still under way.'}
    records[0]['labels']['key_events'] = [
        {'t_s': boundary - .05, 'outcome': 'failure', 'label': 'The cup falls and breaks.'},
        {'t_s': boundary, 'outcome': 'failure', 'label': 'The clip ends while the cup falls and breaks.'},
        {'t_s': boundary, 'outcome': 'unclear', 'label': 'The clip ends before the fold can finish.'}]
    records[1]['labels']['completion'] = {'task_completed': 'success_then_undone',
        'completed_at_s': .3, 'goal_reached_at_s': .3, 'undone_at_s': .6, 'undone_by': 'operator',
        'reason': 'The operator knocks the completed stack over.'}
    result = pieces.stitch(src, list(zip(pcs, records)))
    assert result['labels']['tasks'] == []
    local = result['labels']['part_assessments']
    assert local[0]['completion'] == records[0]['labels']['completion']
    assert local[1]['completion']['task_completed'] == 'success_then_undone'
    assert local[1]['completion']['goal_reached_at_s'] == round(boundary + .3, 3)
    assert local[1]['completion']['undone_at_s'] == round(boundary + .6, 3)
    assert local[1]['completion']['undone_by'] == 'operator'
    assert [event['label'] for event in result['labels']['key_events']] == [
        'The cup falls and breaks.', 'The clip ends while the cup falls and breaks.']
    excluded = result['labels']['_excluded']
    assert len(excluded) == 1 and excluded[0]['excluded_by'] == 'piece_cut'
    assert excluded[0]['source_piece']['index'] == 1


def test_saved_parts_cannot_reuse_an_oversized_final_window(tmp_path, monkeypatch):
    _, src, _ = _three_parts(tmp_path, monkeypatch)
    job = tmp_path / 'next_job'
    job.mkdir()
    saved = pieces.write_units(job, src.parent)
    last = job / 'pieces' / saved[src.name][-1] / 'context.json'
    ctx = json.loads(last.read_text())
    ctx['duration_s'] = 1.2
    last.write_text(json.dumps(ctx))
    with pytest.raises(RuntimeError, match='saved part exceeds the request duration limit'):
        pieces.write_units(job, src.parent)
    assert json.loads(last.read_text())['duration_s'] == 1.2


def test_independent_request_goals_do_not_become_episode_tasks_or_progress(tmp_path, monkeypatch):
    """Local success for storing then retrieving the same object is not one episode verdict.

    Request accounting covers every view/grid/detail image, including a failed reply.
    Missing receipts remain unknown instead of silently reporting the first request.
    """
    import copy
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    contexts = [json.loads((part / 'context.json').read_text()) for part in parts]
    records = [_part_result(part, summary) for part, summary in zip(parts,
               ('Store the cup', 'Retrieve the cup', 'Close the cabinet'))]
    for i, record in enumerate(records):
        record['config'].update(n_frames_sent=6 + i, n_image_parts=2 + i,
                                contact_detail_s=[.25], views=['exo', 'left', 'right'])
        record['labels']['timeline'][0].update(object='cup', progress=1.0 if i == 0 else .03)
        record['labels']['completion'].update(success_predicate=record['labels']['task_summary'], completed_at_s=.6)
    original = copy.deepcopy(records)
    stitched = pieces.stitch(src, list(zip(contexts, records)))
    labels = stitched['labels']
    assert 'task_mode' not in stitched['annotation_policy']
    assert stitched['given_prompt'] == json.loads((src / 'context.json').read_text())['instruction']
    assert labels['tasks'] == [] and labels['completion']['task_completed'] == 'unclear'
    assert labels['progress_scope'] == 'request_parts'
    assert all(step['progress'] is None for step in labels['timeline'])
    # each step keeps its progress toward its own part's sub-task, for the task it belongs to (board/scores.py)
    assert [step.get('progress_in_part') for step in labels['timeline'] if step.get('object') == 'cup'] == [1.0, .03, .03]
    assert [assessment['progress'][0]['progress'] for assessment in labels['part_assessments']] == [1.0, .03, .03]
    for i, assessment in enumerate(labels['part_assessments']):
        offset = contexts[i]['piece']['t0_s']
        assert assessment['completion']['completed_at_s'] == round(offset + .6, 3)
        assert assessment['completion']['success_predicate'] == records[i]['labels']['task_summary']
        assert assessment['progress'][0]['source_piece'] == {
            'index': i + 1, 't0_s': offset, 't1_s': contexts[i]['piece']['t1_s']}
    assert stitched['config']['n_frames_sent'] == 21 and stitched['config']['n_image_parts'] == 9
    assert stitched['config']['contact_detail_s'] == [round(context['piece']['t0_s'] + .25, 3) for context in contexts]
    assert stitched['config']['n_timesteps'] == 6
    assert records == original
    # A failed model reply still has a known input receipt, but contributes no labels.
    records[1]['parse_ok'] = False
    for part, record in zip(parts, records):
        (job / 'run' / 'out' / f'{part.name}.json').write_text(json.dumps(record))
    pieces.stitch_run(job, src.parent, {src.name: [part.name for part in parts]}, tmp_path / 'joined')
    incomplete = json.loads((tmp_path / 'joined' / f'{src.name}.json').read_text())
    assert len(incomplete['labels']['part_assessments']) == 2 and len(incomplete['stitched']['missing']) == 1
    assert incomplete['labels']['completion']['task_completed'] == 'unclear'
    assert incomplete['config']['n_frames_sent'] == 21 and incomplete['config']['n_image_parts'] == 9
    assert incomplete['config']['n_timesteps'] == 6
    assert incomplete['config']['pieces'][1]['config'] == records[1]['config']
    assert incomplete['usage']['est_cost_usd'] == 1.5


def test_stitch_preserves_session_policy_and_segmented_jaw_receipts(tmp_path, monkeypatch):
    from label import policy
    from label.evidence_access import source_proof
    from prepare import jaws
    from test_jaws import marker_frames, DECLARATION
    from board.to_board import convert
    _, src, parts = _three_parts(tmp_path, monkeypatch)
    path = src / 'context.json'
    parent = json.loads(path.read_text())
    # a job saved in the session mode, before every recording was labelled as a list of tasks, resumed
    parent['annotation_policy'] = policy.resolve(parent, task_mode='session', dataset_note='Assembly session.')
    path.write_text(json.dumps(parent))
    contexts = [json.loads((part / 'context.json').read_text()) for part in parts]
    measured = jaws.analyse_frames(marker_frames, np.arange(18) * .04, DECLARATION)
    assert measured['status'] == 'measured' and measured['events']
    records = [_part_result(part, 'assemble') for part in parts]
    for index, record in enumerate(records):
        record['annotation_policy'] = parent['annotation_policy']
        record['input_identity'] = {'source_proof': source_proof(contexts[index], parts[index])}
        record['jaw_evidence'] = {'right': measured if index == 2 else {
            'status': 'abstained', 'reason': 'insufficient_marker_coverage', 'events': [],
            'quality': {'observed_fraction': .3}}}
    # These are model-emitted activities, not fallback tasks made from requests.
    records[0]['labels']['tasks'] = [{'start_s': .1, 'end_s': .6, 'task': 'Seat the latch',
        'objects': ['latch'], 'outcome': 'success', 'completed_at_s': .6}]
    records[2]['labels']['tasks'] = [{'start_s': .1, 'end_s': .6, 'task': 'Tighten the screw',
        'objects': ['screw'], 'outcome': 'success', 'completed_at_s': .6}]
    missing = [{'part': 2, 't0_s': contexts[1]['piece']['t0_s'],
                't1_s': contexts[1]['piece']['t1_s'], 'why': 'no reply'}]
    result = pieces.stitch(src, [(contexts[i], records[i]) for i in (0, 2)], missing)
    assert result['annotation_policy'] == parent['annotation_policy']
    assert result['prompt_mode'] == 'session'
    tasks = result['labels']['tasks']
    assert [task['task'] for task in tasks] == ['Seat the latch', 'Tighten the screw']
    assert tasks[1]['completed_at_s'] == round(contexts[2]['piece']['t0_s'] + .6, 3)
    assert [task['source_piece']['index'] for task in tasks] == [1, 3]
    # a session labelled in parts has no recording-wide goal, as one labelled in one request has none
    completion = result['labels']['completion']
    assert completion['task_completed'] is None
    assert completion['reason'] == "a session of several tasks; each task's outcome is under tasks; part 2 has no labels, as no reply"
    evidence = result['jaw_evidence']['right']
    assert evidence['status'] == 'segmented' and evidence['missing'] == missing
    assert evidence['source_proof'] == source_proof(parent, src)
    assert evidence['parts'][0]['record']['status'] == 'abstained'
    last = evidence['parts'][1]
    assert last['record'] == measured and last['source_proof'] == records[2]['input_identity']['source_proof']
    assert last['source_location']
    offset = contexts[2]['piece']['t0_s']
    assert last['times_s'] == pytest.approx(np.asarray(measured['times_s']) + offset)
    assert last['events'][0]['start_s'] == pytest.approx(measured['events'][0]['start_s'] + offset)
    assert last['events'][0]['source_piece']['index'] == 3
    assert measured['times_s'][0] == 0
    assert result['config']['pieces'][2]['annotation_policy'] == parent['annotation_policy']
    assert convert(result)['jaw_evidence'] == result['jaw_evidence']
    failed = [(context, {**record, 'parse_ok': False}) for context, record in zip(contexts, records)]
    gaps = [{'part': pc['piece']['index'], 't0_s': pc['piece']['t0_s'],
             't1_s': pc['piece']['t1_s'], 'why': 'reply did not parse'} for pc in contexts]
    no_labels = pieces.unlabelled(src, failed, gaps)
    assert no_labels['parse_ok'] is False and no_labels['prompt_mode'] == 'session'
    assert no_labels['annotation_policy'] == parent['annotation_policy']
    assert no_labels['jaw_evidence']['right']['parts'][2]['record'] == measured


def test_stitched_objects_keep_their_own_part_unless_a_task_across_the_cut_shows_they_are_one(tmp_path, monkeypatch):
    """Objects were merged across parts by lowercase name: part 1's red cup and part 2's green "Cup" became one object,
    and part 1's second cup was dropped. Each part's objects now stay apart with their part, so all three cups remain."""
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    out = job / "run" / "out"
    objects = {0: [{"name": "cup", "attributes": ["red"]}, {"name": "cup", "attributes": ["blue"]}],
               1: [{"name": "Cup", "attributes": ["green"]}], 2: []}
    for k, part in enumerate(parts):
        result = _part_result(part, "reach")
        result["labels"]["scene"] = {"objects": objects[k], "setting": "table"}
        (out / f"{part.name}.json").write_text(json.dumps(result))
    pieces.stitch_run(job, src.parent, {src.name: [p.name for p in parts]}, tmp_path / "final")
    r = json.loads((tmp_path / "final" / f"{src.name}.json").read_text())
    got = r["labels"]["scene"]["objects"]
    assert [(o["name"], o["attributes"]) for o in got] == [("cup", ["red"]), ("cup", ["blue"]), ("Cup", ["green"])]
    assert [o["source_piece"]["index"] for o in got] == [1, 1, 2]


def test_an_object_a_task_carries_across_the_cut_is_one_object_with_each_parts_record():
    # The UMI pilot's blanket task joined at 304.833s names the blanket in both halves, each part naming it its own way.
    source = lambda index, lo, hi: {'index': index, 't0_s': lo, 't1_s': hi}
    first = {'start_s': 190., 'end_s': 304.8, 'task': 'Unfold the blanket over the bed',
             'objects': ['floral fleece blanket', 'penguin plush'], 'outcome': 'partial',
             'note': 'The clip ends while the task is still under way.', 'source_piece': source(1, 0., 304.833)}
    second = {'start_s': 304.833, 'end_s': 313.833, 'task': 'Settle the floral blanket over the bed',
              'objects': ['floral blanket', 'bed'], 'outcome': 'success', 'completed_at_s': 313.4,
              'note': 'Continues the task from the previous part.', 'source_piece': source(2, 304.833, 500.)}
    # the halves are one task because a seam check said so; their notes and object names decide nothing
    seams = {304.833: {'same_task': True, 'evidence': 'one blanket'}}
    tasks = pieces.join_across_cuts([first, second], [304.833], seams)
    assert len(tasks) == 1
    one, two = source(1, 0., 304.833), source(2, 304.833, 500.)
    objects = [(1, {'name': 'floral fleece blanket', 'source_piece': one}), (1, {'name': 'penguin plush', 'source_piece': one}),
               (1, {'name': 'cup', 'source_piece': one}),
               (2, {'name': 'floral blanket', 'source_piece': two}), (2, {'name': 'bed', 'source_piece': two}),
               (2, {'name': 'Cup', 'source_piece': two})]
    joined = pieces.join_scene_objects(objects, tasks)
    assert joined == [{'name': 'floral fleece blanket', 'source_piece': one,
                       'continues': [{'name': 'floral blanket', 'source_piece': two}]},
                      {'name': 'penguin plush', 'source_piece': one}, {'name': 'cup', 'source_piece': one},
                      {'name': 'bed', 'source_piece': two}, {'name': 'Cup', 'source_piece': two}]
    # Two blankets in the second part: the task's name no longer picks out one, so nothing is joined.
    crowded = objects + [(2, {'name': 'floral blanket', 'source_piece': two})]
    assert all('continues' not in o for o in pieces.join_scene_objects(crowded, tasks))
    # Without the joined task the names alone join nothing.
    assert all('continues' not in o for o in pieces.join_scene_objects(objects, [first, second]))


def test_a_join_across_the_cut_never_puts_two_objects_of_one_part_into_one():
    """The join was transitive: part 1's red cup and blue cup each answered to part 2's cup, so all three became one
    object, red cup, continued by blue cup and cup."""
    one, two = {'index': 1, 't0_s': 0., 't1_s': 5.}, {'index': 2, 't0_s': 5., 't1_s': 9.}
    objects = [(1, {'name': 'red cup', 'source_piece': one}), (1, {'name': 'blue cup', 'source_piece': one}),
               (2, {'name': 'cup', 'source_piece': two}), (2, {'name': 'tray', 'source_piece': two})]
    half = lambda names, source: {'objects': names, 'source_piece': source}
    stack = {'task': 'stack the cups', 'joined_from': [half(['red cup', 'blue cup'], one), half(['cup'], two)]}
    # the task's cup answers to two of its first half's names, so it names neither of them
    assert pieces.join_scene_objects(objects, [stack]) == [o for _, o in objects]
    # two tasks, each pairing one of the cups with part 2's cup, contradict each other, in either order
    red = {'task': 'move the red cup', 'joined_from': [half(['red cup'], one), half(['cup'], two)]}
    blue = {'task': 'move the blue cup', 'joined_from': [half(['blue cup'], one), half(['cup'], two)]}
    assert pieces.join_scene_objects(objects, [red, blue]) == pieces.join_scene_objects(objects, [blue, red]) == [
        o for _, o in objects]
    # one task alone joins its cup
    joined = pieces.join_scene_objects(objects, [red])
    assert [(o['name'], [c['name'] for c in o.get('continues', [])]) for o in joined] == [
        ('red cup', ['cup']), ('blue cup', []), ('tray', [])]
    # joins chain across three parts into one object, one record of each part
    three = {'index': 3, 't0_s': 9., 't1_s': 12.}
    chained = objects + [(3, {'name': 'blue cup', 'source_piece': three})]
    on = {'task': 'carry the cup on', 'joined_from': [half(['cup'], two), half(['blue cup'], three)]}
    joined = pieces.join_scene_objects(chained, [red, on])
    assert [(o['name'], [c['name'] for c in o.get('continues', [])]) for o in joined] == [
        ('red cup', ['cup', 'blue cup']), ('blue cup', []), ('tray', [])]
    # the normal join: each cup is named in both halves and answers to one name in the other half
    objects2 = objects[:2] + [(2, {'name': 'red cup', 'source_piece': two}), (2, {'name': 'blue cup', 'source_piece': two})]
    both = {'task': 'stack the cups', 'joined_from': [half(['red cup', 'blue cup'], one),
                                                      half(['red cup', 'blue cup'], two)]}
    joined = pieces.join_scene_objects(objects2, [both])
    assert [(o['name'], [c['name'] for c in o.get('continues', [])]) for o in joined] == [
        ('red cup', ['red cup']), ('blue cup', ['blue cup'])]
    assert joined[0]['continues'][0]['source_piece'] == two


def test_a_half_naming_its_object_twice_still_joins_it_across_the_cut():
    """The uniqueness count ran over every name a half listed, so a half naming its one cup twice ("cup" and "the cup")
    read as two cups and the object never joined. A half's names are counted once each after normalizing."""
    one, two = {'index': 1, 't0_s': 0., 't1_s': 5.}, {'index': 2, 't0_s': 5., 't1_s': 9.}
    objects = [(1, {'name': 'cup', 'source_piece': one}), (2, {'name': 'cup', 'source_piece': two})]
    half = lambda names, source: {'objects': names, 'source_piece': source}
    for names in (['cup'], ['cup', 'the cup'], ['cups', 'cup'], ['Cup', 'cup'], ['a cup', 'cup', 'the cups']):
        for task in ({'task': 'carry the cup', 'joined_from': [half(names, one), half(['cup'], two)]},
                     {'task': 'carry the cup', 'joined_from': [half(['cup'], one), half(names, two)]}):
            joined = pieces.join_scene_objects(objects, [task])
            assert [(o['name'], [c['name'] for c in o.get('continues', [])]) for o in joined] == [('cup', ['cup'])], names


def test_a_name_answering_to_two_names_of_the_other_half_joins_nothing_even_when_one_part_lists_one():
    """Part 1 lists only the red cup, but the task's first half names a red cup and a blue cup: its second half's cup
    could be either, so it joins neither, whichever half names the two."""
    one, two = {'index': 1, 't0_s': 0., 't1_s': 5.}, {'index': 2, 't0_s': 5., 't1_s': 9.}
    half = lambda names, source: {'objects': names, 'source_piece': source}
    objects = [(1, {'name': 'red cup', 'source_piece': one}), (2, {'name': 'cup', 'source_piece': two})]
    task = {'task': 'stack the cups', 'joined_from': [half(['red cup', 'blue cup'], one), half(['cup'], two)]}
    assert pieces.join_scene_objects(objects, [task]) == [o for _, o in objects]
    objects = [(1, {'name': 'cup', 'source_piece': one}), (2, {'name': 'red cup', 'source_piece': two})]
    task = {'task': 'stack the cups', 'joined_from': [half(['cup'], one), half(['red cup', 'blue cup'], two)]}
    assert pieces.join_scene_objects(objects, [task]) == [o for _, o in objects]
    # the same task with each half naming one cup joins it
    task = {'task': 'carry the cup', 'joined_from': [half(['cup'], one), half(['red cup'], two)]}
    assert [len(o.get('continues', [])) for o in pieces.join_scene_objects(objects, [task])] == [1]


def test_halves_of_parts_that_are_not_next_to_each_other_join_nothing():
    one, three = {'index': 1, 't0_s': 0., 't1_s': 5.}, {'index': 3, 't0_s': 9., 't1_s': 12.}
    objects = [(1, {'name': 'cup', 'source_piece': one}), (3, {'name': 'cup', 'source_piece': three})]
    task = {'task': 'carry the cup', 'joined_from': [{'objects': ['cup'], 'source_piece': one},
                                                     {'objects': ['cup'], 'source_piece': three}]}
    assert pieces.join_scene_objects(objects, [task]) == [o for _, o in objects]


def test_on_random_recordings_a_joined_object_holds_one_record_per_part_in_any_task_order():
    import random
    rng = random.Random(61)
    words = ['cup', 'red cup', 'blue cup', 'the cup', 'cups', 'tray', 'red tray', 'bowl', 'blue bowl', 'red']
    joins = 0
    for _ in range(3000):
        n_parts = rng.randint(2, 4)
        source = {i: {'index': i, 't0_s': float(i), 't1_s': i + 1.} for i in range(1, n_parts + 1)}
        objects = [(i, {'name': rng.choice(words), 'source_piece': source[i]}) for i in range(1, n_parts + 1)
                   for _ in range(rng.randint(1, 3))]
        tasks = []
        for _ in range(rng.randint(1, 4)):
            start = rng.randint(1, n_parts - 1)
            tasks.append({'task': 't', 'joined_from': [{'objects': rng.sample(words, rng.randint(1, 3)),
                                                        'source_piece': source[p]}
                                                       for p in range(start, rng.randint(start + 1, n_parts) + 1)]})
        joined = pieces.join_scene_objects(objects, tasks)
        assert joined == pieces.join_scene_objects(objects, tasks[::-1])
        for o in joined:
            parts = [r['source_piece']['index'] for r in [o] + o.get('continues', [])]
            assert len(parts) == len(set(parts)), (objects, tasks)
            joins += len(parts) > 1
    assert joins > 300      # the recordings do join objects, so the check has something to hold


def test_the_board_names_the_parts_a_stitched_object_came_from():
    from board import to_board
    one, two = {'index': 1, 't0_s': 0., 't1_s': 5.}, {'index': 2, 't0_s': 5., 't1_s': 9.}
    result = {'episode_dir': 'episode_x', 'parse_ok': True, 'labels': {**complete_reply.labels(), 'scene': {'objects': [
        {'name': 'blanket', 'source_piece': one, 'continues': [{'name': 'floral blanket', 'source_piece': two}]},
        {'name': 'cup', 'attributes': ['red'], 'source_piece': two}, {'name': 'plate'}]}}}
    assert to_board.convert(result, 'demo')['objects'] == [
        {'name': 'blanket', 'color': None, 'parts': [1, 2]}, {'name': 'cup', 'color': 'red', 'parts': [2]},
        {'name': 'plate', 'color': None}]


def _tasks_part(p: Path, ctx: dict, tasks: list) -> dict:
    """A part's record as the harness writes it in the tasks format: label_schema, and the part's completion derived
    from its own tasks."""
    from board.label_projection import recording_completion
    record = _part_result(p, 'stack the cups')
    record['label_schema'] = 'tasks'
    record['labels'].update(tasks=tasks, completion=recording_completion(tasks))
    return record


def _cup_task(ctx, *, start=False, end=False, **fields):
    """A task over the whole part on its own clock; start and end say it continues across that cut."""
    span = round(ctx['piece']['t1_s'] - ctx['piece']['t0_s'], 3)
    note = ' '.join([*(['Continues the task from the previous part.'] if start else []),
                     *(['The clip ends while the task is still under way.'] if end else [])])
    return {'start_s': 0.0, 'end_s': span, 'task': 'stack the cups', 'objects': ['cup'],
            'judged_against': 'instruction', 'success_predicate': 'three cups stacked', 'note': note,
            'outcome': 'partial' if end else 'success', 'completed_at_s': None if end else span - .05, **fields}


def _without_instruction(src: Path) -> None:
    """The recording as a dataset that ships no instruction gives it."""
    ctx = json.loads((src / 'context.json').read_text())
    ctx['instruction'] = ''
    (src / 'context.json').write_text(json.dumps(ctx))


def test_in_the_tasks_format_one_task_joined_across_the_cuts_is_the_recordings_verdict(tmp_path, monkeypatch):
    """With no instruction a recording of one task takes that task's verdict, also when the task is joined across the
    cuts; with an instruction no part sees the whole asked work, so the recording stays unclear, as an episode labelled
    in parts always was, and its joined task is listed beside that verdict."""
    _, src, parts = _three_parts(tmp_path, monkeypatch)
    contexts = [json.loads((part / 'context.json').read_text()) for part in parts]
    tasks = [[_cup_task(contexts[0], end=True)], [_cup_task(contexts[1], start=True, end=True)],
             [_cup_task(contexts[2], start=True)]]
    records = [_tasks_part(p, c, t) for p, c, t in zip(parts, contexts, tasks)]
    # the parts touch at each cut with no shared seconds, so a seam check decides each join; without its verdicts the
    # task stays one row per part
    seams = {c['piece']['t0_s']: {'same_task': True, 'evidence': 'the same cups on both sides'} for c in contexts[1:]}
    assert len(pieces.stitch(src, list(zip(contexts, records)))['labels']['tasks']) == 3
    given = pieces.stitch(src, list(zip(contexts, records)), seams=seams)
    (task,) = given['labels']['tasks']
    assert task['outcome'] == 'success' and task['judged_against'] == 'instruction' and len(task['joined_from']) == 3
    completion = given['labels']['completion']
    assert (completion['task_completed'], completion['basis']) == ('unclear', 'parts')
    assert 'do not establish a whole-recording outcome' in completion['reason']
    assert given['label_schema'] == 'tasks' and given['prompt_mode'] == 'given'
    from board.label_projection import shown_tasks, verdict_or_tasks
    assert verdict_or_tasks(given['labels']) == ('unclear', []) and shown_tasks(given['labels']) == [task]
    _without_instruction(src)
    own = [[{**t, 'judged_against': 'own_end_state'} for t in part] for part in tasks]
    records = [_tasks_part(p, c, t) for p, c, t in zip(parts, contexts, own)]
    stitched = pieces.stitch(src, list(zip(contexts, records)), seams=seams)
    (task,) = stitched['labels']['tasks']
    completion = stitched['labels']['completion']
    assert task['outcome'] == 'success' and task['judged_against'] == 'own_end_state'
    assert (completion['task_completed'], completion['from_tasks'], completion['judged_against'],
            completion['basis']) == ('success', 1, 'own_end_state', 'one_task')
    assert completion['completed_at_s'] == task['completed_at_s']
    assert stitched['label_schema'] == 'tasks' and stitched['prompt_mode'] == 'inferred'
    # a part with no labels may hold more of the work: no verdict, and the reason names the part
    missing = [{'part': 3, 't0_s': contexts[2]['piece']['t0_s'], 't1_s': contexts[2]['piece']['t1_s'],
                'why': 'no reply'}]
    open_end = pieces.stitch(src, list(zip(contexts[:2], records[:2])), missing, seams=seams)
    completion = open_end['labels']['completion']
    assert completion['task_completed'] is None and completion['from_tasks'] == 1
    assert completion['reason'] == '1 task in the parts with labels; part 3 has no labels, as no reply'


def test_in_the_tasks_format_a_task_open_at_the_last_cut_or_parts_of_two_formats_give_no_verdict(tmp_path, monkeypatch):
    _, src, parts = _three_parts(tmp_path, monkeypatch)
    _without_instruction(src)
    contexts = [json.loads((part / 'context.json').read_text()) for part in parts]
    tasks = [[_cup_task(contexts[0], end=True, judged_against='own_end_state')],
             [_cup_task(contexts[1], start=True, end=True, judged_against='own_end_state')], []]
    records = [_tasks_part(p, c, t) for p, c, t in zip(parts, contexts, tasks)]
    seams = {contexts[1]['piece']['t0_s']: {'same_task': True, 'evidence': 'the same cups on both sides'}}
    stitched = pieces.stitch(src, list(zip(contexts, records)), seams=seams)
    (task,) = stitched['labels']['tasks']
    completion = stitched['labels']['completion']
    assert task['outcome'] == 'continues_past_cut'
    assert completion['task_completed'] is None and completion['from_tasks'] == 1 and 'cut' in completion['reason']
    # one part asked before the tasks format: the parts give no single reading
    older = _part_result(parts[2], 'stack the cups')
    mixed = pieces.stitch(src, list(zip(contexts, records[:2] + [older])))
    assert mixed['labels']['completion']['task_completed'] is None
    assert mixed['labels']['completion']['reason'] == 'parts labelled with different output formats'
    assert 'label_schema' not in mixed and mixed['prompt_mode'] == 'inferred'


# ---- shared windows at the cuts and seam verdicts ---------------------------------------------------------------

def test_cuts_planned_with_a_shared_window_keep_every_part_within_the_request_limit():
    """A 1,000s recording at the 450s limit: neighbouring parts share 15s around each cut, the shared seconds count
    toward each part's limit, and each window is exactly where one part ends and the next begins."""
    rng = np.random.default_rng(7)
    t = np.arange(0, 1000, 1 / 30)
    window = pieces.overlap_for(450.)
    assert window == pieces.OVERLAP_S == 15.
    cuts = pieces.choose_cuts(t, rng.random(len(t)), 450. - window, 1000.)
    spans, windows = pieces.part_spans(t, cuts, len(t), window)
    ends = [float(t[b]) if b < len(t) else 1000. for _, b in spans]
    assert len(spans) == 3 and max(e - float(t[a]) for (a, _), e in zip(spans, ends)) <= 450.
    for k, (cut, w) in enumerate(zip(cuts, windows)):
        assert w['t0_s'] == round(float(t[spans[k + 1][0]]), 3) and w['t1_s'] == round(ends[k], 3)
        assert w['t0_s'] < cut['t_s'] < w['t1_s'] and 15. - 2 / 30 - .002 <= w['t1_s'] - w['t0_s'] <= 15.002
    # a window that does not fit in a quarter of the part limit is not used, and none is asked for at 0
    assert pieces.overlap_for(40.) == 0. and pieces.overlap_for(450., 0) == 0.
    assert pieces.part_spans(t, cuts, len(t))[1] == [None, None]


def _windowed_parts(tmp_path, monkeypatch, overlap_s=0.4):
    """A 3s recording cut into two parts that share overlap_s seconds around their cut, written where a job keeps
    them."""
    monkeypatch.setattr(pieces, 'piece_max', lambda ctx: 2.0)
    recorder_folder(tmp_path / 'upload', n=90, duplicate_stamps=False)
    rep = f.convert(tmp_path / 'upload', 'teleop_arms', tmp_path / 'eps', 'test', 900)
    src = tmp_path / 'eps' / rep['episodes'][0]['episode_id']
    parts = pieces.write_pieces(src, tmp_path / 'job' / 'pieces', overlap_s=overlap_s)
    (tmp_path / 'job' / 'run' / 'out').mkdir(parents=True)
    return src, parts, [json.loads((p / 'context.json').read_text()) for p in parts]


def test_parts_cut_with_a_shared_window_record_it_and_both_hold_its_frames(tmp_path, monkeypatch):
    src, parts, contexts = _windowed_parts(tmp_path, monkeypatch)
    whole = me.load(src)
    n = int(whole['sources']['exo']['n_frames'])
    assert len(parts) == 2
    (window,) = contexts[0]['piece']['windows']
    assert contexts[1]['piece']['windows'] == [window]
    assert window['t1_s'] == contexts[0]['piece']['t1_s'] and window['t0_s'] == contexts[1]['piece']['t0_s']
    assert 0.4 - 2 / 30 - .002 <= window['t1_s'] - window['t0_s'] <= 0.4 + .002      # whole frames, times in ms
    for p, ctx in zip(parts, contexts):
        assert ctx['duration_s'] <= 2.0 and ctx['piece']['t1_s'] - ctx['piece']['t0_s'] <= 2.0
        assert not pieces.needs_pieces(p)
        assert 'Neighbouring parts share a few seconds around each cut' in ctx['collection_note']
    clock = [me.frame_time(whole, k) for k in range(n)]
    shared = sum(window['t0_s'] - 1e-6 <= x < window['t1_s'] - 1e-6 for x in clock)
    assert sum(int(me.load(p)['sources']['exo']['n_frames']) for p in parts) == n + shared and shared > 0
    stored = json.loads((src / 'context.json').read_text())['pieces']
    assert stored['overlap_s'] == 0.4 and stored['cuts'][0]['window'] == window
    # the default 15s window does not fit a 2s part limit, so the parts touch at the cut as before, and the recording
    # says why
    plain = pieces.write_pieces(src, tmp_path / 'plain')
    assert all('windows' not in json.loads((p / 'context.json').read_text())['piece'] for p in plain)
    assert sum(int(me.load(p)['sources']['exo']['n_frames']) for p in plain) == n
    assert 'does not fit' in json.loads((src / 'context.json').read_text())['pieces']['overlap_not_used']


def test_tasks_in_a_shared_window_join_one_to_one_by_the_time_they_share():
    window = {'cut_s': 300., 't0_s': 292.5, 't1_s': 307.5}
    one, two = {'index': 1, 't0_s': 0., 't1_s': 307.5}, {'index': 2, 't0_s': 292.5, 't1_s': 600.}
    task = lambda name, lo, hi, part, **x: {'start_s': lo, 'end_s': hi, 'task': name, 'source_piece': dict(part),
                                            'objects': [], 'note': '', **x}
    tower = task('Build a freestanding tower from the loose blocks', 200., 307.5, one, outcome='unclear',
                 success_predicate='All blocks stand in one tower.', objects=['wooden blocks'])
    finish = task('Finish stacking the remaining cubes', 292.5, 400., two, outcome='success', completed_at_s=399.,
                  success_predicate='The last cube is on top.', objects=['cubes'], note='Already under way.')
    lid = task('Set the lid down', 280., 295., one, outcome='success', completed_at_s=294.)
    lid_again = task('Put the lid on the table', 292.5, 296., two, outcome='success', completed_at_s=295.8)
    new = task('Wipe the table', 301., 306., two, outcome='success', completed_at_s=305.)
    tasks = pieces.join_across_cuts([tower, finish, lid, lid_again, new], [300.], windows={300.: window})
    by_name = {t['task']: t for t in tasks}
    assert len(tasks) == 3 and set(by_name) == {tower['task'], lid['task'], new['task']}
    # the tower and its finish share the whole window: one task, judged on the left half's goal with the right
    # half's outcome, whatever each part called it or its objects
    joined = by_name[tower['task']]
    assert [h['task'] for h in joined['joined_from']] == [tower['task'], finish['task']]
    assert (joined['outcome'], joined['completed_at_s'], joined['end_s']) == ('success', 399., 400.)
    assert joined['success_predicate'] == tower['success_predicate']
    assert joined['cut_joins'] == [{'cut_s': 300., 'by': 'shared_window', 'window_s': [292.5, 307.5],
                                    'shared_s': 15., 'shorter_s': 15.}]
    # the lid's two copies share all of the shorter one's 2.5s in the window, after the tower took its partner first
    assert [h['task'] for h in by_name[lid['task']]['joined_from']] == [lid['task'], lid_again['task']]
    assert by_name[new['task']] == new          # no partner: kept as it came
    # two tasks of each part over the whole window cannot be told apart by time: none of them joins
    twin = [task('Hold the left cup', 250., 307.5, one), task('Hold the right cup', 250., 307.5, one),
            task('Hold a cup', 292.5, 350., two), task('Hold the other cup', 292.5, 350., two)]
    assert len(pieces.join_across_cuts(twin, [300.], windows={300.: window})) == 4
    # spans that share less than half of the shorter one's time inside the window stay apart
    brief = [task('Reach', 250., 294., one), task('Carry', 293.5, 330., two)]
    assert len(pieces.join_across_cuts(brief, [300.], windows={300.: window})) == 2
    # a seam verdict is not read at a cut with a window
    assert len(pieces.join_across_cuts(brief, [300.], {300.: {'same_task': True}}, {300.: window})) == 2


def test_steps_and_key_events_of_a_shared_window_are_kept_from_one_part_only(tmp_path, monkeypatch):
    src, parts, contexts = _windowed_parts(tmp_path, monkeypatch)
    (window,) = contexts[0]['piece']['windows']
    w0, w1 = window['t0_s'], window['t1_s']
    at = lambda share: round(w0 + share * (w1 - w0), 3)
    origin = [ctx['piece']['t0_s'] for ctx in contexts]
    local = lambda i, x: round(x - origin[i], 3)
    records = []
    for i, (p, ctx) in enumerate(zip(parts, contexts)):
        record = _part_result(p, 'stack')
        span = round(ctx['piece']['t1_s'] - ctx['piece']['t0_s'], 3)
        times = [0.3, at(.25), at(.75)] if i == 0 else [at(.25), at(.75), 2.8]
        record['labels'].update(
            timeline=[{'start_s': local(i, x), 'end_s': local(i, x) + .05, 'action': f'part {i + 1} at {x}'}
                      for x in times],
            key_events=[{'t_s': local(i, x), 'label': f'part {i + 1} event at {x}', 'outcome': 'success'}
                        for x in times],
            tasks=[{'start_s': 0.1 if i == 0 else 0.0, 'end_s': span if i == 0 else local(i, 2.5),
                    'task': 'Build the tower' if i == 0 else 'Finish stacking the cubes', 'objects': [],
                    'outcome': 'unclear' if i == 0 else 'success', 'completed_at_s': None if i == 0 else local(i, 2.4),
                    'success_predicate': 'four blocks stacked' if i == 0 else 'last cube on top', 'note': ''}],
            data_issues=[{'category': 'starts_mid_task', 'issue': 'starts mid task', 't_s': 0.0, 'severity': 'low'}]
            if i == 1 else [])
        records.append(record)
    stitched = pieces.stitch(src, list(zip(contexts, records)))
    labels = stitched['labels']
    assert [s['action'] for s in labels['timeline']] == ['part 1 at 0.3', f'part 1 at {at(.25)}',
                                                         f'part 2 at {at(.75)}', 'part 2 at 2.8']
    assert [e['label'] for e in labels['key_events']] == ['part 1 event at 0.3', f'part 1 event at {at(.25)}',
                                                          f'part 2 event at {at(.75)}', 'part 2 event at 2.8']
    copies = [x for x in labels['_excluded'] if x['excluded_by'] == 'shared_window']
    assert sorted((x['list'], x['source_piece']['index']) for x in copies) == [
        ('key_events', 1), ('key_events', 2), ('timeline', 1), ('timeline', 2)]
    # the task across the window is one task, and nothing is left open at the cut
    (task,) = labels['tasks']
    assert task['task'] == 'Build the tower' and task['outcome'] == 'success' and len(task['joined_from']) == 2
    assert task['success_predicate'] == 'four blocks stacked'
    # the right part's "starts mid task" is about where its footage starts, the window's start, so it is our cut's
    assert [x['excluded_by'] for x in labels['_excluded'] if x.get('category') == 'starts_mid_task'] == ['piece_cut']
    assert stitched['stitched']['cuts_s'] == [window['cut_s']] and stitched['stitched']['windows'] == [window]


def test_a_job_with_seam_verdicts_joins_the_cut_they_say_is_one_task(tmp_path, monkeypatch):
    """Saved parts without a shared window are stitched as on main, except that a task across the cut is joined only
    by the job's seam verdicts (JOB/seams.json, which the seam check writes)."""
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    contexts = [json.loads((part / 'context.json').read_text()) for part in parts]
    tasks = [[_cup_task(contexts[0], end=True)], [_cup_task(contexts[1], start=True)], []]
    for p, c, t in zip(parts, contexts, tasks):
        (job / 'run' / 'out' / f'{p.name}.json').write_text(json.dumps(_tasks_part(p, c, t)))
    long_eps = {src.name: [p.name for p in parts]}
    pieces.stitch_run(job, src.parent, long_eps, tmp_path / 'apart')
    apart = json.loads((tmp_path / 'apart' / f'{src.name}.json').read_text())
    assert [t['outcome'] for t in apart['labels']['tasks']] == ['continues_past_cut', 'success']
    cut = contexts[1]['piece']['t0_s']
    verdict = {'same_task': True, 'evidence': 'The same cups are stacked on both sides of the cut.'}
    (job / 'seams.json').write_text(json.dumps({'seams': {src.name: {str(cut): verdict}}}))
    pieces.stitch_run(job, src.parent, long_eps, tmp_path / 'joined')
    joined = json.loads((tmp_path / 'joined' / f'{src.name}.json').read_text())
    (task,) = joined['labels']['tasks']
    assert task['outcome'] == 'success' and len(task['joined_from']) == 2
    assert task['cut_joins'] == [{'cut_s': cut, 'by': 'seam_check', **verdict}]
    assert joined['stitched']['seams'] == {str(cut): verdict}


# ---- the recording's goal alignment from its parts ---------------------------------------------------------------

PARTED = {r['episode']: r for r in json.loads(
    (Path(__file__).parent / 'fixtures' / 'parted_goal_alignments.json').read_text())['records']}


def _real_alignments(name: str) -> list[dict]:
    return [a['goal_alignment'] for a in PARTED[name]['part_assessments']]


def test_a_recording_labelled_in_parts_reads_its_goal_alignment_from_what_each_part_saw():
    """The 8 real C' recordings labelled in parts. Polymailer 153320 shows mailers filed into a rigid bin, and both
    parts say different: the recording is different, with each part's note. The dishes recording 163942 is loaded in
    part 1 (broader: its window holds only the loading) and returned to the table in part 2 (aligned): together the
    parts show the asked cycle, so the recording is aligned, not "only part of the given goal". The other six say
    aligned in every part."""
    got = {name: pieces.parts_goal_alignment(r['part_assessments'], len(r['part_assessments']))
           for name, r in PARTED.items()}
    poly = got['episode_20261013_episode_153320_8aefd0dd']
    assert (poly['relation'], poly['matches_given'], poly['basis']) == ('different', False, 'parts')
    assert poly['note'].startswith('Part 1: The visible work collects mailer envelopes into a rigid plastic bin')
    assert 'Part 2: The visible receiving container is a rigid plastic bin' in poly['note']
    assert [p['relation'] for p in poly['parts']] == ['different', 'different']
    dishes = got['episode_20261001_episode_163942_b4c0d44b']
    assert (dishes['relation'], dishes['matches_given']) == ('aligned', True)
    assert [p['relation'] for p in dishes['parts']] == ['broader', 'aligned']
    assert sorted(g['relation'] for g in got.values()) == ['aligned'] * 7 + ['different']


def test_parts_establish_a_goal_alignment_only_when_each_part_compared_or_one_saw_other_work():
    poly = PARTED['episode_20261013_episode_153320_8aefd0dd']['part_assessments']
    cables = PARTED['episode_20260914_episode_162358_1b6d49f5']['part_assessments']
    # a part with no labels may hold other work: the aligned parts alone establish nothing
    assert pieces.parts_goal_alignment(cables[:2], 3) is None
    assert pieces.parts_goal_alignment([*cables[:2], {**cables[2], 'goal_alignment': None}], 3) is None
    # one part showing other work is enough, whatever the others say or whether they gave labels
    one_off = pieces.parts_goal_alignment([poly[0], cables[1]], 3)
    assert (one_off['relation'], one_off['matches_given']) == ('different', False)
    # unrelated (likely mislabelled) outranks different; a part that says different yet matches_given true is no
    # mismatch, and no reading at all
    mixed = [poly[0], {**poly[1], 'goal_alignment': {**poly[1]['goal_alignment'], 'relation': 'unrelated'}}]
    assert pieces.parts_goal_alignment(mixed, 2)['relation'] == 'unrelated'
    unsure = [{**a, 'goal_alignment': {**a['goal_alignment'], 'matches_given': True}} for a in poly]
    assert pieces.parts_goal_alignment(unsure, 2) is None
    # every part saw only a share of the asked work: broader; one saw it and more: narrower
    share = [{**a, 'goal_alignment': {**a['goal_alignment'], 'relation': 'broader'}} for a in cables]
    assert pieces.parts_goal_alignment(share, 3)['relation'] == 'broader'
    more = [*share[:2], {**cables[2], 'goal_alignment': {**cables[2]['goal_alignment'], 'relation': 'narrower'}}]
    assert pieces.parts_goal_alignment(more, 3)['relation'] == 'narrower'


def test_a_stitched_recording_carries_its_parts_goal_alignment_to_every_reader(tmp_path, monkeypatch):
    """A recording labelled in two parts, each part's reply carrying the real polymailer part's goal_alignment: the
    stitched record has a recording-level goal_alignment (main wrote none), so the consistency rules, the gate's align
    fact and the scores' mislabel test read it."""
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.6)
    recorder_folder(tmp_path / "upload", n=90, duplicate_stamps=False)
    rep = f.convert(tmp_path / "upload", "teleop_arms", tmp_path / "eps", "test", 900)
    src = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    parts = pieces.write_pieces(src, tmp_path / "job" / "pieces")
    assert len(parts) == 2
    contexts = [json.loads((p / 'context.json').read_text()) for p in parts]
    poly = _real_alignments('episode_20261013_episode_153320_8aefd0dd')
    records = []
    for p, alignment in zip(parts, poly):
        record = _part_result(p, 'collect the mailers into the bin')
        record['labels']['goal_alignment'] = alignment
        records.append(record)
    labels = pieces.stitch(src, list(zip(contexts, records)))['labels']
    alignment = labels['goal_alignment']
    assert (alignment['relation'], alignment['matches_given'], alignment['basis']) == ('different', False, 'parts')
    assert [a['goal_alignment'] for a in labels['part_assessments']] == poly
    from board.label_projection import project
    assert project(labels)[0]['goal_alignment'] == alignment
    from gate.score import judge
    assert judge({'align': ['different', 'unrelated']}, labels)[0]
    from board.scores import task_source
    assert task_source({'goal_alignment': alignment, '_meta': {'given_prompt': 'Place items into polymailer bag'},
                        'episode_prompt': 'collect the mailers into the bin'})[1] is False
    from checks.label_consistency import check
    reached = {**labels, 'completion': {'task_completed': 'success'}}
    assert 'outcome_vs_alignment' in {x['rule'] for x in check(reached)}
