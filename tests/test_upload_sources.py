"""A member whose owner an archive does not prove is an upload of its own in every way (prepare/formats.py
_unpack_archives, _convert_upload): its storage belongs to the conversion that read it, an archive it holds is opened
on the same budget, and nothing of another upload (a table, a task, folder metadata) is joined to its recordings.
A private name (label/atomic.py private_name) is valid whenever its source is, and a shared name (an output, its
receipts, a proof) keeps its one exact format, so every reader finds what the writer wrote, at any length the base
format holds. Every count and list of the pictures a video shows reads one definition (label/presented.py)."""
from __future__ import annotations

import io
import json
import shutil
import struct
import subprocess
import zipfile

import av
import numpy as np
import pytest

from label import atomic, episode
from label.presented import presented, presented_pts
from prepare import formats


def movie(level, frames=8):
    out = io.BytesIO()
    with av.open(out, 'w', format='mp4') as container:
        stream = container.add_stream('mpeg4', rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for _ in range(frames):
            picture = av.VideoFrame.from_ndarray(np.full((48, 64, 3), level, np.uint8), format='rgb24')
            for packet in stream.encode(picture):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out.getvalue()


def zipped(members: dict) -> bytes:
    sink = io.BytesIO()
    with zipfile.ZipFile(sink, 'w', zipfile.ZIP_STORED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return sink.getvalue()


def rename_first_local(raw: bytes, old: str, new: str) -> bytes:
    """The first member's own header names it `new`; its directory entry still says `old`."""
    raw = bytearray(raw)
    length = struct.unpack_from('<H', raw, 26)[0]
    assert raw[30:30 + length] == old.encode() and len(new) == len(old)
    raw[30:30 + length] = new.encode()
    return bytes(raw)


def contexts(out, report):
    return {e['episode_id']: json.loads((out / e['episode_id'] / 'context.json').read_text())
            for e in report['episodes']}


def mean_level(out, episode_id):
    loaded = episode.load(out / episode_id)
    view = next(iter(loaded['sources']))
    return round(float(np.asarray(episode._decode_view(loaded, view, [0])[0]).mean()))


@pytest.mark.parametrize('upload', ['folder', 'archive'])
def test_a_later_conversion_never_changes_the_footage_of_an_earlier_one(tmp_path, upload):
    """Two uploads converted beside each other: the second never writes where the first one's episodes read."""
    levels = {}
    for name, level in (('a', 40), ('b', 180)):
        raw = rename_first_local(zipped({'takeA/first.mp4': movie(level), 'healthy/front.mp4': movie(level)}),
                                 'takeA/first.mp4', 'takeB/first.mp4')
        source = tmp_path / f'source_{name}'
        source.mkdir()
        (source / 'recordings.zip').write_bytes(raw)
        root = source if upload == 'folder' else source / 'recordings.zip'
        out = tmp_path / f'out_{name}'
        report = formats.convert(root, 'ego_head', out, name, 900)
        assert len(report['episodes']) == 2 and not report['failed'], report
        levels[name] = (out, report)
    out_a, report_a = levels['a']
    for e in report_a['episodes']:
        assert abs(mean_level(out_a, e['episode_id']) - 40) < 8, e
    sources = [json.loads((out_a / e['episode_id'] / 'sources.json').read_text()) for e in report_a['episodes']]
    files_a = {v['packed'] for s in sources for v in s.values() if isinstance(v, dict) and v.get('packed')}
    out_b, report_b = levels['b']
    sources = [json.loads((out_b / e['episode_id'] / 'sources.json').read_text()) for e in report_b['episodes']]
    files_b = {v['packed'] for s in sources for v in s.values() if isinstance(v, dict) and v.get('packed')}
    assert files_a and files_b and not files_a & files_b


def test_a_member_kept_apart_that_is_an_archive_is_opened(tmp_path):
    inner = zipped({'take/front.mp4': movie(60)})
    raw = rename_first_local(zipped({'takeA/first.zip': inner, 'healthy/front.mp4': movie(200)}),
                             'takeA/first.zip', 'takeB/first.zip')
    path = tmp_path / 'up.zip'
    path.write_bytes(raw)
    out = tmp_path / 'out'
    report = formats.convert(path, 'ego_head', out, 'nested apart', 900)
    assert len(report['episodes']) == 2, report
    levels = sorted(mean_level(out, e['episode_id']) for e in report['episodes'])
    assert abs(levels[0] - 60) < 8 and abs(levels[1] - 200) < 8, levels
    apart = [e for e in report['episodes'] if 'name_unverified' in e['episode_id']]
    assert len(apart) == 1


def test_no_table_of_the_upload_describes_a_member_kept_apart(tmp_path):
    """The upload's own table names member.h5; the member kept apart is read as member.h5 too, but the table is not its
    upload's, so it gets no row, no task and no table in its source."""
    h5py = pytest.importorskip('h5py')
    files = {}
    for name, level in (('source.h5', 30), ('member.h5', 200)):
        with h5py.File(tmp_path / name, 'w') as h:
            h.attrs['fps'] = 30
            for group in ('demo_00', 'demo_01'):
                h[group + '/observations/images/front'] = np.full((8, 48, 64, 3), level, dtype=np.uint8)
        files[name] = (tmp_path / name).read_bytes()
    raw = bytearray(zipped({**files, 'annotations.csv': b'file,group,task\nmember.h5,demo_00,TASK OF THE INTACT FILE\n'}))
    central = raw.find(b'PK\x01\x02')
    raw[central + 46:central + 46 + 9] = b'member.h5'            # source.h5's directory entry now says member.h5
    path = tmp_path / 'up.zip'
    path.write_bytes(bytes(raw))
    out = tmp_path / 'out'
    report = formats.convert(path, 'ego_head', out, 'table ownership', 900)
    ctx = contexts(out, report)
    apart = {eid: c for eid, c in ctx.items() if 'name_unverified' in eid}
    intact = {eid: c for eid, c in ctx.items() if 'name_unverified' not in eid}
    assert apart and intact, report
    tasked = [eid for eid, c in ctx.items() if 'TASK OF THE INTACT FILE' in json.dumps(c)]
    assert tasked and all(eid in intact for eid in tasked), tasked
    for eid, c in apart.items():
        assert not c.get('uploader_notes') and not (c.get('source') or {}).get('tables'), c
        assert not (out / eid / 'instruction.txt').exists() or 'INTACT' not in (out / eid / 'instruction.txt').read_text()


def test_a_cut_with_preroll_has_no_damaged_frames_and_every_count_agrees(tmp_path):
    """A stream copy cut away from a key frame keeps the frames before its start as discarded preroll, which the decoder
    needs and never shows: no picture of it is damaged, and the probe, the frame count and the decoder agree."""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('ffmpeg is needed to cut a stream copy')
    clean = tmp_path / 'clean.mp4'
    with av.open(str(clean), 'w') as out:
        s = out.add_stream('libx264', rate=30)
        s.width, s.height, s.pix_fmt = 64, 48, 'yuv420p'
        s.options = {'g': '30', 'keyint_min': '30', 'sc_threshold': '0'}
        for i in range(90):
            for packet in s.encode(av.VideoFrame.from_ndarray(np.full((48, 64, 3), (i * 3) % 256, np.uint8),
                                                             format='rgb24')):
                out.mux(packet)
        for packet in s.encode():
            out.mux(packet)
    cut = tmp_path / 'cut.mp4'
    subprocess.run([ffmpeg, '-y', '-v', 'error', '-ss', '0.5', '-i', str(clean), '-t', '1', '-c:v', 'copy', str(cut)],
                   check=True)
    with av.open(str(cut)) as c:
        discarded = sum(1 for p in c.demux(video=0) if p.size and p.is_discard)
    with av.open(str(cut)) as c:
        decoded = sorted(int(f.pts) for f in c.decode(video=0))
    with av.open(str(cut)) as c:
        shown = presented_pts(c, c.streams.video[0])
    with av.open(str(cut)) as c:
        counted = sum(1 for _ in presented(c, c.streams.video[0]))
    assert discarded, 'the cut should carry preroll for this test to mean anything'
    assert shown == decoded and counted == len(decoded)
    assert formats.decoded_damage(cut) == set()
    out = tmp_path / 'out'
    report = formats.convert(cut, 'ego_head', out, 'preroll', 900)
    ctx = contexts(out, report)
    assert not any(i['kind'] == 'frames_damaged' for c in ctx.values() for i in c.get('reader_issues') or [])


@pytest.mark.parametrize('length', [1, 150, 191, 192, 220, 255])
def test_every_private_name_is_valid_and_its_own(tmp_path, length):
    name = 'x' * (length - 5) + '.json' if length > 5 else 'x' * length
    other = 'y' + name[1:]
    for prefix, suffix in (('.', f'.{"0" * 32}.tmp.mp4'), ('.', f'.{"0" * 7}.{"0" * 32}.removing'), ('.', '.unpacked'),
                           ('000-', '')):
        private = atomic.private_name(name, prefix, suffix)
        assert len(private.encode()) <= atomic.NAME_MAX, (prefix, suffix, len(private))
        assert private != atomic.private_name(other, prefix, suffix)
        assert private.startswith(prefix) and private.endswith(suffix)
        if len(name.encode()) <= atomic.NAME_MAX - atomic.PRIVATE_RESERVE:
            assert private == prefix + name + suffix          # a name that leaves room is kept as it is
    target = tmp_path / name
    atomic.write_atomic(target, {'ok': True})                  # its temporary file fits too
    assert json.loads(target.read_text()) == {'ok': True}
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]


@pytest.mark.parametrize('name', ['episode_short', 'episode_' + 'a' * 187])
def test_a_dispatch_left_unresolved_blocks_a_plain_resume_at_any_name_length(resends_used_up, tmp_path, monkeypatch, name):
    """Review of 2514ce5: for a 195-byte output name the claim of a dispatch whose reply was lost was written under a
    shortened name its readers never looked for, so a plain resume paid again and its reservation left the ledger. The
    claim's name is the base format exactly, so the resume finds it, sends nothing and the ledger keeps it."""
    from label import harness
    from label.dictionary_stage import output_spend, output_spend_complete
    calls = []

    def provider(*args, **kwargs):
        calls.append('dispatched')
        raise TimeoutError('the reply was lost after dispatch')
    identity = {'version': 2, 'source_proof': [], 'settings': {'cell_w': 0, 'max_tokens': 100,
                                                               'annotation_policy': {'task_mode': 'observe'}}}
    monkeypatch.setattr(harness, 'call_model_once', provider)
    monkeypatch.setattr(harness, '_release_memory', lambda: None)
    monkeypatch.setattr(harness, 'input_identity', lambda *a, **kw: identity)

    def label(ep, out, **kw):
        return harness._call_and_record(ep, out, [{'type': 'text', 'text': 'fixture'}], 0, {'input_identity': identity},
                                        model=harness.DEFAULT_MODEL, reasoning='low', api_key='fixture', max_tokens=100,
                                        timeout=1, reserve_final=kw['reserve_final'],
                                        require_known_pricing=kw['require_known_pricing'])
    monkeypatch.setattr(harness, 'label_episode', label)
    ep = tmp_path / 'eps' / name
    ep.mkdir(parents=True)
    out = tmp_path / 'run' / 'out'
    for run in (1, 2):
        harness._run_batch([ep], out, keys=['fixture'], concurrency=1, force=False, max_spend=1,
                           model=harness.DEFAULT_MODEL, reasoning='low', max_tokens=100)
        assert len(calls) == 1, (run, calls)                   # the resume sends nothing
        claim = out / f'noreply_{name}.json'
        assert claim.exists(), sorted(p.name for p in out.iterdir())
        assert output_spend(out) > 0 and not output_spend_complete(out)
        assert harness.billing_record(out, f'{name}.json') is not None


def test_pieces_of_a_long_recording_stitch_from_the_proof_their_cutter_wrote(tmp_path, monkeypatch):
    """Review of 2514ce5: the cutter wrote a 192-byte recording's source proof under a shortened name and the stitcher
    read the exact one, so paid part replies could not be put together. Both use the one exact name."""
    from label import pieces
    from label.evidence_access import source_proof
    from test_formats import recorder_folder
    from test_pieces import _part_result
    upload = tmp_path / 'upload'
    recorder_folder(upload, n=90, duplicate_stamps=False)
    report = formats.convert(upload, 'teleop_arms', tmp_path / 'eps', 'test', 900)
    source = tmp_path / 'eps' / report['episodes'][0]['episode_id']
    long_source = source.with_name('episode_' + 'a' * 184)
    source.rename(long_source)
    monkeypatch.setattr(pieces, 'piece_max', lambda ctx: 1.0)
    job = tmp_path / 'job'
    parts = pieces._write_or_reuse(long_source, job / 'pieces')
    assert len(parts) > 1
    out = job / 'run' / 'out'
    out.mkdir(parents=True)
    for part in parts:
        record = _part_result(part, 'reach')
        ctx = json.loads((part / 'context.json').read_text())
        record['input_identity'] = {'version': 2, 'source_proof': source_proof(ctx, part)}
        (out / (part.name + '.json')).write_text(json.dumps(record))
    stitched = pieces.stitch_run(job, long_source.parent, {long_source.name: [p.name for p in parts]}, tmp_path / 'final')
    assert stitched, stitched


def test_an_archive_kept_apart_keeps_the_metadata_it_holds(tmp_path):
    """Review of 2514ce5: a recovered archive was read from a folder that also held its own bytes, so the dataset root
    inside it was not found and its info.json (a session task mode, a collection description) was lost. It is now held
    outside the folder it unpacks to, exactly as a direct archive upload is, and reads as that upload would."""
    inner = zipped({'take/front.mp4': movie(30, frames=45),
                    'take/info.json': json.dumps({'task_mode': 'session',
                                                  'description': 'Continuous recording of repeated different tasks'})})
    (tmp_path / 'inner.zip').write_bytes(inner)
    raw = rename_first_local(zipped({'takeA/first.zip': inner, 'healthy/front.mp4': movie(200, frames=45)}),
                             'takeA/first.zip', 'takeB/first.zip')
    (tmp_path / 'outer.zip').write_bytes(raw)
    direct = formats.convert(tmp_path / 'inner.zip', 'ego_head', tmp_path / 'direct', 'direct', 900)
    nested = formats.convert(tmp_path / 'outer.zip', 'ego_head', tmp_path / 'nested', 'nested', 900)
    want = contexts(tmp_path / 'direct', direct)
    got = contexts(tmp_path / 'nested', nested)
    apart = [c for eid, c in got.items() if 'name_unverified' in eid]
    healthy = [c for eid, c in got.items() if 'name_unverified' not in eid]
    assert len(want) == 1 and len(apart) == 1 and len(healthy) == 1, (direct, nested)
    (want,) = want.values()
    assert want.get('task_mode') == 'session' and want.get('collection_note')
    assert apart[0].get('task_mode') == want['task_mode'] and apart[0].get('collection_note') == want['collection_note']
    assert healthy[0].get('task_mode') != 'session' and not healthy[0].get('collection_note')


def test_an_archive_the_filesystem_cannot_hold_costs_only_itself(tmp_path):
    """Review of 2514ce5: a chain of archives whose unpacked folders grew past the filesystem's path limit raised out of
    the conversion and cost the healthy recording beside it. Every filesystem step of an archive or a member is inside
    its own boundary, so the chain is left out where the filesystem refuses it and named, and the rest is read."""
    data, name = movie(100), 'front.mp4'
    for level in range(32):
        data = zipped({name: data})
        name = f'level{level:02d}_' + 'x' * 180 + '.zip'
    path = tmp_path / 'source.zip'
    path.write_bytes(zipped({'healthy/front.mp4': movie(100), name: data}))
    report = formats.convert(path, 'ego_head', tmp_path / 'out', 'deep', 900)
    assert any('healthy' in e['episode_id'] for e in report['episodes']), report
    said = ' '.join(str(n) for n in report['used'] + report['missing'])
    assert 'could not be' in said, said


def test_metadata_that_contradicts_itself_in_an_archive_kept_apart_refuses_nothing(tmp_path):
    """Review of 2ab6857: a recovered archive whose own metadata declared both a session and an episode task scope
    raised out of the conversion and cost the healthy recording beside it. A declared task scope is provenance only
    (every recording is labelled as a list of tasks), so the contradiction now costs nothing: both recordings are
    prepared, the member kept apart is still named as read as an upload of its own, and neither keeps a task mode."""
    inner = zipped({'take/front.mp4': movie(30, frames=45), 'take/info.json': json.dumps({'task_mode': 'session'}),
                    'take/dataset.json': json.dumps({'task_mode': 'episode'})})
    raw = rename_first_local(zipped({'takeA/first.zip': inner, 'healthy/front.mp4': movie(200, frames=45)}),
                             'takeA/first.zip', 'takeB/first.zip')
    path = tmp_path / 'outer.zip'
    path.write_bytes(raw)
    report = formats.convert(path, 'ego_head', tmp_path / 'out', 'scope conflict', 900)
    ids = sorted(e['episode_id'] for e in report['episodes'])
    assert len(ids) == 2 and 'episode_healthy_front' in ids and not report['failed'], report
    assert any('read as an upload of its own' in str(note) for note in report['missing']), report['missing']
    for name in ids:
        assert 'task_mode' not in json.loads((tmp_path / 'out' / name / 'context.json').read_text())


def test_a_member_kept_apart_that_cannot_be_written_never_takes_another_ones_folder(tmp_path):
    """Review of 2ab6857: a member kept apart whose file could not be written left its folder behind without counting
    it, so the next archive's member was given the same folder and lost. Every folder made is counted."""
    upload = tmp_path / 'upload'
    long_name = 'f.' + 'x' * 250
    for name, first, data in (('a', long_name, b'not a recording'), ('b', 'takeA/first.mp4', movie(200))):
        (upload / name).mkdir(parents=True)
        replaced = ('g' + first[1:]) if first == long_name else 'takeB/first.mp4'
        raw = rename_first_local(zipped({first: data, 'control/front.mp4': movie(80 if name == 'a' else 140)}),
                                 first, replaced)
        (upload / name / 'recordings.zip').write_bytes(raw)
    out = tmp_path / 'out'
    report = formats.convert(upload, 'ego_head', out, 'apart slots', 900)
    levels = sorted(mean_level(out, e['episode_id']) for e in report['episodes'])
    assert len(levels) == 3 and abs(levels[0] - 80) < 8 and abs(levels[1] - 140) < 8 and abs(levels[2] - 200) < 8, \
        (levels, report)
