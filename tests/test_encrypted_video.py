"""An encrypted video (MPEG common encryption) cannot be shown without its key, though its timestamps still read. Every
read of its pictures goes through label/presented.py presented(), which stops at the first encrypted packet with the
reason (EncryptedVideo), so an upload of one is refused at once with what to do, instead of being read through, cut
into a clip of decoder noise, and failing at the frame sampler as "no camera has a decodable frame" (a real robocap
head-camera upload, 160 MB, about 15 minutes of work)."""
from __future__ import annotations

import subprocess

import av
import pytest

from board import clips
from label.presented import EncryptedVideo, presented_pts
from prepare import formats


def _videos(tmp_path):
    ffmpeg = clips.find_ffmpeg()
    plain, enc = tmp_path / 'plain.mp4', tmp_path / 'take.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=size=64x48:rate=30', '-frames:v', '30',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(plain)], check=True)
    made = subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(plain), '-c', 'copy', '-encryption_scheme',
                           'cenc-aes-ctr', '-encryption_key', '76a6c65c5ea762046bd749a2e632ccbb', '-encryption_kid',
                           'a7e61c373e219033c21091fa607bf3b8', str(enc)], capture_output=True)
    if made.returncode:
        pytest.skip('this ffmpeg cannot write common encryption')
    return plain, enc


def test_an_encrypted_video_is_refused_at_its_first_packet(tmp_path):
    plain, enc = _videos(tmp_path)
    with av.open(str(plain)) as c:
        assert len(presented_pts(c, c.streams.video[0])) == 30
    with av.open(str(enc)) as c, pytest.raises(EncryptedVideo, match='take.mp4 is encrypted'):
        presented_pts(c, c.streams.video[0])


def test_an_upload_of_an_encrypted_video_says_so(tmp_path):
    _, enc = _videos(tmp_path)
    up = tmp_path / 'up'
    up.mkdir()
    enc.rename(up / 'take.mp4')
    try:
        report = formats.convert(up, 'ego_head', tmp_path / 'out', 'enc', 900)
        reasons = ' '.join(f['why'] for f in report['failed'])
    except ValueError as error:
        reasons = str(error)
    assert 'take.mp4 is encrypted' in reasons and 'without encryption' in reasons

