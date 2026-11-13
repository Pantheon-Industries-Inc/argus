"""Display transport precision is bounded and never changes canonical inputs."""
import copy
import gzip
import json
import math
import random
import subprocess
from pathlib import Path

import pytest


def test_compact_geometry_bounds_structure_source_and_browser_decode():
    from board.hand_geometry_transport import compact_geometry, XY_MAX_ERROR, Z_MAX_ERROR
    rng = random.Random(41)
    points = [[rng.uniform(-2000,2000),rng.uniform(-2000,2000),rng.uniform(.1,5)] for _ in range(1000)]
    points += [[1.2345,-2.3455,1e-10],[0.,-0.,.00000051],[1e15,-1e15,1e-15]]
    chunk = {'format':'board-hand-geometry-chunk/1','prediction_id':'same-prediction','start':0,
             'frames':[{'left':{'joints':points,'vertices':points,'presence':.1234567890123},'right':None},
                       {'left':None,'right':{'vertices':None,'joints':None}},{}]}
    original = copy.deepcopy(chunk)
    compact = compact_geometry(chunk)
    assert chunk == original and compact is not chunk
    assert compact['format']==chunk['format'] and compact['prediction_id']==chunk['prediction_id']
    assert compact['start']==0 and len(compact['frames'])==3
    assert compact['frames'][0]['right'] is None and compact['frames'][1:]==chunk['frames'][1:]
    assert compact['frames'][0]['left']['presence']==.1234567890123
    for field in ('joints','vertices'):
        for before,after in zip(points,compact['frames'][0]['left'][field]):
            assert len(after)==3 and after[2]>0
            assert abs(before[0]-after[0])<=XY_MAX_ERROR
            assert abs(before[1]-after[1])<=XY_MAX_ERROR
            assert abs(before[2]-after[2])<=Z_MAX_ERROR
    encode=lambda doc:json.dumps(doc,separators=(',',':')).encode()
    assert len(gzip.compress(encode(compact)))<.65*len(gzip.compress(encode(chunk)))
    header={'format':'board-hand-geometry/1','prediction_id':'same-prediction',
            'coordinates':{'xy':'source_pixels','z':'camera_meters'},
            'clip':{'w':100,'h':50,'times':[0,.0333333333333333,.0777777777777777]},
            'edges':[[0,1]],'topology':{'left':[[0,1,2]],'right':[]},
            'chunks':[{'start':0,'end':3,'file':'part.json'}]}
    assert compact_geometry(header)==header
    js=r"""
const assert=require('node:assert/strict'),{piece}=require('./tests/page_functions')();
eval(piece('hpGeometry')+piece('hpGeometryChunk'));
const header=JSON.parse(process.argv[1]),chunk=JSON.parse(process.argv[2]);
assert.deepEqual(hpGeometryChunk(hpGeometry(header),header.chunks[0],chunk),chunk);
"""
    subprocess.run(['node','-e',js,json.dumps(header),json.dumps(compact)],check=True)


def test_compact_geometry_rejects_invalid_input():
    from board.hand_geometry_transport import compact_geometry
    for point in ([1,2,0],[1,2,-1],[1,2,float('nan')],[True,2,1],[1,float('inf'),1],[1,2],None):
        chunk={'format':'board-hand-geometry-chunk/1','prediction_id':'p','start':0,
               'frames':[{'left':{'joints':[point]}}]}
        with pytest.raises(ValueError):compact_geometry(chunk)
    with pytest.raises(ValueError):compact_geometry({'format':'other'})


def test_static_compact_option_preserves_native_and_downloads(tmp_path,monkeypatch):
    import argparse
    from board import static,hands
    board=tmp_path/'board';qa=board/'qa';qa.mkdir(parents=True)
    label={'episode_prompt':'Place cup','dataset':'Example','_rig':'ego_head','camera_views':['exo'],
           '_meta':{'episode_id':'episode_000001'},'duration_s':1}
    (qa/'episode_000001.json').write_text(json.dumps(label))
    geometry=board/'hands';geometry.mkdir()
    header={'format':'board-hand-geometry/1','prediction_id':'p','coordinates':{'xy':'source_pixels','z':'camera_meters'},
            'clip':{'w':100,'h':50,'times':[0]},'edges':[[0,1]],'topology':{'left':[[0,1,2]],'right':[]},
            'chunks':[{'start':0,'end':1,'file':'part.json'}]}
    points=[[1.123456789,2.987654321,.123456789],[3.123456789,4.987654321,.234567891],[5.123456789,6.987654321,.345678912]]
    chunk={'format':'board-hand-geometry-chunk/1','prediction_id':'p','start':0,'frames':[{'left':{'joints':points,'vertices':points}}]}
    (geometry/'episode_000001.json').write_text(json.dumps(header));(geometry/'part.json').write_text(json.dumps(chunk))
    kp=board/'hand_keypoints';kp.mkdir();(kp/'index.json').write_text('{"files":{"episode_000001.json":{"frames":1}}}')
    download={'format':'hand-keypoints/1','points':points};(kp/'episode_000001.json').write_text(json.dumps(download))
    source_bytes={str(p):p.read_bytes()for p in [*geometry.iterdir(),*kp.iterdir()]}
    rec={'file':'episode_000001.json','episode_id':'episode_000001','duration_s':1}
    monkeypatch.setattr(static,'plan',lambda *args:[{'rec':rec,'ds':'Example','eid':'episode_000001','media':{'exo':{'rel':'v/e.mp4'}},'frames':{}}])
    monkeypatch.setattr(hands,'for_copy',lambda doc,clip:doc)
    outputs=[];download_bytes=[]
    for compact in (False,True):
        out=tmp_path/('compact'if compact else'exact');(out/'media/v').mkdir(parents=True);(out/'media/v/e.mp4').write_bytes(b'fixture')
        args=argparse.Namespace(board=board,qa=qa,clips=tmp_path/'clips',compare=None,hands=geometry,keypoints=kp,sensors=None,
                                out=out,build_id=None,force=False,public_base=None,title='Example',header=None,compact_hand_geometry=compact)
        assert static.cmd_site(args)==0
        build=out/(out/'LATEST').read_text().strip();outputs.append(build.name)
        assert json.loads((build/'data/hands/episode_000001.json').read_text())==header
        got=json.loads((build/'data/hands/part.json').read_text())
        if compact:assert got['frames'][0]['left']['joints'][0]==[1.123,2.988,.123457]
        else:assert got==chunk
        index=json.loads((build/'data/keypoints/index.json').read_text())
        downloaded=out/'media'/index['files']['episode_000001.json']['path']
        assert json.loads(downloaded.read_text())==download
        download_bytes.append(downloaded.read_bytes())
        assert dict(hands.geometry_assets(header,build/'data/hands'))['part.json']==got
    assert outputs[0]!=outputs[1]
    assert download_bytes[0]==download_bytes[1]
    assert all(Path(path).read_bytes()==body for path,body in source_bytes.items())
