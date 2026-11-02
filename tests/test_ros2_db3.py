"""A ROS 2 bag stored as SQLite (.db3) is read like the same bag stored as MCAP (prepare/ros2_bag.py)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from prepare import formats as f

ROWS = 10           # messages per split file, per topic
W, H = 96, 72


def _typestore():
    from rosbags.typesys import Stores, get_typestore
    return get_typestore(Stores.ROS2_HUMBLE)


def _image(ts, t_ns: int, encoding: str, data: np.ndarray):
    Header, Time = ts.types['std_msgs/msg/Header'], ts.types['builtin_interfaces/msg/Time']
    step = W * (3 if encoding == 'rgb8' else 2)
    return ts.types['sensor_msgs/msg/Image'](
        header=Header(stamp=Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9), frame_id='camera'),
        height=H, width=W, encoding=encoding, is_bigendian=0, step=step, data=data.reshape(-1).view(np.uint8))


def _imu(ts, t_ns: int, k: int):
    Header, Time = ts.types['std_msgs/msg/Header'], ts.types['builtin_interfaces/msg/Time']
    V, Q = ts.types['geometry_msgs/msg/Vector3'], ts.types['geometry_msgs/msg/Quaternion']
    return ts.types['sensor_msgs/msg/Imu'](
        header=Header(stamp=Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9), frame_id='imu'),
        orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0), orientation_covariance=np.zeros(9),
        angular_velocity=V(x=0.1 * k, y=0.0, z=0.0), angular_velocity_covariance=np.zeros(9),
        linear_acceleration=V(x=0.0, y=0.0, z=9.8 + 0.01 * k), linear_acceleration_covariance=np.zeros(9))


TOPICS = [(1, '/camera/color/image_raw', 'sensor_msgs/msg/Image'),
          (2, '/camera/depth/image_rect_raw', 'sensor_msgs/msg/Image'),
          (3, '/imu/data', 'sensor_msgs/msg/Imu')]


def _write_db3(path: Path, first: int) -> None:
    """One rosbag2 sqlite3 storage file (its topics and messages tables), ROWS frames of each topic at 10 fps."""
    ts = _typestore()
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE schema(schema_version INTEGER PRIMARY KEY, ros_distro TEXT NOT NULL);"
        "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, "
        "serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL);"
        "CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, timestamp INTEGER NOT NULL, "
        "data BLOB NOT NULL);"
        "CREATE INDEX timestamp_idx ON messages (timestamp ASC);")
    con.execute("INSERT INTO schema VALUES (3, 'humble')")
    con.executemany("INSERT INTO topics VALUES (?, ?, ?, 'cdr', '')", TOPICS)
    rng = np.random.default_rng(first)
    for k in range(first, first + ROWS):
        t = 1_700_000_000 * 10**9 + k * 100_000_000
        color = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        depth = np.full((H, W), 1000 + k, dtype=np.uint16)
        for tid, msg, typ in [(1, _image(ts, t, 'rgb8', color), 'sensor_msgs/msg/Image'),
                              (2, _image(ts, t, '16UC1', depth), 'sensor_msgs/msg/Image'),
                              (3, _imu(ts, t, k), 'sensor_msgs/msg/Imu')]:
            con.execute("INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)",
                        (tid, t, bytes(ts.serialize_cdr(msg, typ))))
    con.commit()
    con.close()


METADATA = """rosbag2_bagfile_information:
  version: 5
  storage_identifier: sqlite3
  relative_file_paths:
    - walk_0.db3
    - walk_1.db3
  duration:
    nanoseconds: 1900000000
  message_count: 60
  compression_format: ""
  compression_mode: ""
"""


@pytest.mark.parametrize('with_metadata', [True, False])
def test_split_db3_bag_is_one_episode_read_as_mcap(tmp_path, with_metadata):
    root = tmp_path / 'upload'
    root.mkdir()
    _write_db3(root / 'walk_0.db3', 0)
    _write_db3(root / 'walk_1.db3', ROWS)
    if with_metadata:
        (root / 'metadata.yaml').write_text(METADATA)
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'bag', 900)
    assert not report['failed'], report
    assert len(report['episodes']) == 1, report
    ep = tmp_path / 'episodes' / report['episodes'][0]['episode_id']
    ctx = json.loads((ep / 'context.json').read_text())
    sources = json.loads((ep / 'sources.json').read_text())
    # the colour camera is the episode's one camera, every frame of both split files; the depth topic is that
    # camera's depth stream, never a camera of its own
    keys = [c.get('key') for c in ctx['cameras'].values()]
    assert keys == ['/camera/color/image_raw'], ctx['cameras']
    assert [s['n_frames'] for s in sources.values()] == [2 * ROWS]
    assert [d['source'] for d in ctx['depth'].values()] == ['/camera/depth/image_rect_raw']
    # the IMU's numbers are signals, as on an MCAP
    names = {s['name'] for s in ctx['signals']}
    assert {'/imu/data angular_velocity', '/imu/data linear_acceleration'} <= names, names

def test_db3_in_subfolder_named_by_bag_and_metadata_consumed(tmp_path):
    root = tmp_path / 'upload'
    (root / 'take').mkdir(parents=True)
    _write_db3(root / 'take' / 'walk_0.db3', 0)
    (root / 'take' / 'metadata.yaml').write_text(METADATA.replace('    - walk_1.db3\n', ''))
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'bag', 900)
    assert len(report['episodes']) == 1, report
    assert 'walk' in report['episodes'][0]['episode_id']
    assert any('ROS 2 bag walk' in line for line in report['used']), report['used']
