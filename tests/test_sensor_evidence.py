"""Saved sensor observations stay distinct from model interpretation in the viewer."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_sensor_evidence_presentation():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([shutil.which('node'), str(root / 'tests/sensor_evidence.js'),
                             str(root / 'board/serve.py')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
