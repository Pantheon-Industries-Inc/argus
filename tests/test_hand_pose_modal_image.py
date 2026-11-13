"""The hand-pose image imports its packaged modules without the rest of the repository."""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def image_files(tmp_path):
    here = Path(__file__).resolve().parents[1] / 'board' / 'hand_pose'
    tree = ast.parse((here / 'modal_app.py').read_text())
    image = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'image' for target in node.targets))

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def apt_install(self, *args, **kwargs):
            return self

        pip_install = run_commands = env = apt_install

        def add_local_file(self, src, dst, **kwargs):
            target = tmp_path / dst.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            return self

    # Evaluate only the image declaration. No Modal client, dependency installation or GPU is needed.
    scope = {'modal': SimpleNamespace(Image=Image), 'HERE': here, 'PREFIX': 'test', 'OUT': 'test-out',
             'SPECS': None, 'ACE_COMMIT': 'not-fetched', 'VIDEOX_COMMIT': 'not-fetched'}
    eval(compile(ast.Expression(image), str(here / 'modal_app.py'), 'eval'), scope)
    return tmp_path / 'opt' / 'ace'


def _isolated(image_files, body):
    # -I excludes the checkout and PYTHONPATH, as in the image that carries only declared files.
    code = f'import sys; sys.path.insert(0, {str(image_files)!r}); ' + body
    return subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)


def test_packaged_display_imports_without_the_checkout(image_files):
    result = _isolated(image_files, 'import display')
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not (shutil.which('ffmpeg') and shutil.which('ffprobe')), reason='no ffmpeg')
def test_packaged_display_probes_a_real_video_without_the_checkout(image_files, tmp_path):
    video = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i', 'color=black:s=32x24:r=30',
                    '-frames:v', '2', '-c:v', 'libx264', str(video)], check=True)
    result = _isolated(image_files, f'import display,json; g=display.geometry({str(video)!r}); '
                       'print(json.dumps({"stored": g["stored"], "rotation": g["rotation"], '
                       '"mirror": g["mirror"], "sar": str(g["sar"])}))')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'stored': [32, 24], 'rotation': 0.0, 'mirror': False, 'sar': '1'}
