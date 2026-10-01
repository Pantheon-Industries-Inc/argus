"""A camera smaller than the grid cell is sent at its own size, never enlarged, and the prompt says so; cameras at least
as wide as the cell are cut to it exactly as before."""
import io

from PIL import Image

from label import episode
from label import frames as mf


def _im(w, h, c=(200, 30, 30)):
    return Image.new("RGB", (w, h), c)


def test_a_small_frame_is_never_enlarged():
    assert Image.open(io.BytesIO(mf.to_jpeg(_im(160, 120), 448))).size == (160, 120)
    assert Image.open(io.BytesIO(mf.to_jpeg(_im(640, 480), 448))).size == (448, 336)
    assert Image.open(io.BytesIO(mf.to_jpeg(_im(448, 252), 448))).size == (448, 252)


def test_a_small_camera_sits_centred_in_a_column_as_wide_as_the_widest():
    big, small = mf.to_jpeg(_im(640, 480), 448), mf.to_jpeg(_im(160, 120, (30, 200, 30)), 448)
    g = Image.open(io.BytesIO(mf.compose_grid([(0.0, {"exo": big, "left": small}), (1.0, {"exo": big, "left": small})],
                                             ["exo", "left"], 90, gutter=84, header=36)))
    assert g.width == 84 + 2 * (448 + 4)
    # the small camera's cell: its green fills only the middle 160 px of the 448 px column
    row = [g.getpixel((x, 36 + 336 + 4 + 60)) for x in range(84, 84 + 448)]
    green = [i for i, p in enumerate(row) if p[1] > 150 and p[0] < 100]
    assert green and abs(green[0] - (448 - 160) // 2) <= 2 and abs(green[-1] - ((448 - 160) // 2 + 159)) <= 2


def test_the_prompt_states_each_cameras_real_cell_size():
    def ep(cams):
        return {"context": {"profile": "teleop_arms", "cameras": cams}}
    same = ep({"exo": {"width": 640, "height": 480, "name": "exo"}, "left": {"width": 640, "height": 480, "name": "left"}})
    assert episode._cell_sizes(same, 448, 336) == "448x336"
    mixed = ep({"exo": {"width": 640, "height": 480, "name": "exo"}, "left": {"width": 1280, "height": 720, "name": "left"}})
    assert episode._cell_sizes(mixed, 448, 336) == "448 px wide (exo 448x336, left 448x252)"
    tiny = ep({"exo": {"width": 640, "height": 480, "name": "exo"}, "left": {"width": 160, "height": 120, "name": "left"}})
    assert episode._cell_sizes(tiny, 448, 336) == ("at most 448 px wide (exo 448x336, left 160x120 at its own size, "
                                                   "not enlarged)")
    one = ep({"exo": {"width": 160, "height": 120, "name": "exo"}})
    assert episode._cell_sizes(one, 448, 336) == "160x120 at its own size, not enlarged"
