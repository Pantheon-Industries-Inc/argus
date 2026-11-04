"""The size of the default model's final reservation (label/harness.py final_cost_bound), pinned to its formula.

The spend cap holds this amount before each request is sent, so a bound that grows defers or refuses work a run could
afford, and every billing test that sets a roomy cap (PR 191) passes whatever its size: a bound inflated 1.3x or 1.6x
passed them all. Here the reservation of two fixed requests is worked out by hand from the published rates, and a
bound a micro-dollar off it fails, so 1.1x fails by seven cents or more.

Bound = (prompt bound x input rate + output limit x output rate), rounded up to the micro-dollar, where the prompt
bound is the text's bytes, four times each image's token estimate (68.8 + 1.115 per 32x32 px patch) and 1,024. The
standard rates of gpt-6-astra are $10 per million input tokens and $50 per million output tokens; from 272,000 prompt
tokens the long-context rates apply, $20 and $75.
"""
import base64
import io

import pytest
from PIL import Image

from label import harness


def png(w, h):
    buf = io.BytesIO()
    Image.new('RGB', (w, h), (90, 120, 150)).save(buf, format='PNG')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()}}


def test_the_default_model_reserves_its_formula_for_a_standard_prompt():
    # 30,000 text bytes and one 512x384 image (16 x 12 patches: 282.88 tokens, 1,131.52 bound): 32,155.52 prompt
    # tokens at $10/M and 8,000 output tokens at $50/M is $0.3215552 + $0.40
    content = [{'type': 'text', 'text': 'x' * 30_000}, png(512, 384)]
    assert harness.DEFAULT_MODEL == 'openai/gpt-6-astra'
    assert harness.final_cost_bound(content, 8_000) == pytest.approx(0.721556, abs=1e-6)


def test_the_default_model_reserves_its_formula_for_a_long_prompt():
    # 300,000 text bytes put the bound past 272,000 tokens: 301,024 prompt tokens at $20/M and 8,000 output tokens at
    # $75/M is $6.02048 + $0.60
    content = [{'type': 'text', 'text': 'x' * 300_000}]
    assert harness.final_cost_bound(content, 8_000) == pytest.approx(6.62048, abs=1e-6)
