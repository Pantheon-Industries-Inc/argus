"""While a new picture is decoded (a seek, a drag along the scrubber) the page marks the video media-pending; the
overlays must stay up for the picture still on screen, so no style keyed on that state may hide anything."""
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / 'board' / 'index.html'
HIDES = re.compile(r'(visibility\s*:\s*hidden|display\s*:\s*none|opacity\s*:\s*0(?![.\d]))', re.I)


def _rules(css):
    """Each (selector, declarations) of a stylesheet, nested blocks (media, container) flattened."""
    out, stack, buf = [], [], ''
    for ch in css:
        if ch == '{':
            stack.append(buf.strip())
            buf = ''
        elif ch == '}':
            if stack:
                sel = stack.pop()
                if buf.strip():
                    out.append((sel, buf))
            buf = ''
        else:
            buf += ch
    return out


def test_nothing_is_hidden_while_a_picture_is_pending():
    page = PAGE.read_text()
    css = '\n'.join(re.findall(r'<style[^>]*>(.*?)</style>', page, re.S))
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    pending = [(sel, body) for sel, body in _rules(css) if 'media-pending' in sel]
    hiding = [sel for sel, body in pending if HIDES.search(body)]
    assert not hiding, hiding


def test_the_pending_state_is_still_set_during_a_seek():
    # the state the rule above guards is the one the page sets while the presented picture is not yet the requested one
    page = PAGE.read_text()
    assert "classList.toggle('media-pending', blocked)" in page
