"""Map native camera measurements onto anchor intervals without inventing observations."""
import numpy as np


def to_anchor(ep: dict, v: str, own: np.ndarray, T: int) -> np.ndarray | None:
    """Per-anchor-interval values (len T-1) from a camera's own per-frame-pair values (len n-1).
    A camera without a kmap must have exactly the anchor's frames. With a kmap, interval i sums the
    camera's own changes from frame km[i] to km[i+1]; NaN when the camera has no new frame there."""
    km = ep["kmap"].get(v)
    n = len(own) + 1
    if km is None:
        return own[:T - 1].astype(np.float64) if n == T else None
    km = np.asarray(km, dtype=np.int64)[:T]
    if len(km) < T:
        return None
    km = np.clip(km, 0, n - 1)
    finite = np.isfinite(own)
    cum = np.concatenate([[0.0], np.cumsum(np.where(finite, own, 0.0))])
    bad = np.concatenate([[0], np.cumsum(~finite)])
    a, b = km[:-1], km[1:]
    out = cum[b] - cum[a]
    out[(b <= a) | ((bad[b] - bad[a]) > 0)] = np.nan
    return out
