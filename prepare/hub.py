"""Reading datasets from the Hugging Face Hub: listing folders, downloading files, reading whole files and byte
ranges.

A token is used only when HF_TOKEN is set; it is needed only for datasets that ask you to accept their terms
first (ABC-130k, 10Kh-RealOmin, Egocentric-100K, Gen-HumanEgo).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


def token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def _headers(extra: dict | None = None) -> dict:
    h = dict(extra or {})
    if token():
        h["Authorization"] = f"Bearer {token()}"
    return h


def get(url: str, rng: tuple[int, int] | None = None, timeout: int = 300) -> tuple[bytes, dict]:
    """GET, retried up to 6 times on network errors, server errors and rate limits (a missing file or a refused
    token fails at once); rng = (first, last) byte, inclusive. Returns (body, headers)."""
    h = _headers({"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else None)
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code not in (408, 429):
                raise RuntimeError(f"GET {url} {rng or ''}: HTTP {e.code}") from e
            last = e
        except Exception as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GET {url} {rng or ''}: {last}")


def resolve(repo: str, rel: str) -> str:
    """The download URL of one file of a dataset repo."""
    return f"https://huggingface.co/datasets/{repo}/resolve/main/" + urllib.parse.quote(rel)


def fetch(repo: str, rel: str) -> bytes:
    return get(resolve(repo, rel))[0]


def ls(repo: str, path: str = "") -> list[dict]:
    """One level of the repo's tree: [{"type": "file" | "directory", "path": ...}]."""
    api = f"https://huggingface.co/api/datasets/{repo}/tree/main"
    return json.loads(get(api + ("/" + urllib.parse.quote(path) if path else ""))[0])


def download(repo: str, rel: str, raw_root) -> str:
    """One file into raw_root/<rel> (not downloaded again when it is already there); returns its local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, rel, repo_type="dataset", local_dir=str(raw_root), token=token())
