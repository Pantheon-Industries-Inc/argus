"""Dictionary requests get one HTTP attempt, including ambiguous transport failures."""
import io
import json
import urllib.error

import pytest

from label import harness


@pytest.mark.parametrize("key", ["sk-or-test", "sk-proj-test"])
@pytest.mark.parametrize("error", ["timeout", "http"])
def test_dictionary_client_never_retries_an_ambiguous_failure(monkeypatch, key, error):
    attempts = []

    def failed(request, timeout):
        attempts.append(request.full_url)
        if error == "http":
            raise urllib.error.HTTPError(request.full_url, 502, "unavailable", {}, io.BytesIO(b"unavailable"))
        raise urllib.error.URLError("timeout after dispatch")

    monkeypatch.setattr(harness.urllib.request, "urlopen", failed)
    monkeypatch.setattr(harness.time, "sleep", lambda seconds: pytest.fail("single attempt slept before a retry"))
    with pytest.raises(RuntimeError):
        harness.call_model_once([{"type": "text", "text": "field descriptors"}], "openai/gpt-6-sol", "low",
                                key, max_tokens=400, timeout=5)
    assert len(attempts) == 1


@pytest.mark.parametrize("key", ["sk-or-test", "sk-proj-test"])
def test_dictionary_client_keeps_normal_request_and_usage(monkeypatch, key):
    sent = []

    def answered(request, timeout):
        sent.append((request.full_url, json.loads(request.data), timeout))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "{}"}}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode())

    monkeypatch.setattr(harness.urllib.request, "urlopen", answered)
    response = harness.call_model_once([{"type": "text", "text": "field descriptors"}], "openai/gpt-6-sol",
                                       "low", key, max_tokens=400, timeout=5)
    assert len(sent) == 1 and sent[0][2] == 5
    body = sent[0][1]
    assert body["max_completion_tokens"] == 400
    assert body["messages"][0]["content"] == [{"type": "text", "text": "field descriptors"}]
    assert body["response_format"] == {"type": "json_object"}
    if key.startswith("sk-or-"):
        assert body["model"] == "openai/gpt-6-sol" and body["reasoning"] == {"effort": "low"}
    else:
        assert body["model"] == "gpt-6-sol" and body["reasoning_effort"] == "low"
        assert response["usage"]["list_cost"] == pytest.approx(0.00004)
    assert response["choices"][0]["message"]["content"] == "{}"


def test_existing_label_client_keeps_its_six_attempt_limit(monkeypatch):
    attempts, waits = [], []

    def failed(request, timeout):
        attempts.append(request.full_url)
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(harness.urllib.request, "urlopen", failed)
    monkeypatch.setattr(harness.time, "sleep", waits.append)
    with pytest.raises(RuntimeError, match="6 HTTP attempts"):
        harness.call_model([{"type": "text", "text": "label request"}], "openai/gpt-6-astra", "medium",
                           "sk-or-test", max_tokens=400, timeout=5)
    assert len(attempts) == 6 and waits == [1, 2, 4, 8, 16]
