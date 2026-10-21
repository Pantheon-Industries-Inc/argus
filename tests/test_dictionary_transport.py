"""A paid request is sent once when its outcome is unknown, and retried only when it provably cost nothing."""
import http.client
import io
import json
import urllib.error

import pytest

from label import harness


@pytest.mark.parametrize("key", ["sk-or-test", "sk-proj-test"])
@pytest.mark.parametrize("error", ["read_timeout", "reset_after_send", "gateway_timeout"])
def test_paid_client_never_retries_an_unknown_outcome(monkeypatch, key, error):
    # urllib raises these only after the whole request was sent, so the provider may have billed it
    attempts = []

    def failed(request, timeout):
        attempts.append(request.full_url)
        if error == "gateway_timeout":
            raise urllib.error.HTTPError(request.full_url, 504, "gateway timeout", {}, io.BytesIO(b"timeout"))
        if error == "reset_after_send":
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(harness.urllib.request, "urlopen", failed)
    monkeypatch.setattr(harness.time, "sleep", lambda seconds: pytest.fail("single attempt slept before a retry"))
    with pytest.raises(harness.DispatchError) as raised:
        harness.call_model_once([{"type": "text", "text": "field descriptors"}], "openai/gpt-6-sol", "low",
                                key, max_tokens=400, timeout=5)
    assert len(attempts) == 1 and raised.value.verified_unpaid is False


@pytest.mark.parametrize("key", ["sk-or-test", "sk-proj-test"])
@pytest.mark.parametrize("error", ["refused", "http502", "http503", "http429"])
def test_paid_client_retries_only_provably_unpaid_failures(monkeypatch, key, error):
    # a connection refused before the request was sent, or an error status instead of a completion, cost nothing
    attempts, slept = [], []

    def failed(request, timeout):
        attempts.append(request.full_url)
        if error == "refused":
            raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        raise urllib.error.HTTPError(request.full_url, int(error[4:]), "busy", {}, io.BytesIO(b"busy"))

    monkeypatch.setattr(harness.urllib.request, "urlopen", failed)
    monkeypatch.setattr(harness.time, "sleep", slept.append)
    with pytest.raises(harness.DispatchError) as raised:
        harness.call_model_once([{"type": "text", "text": "field descriptors"}], "openai/gpt-6-sol", "low",
                                key, max_tokens=400, timeout=5)
    assert len(attempts) == harness.UNPAID_ATTEMPTS and len(slept) == harness.UNPAID_ATTEMPTS - 1
    assert raised.value.verified_unpaid is True


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
    # a caller's short timeout never cuts off a reply: the socket waits as long as a live reply can be silent
    assert len(sent) == 1 and sent[0][2] == harness.SILENT_CONNECTION_LIMIT_S
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
