"""Shared exceptions for paid dispatch, independent of CLI module execution."""


class DispatchError(RuntimeError):
    """A paid request that returned no completion.

    verified_unpaid is True only when the request provably cost nothing: it never left the machine (the connection
    or the upload failed before the provider had the whole request), or the provider answered with an error status
    instead of a completion. Every other failure (a timeout or reset after the request was sent, a gateway timeout,
    a 200 with no completion) is unknown and keeps its reservation.

    client_timeout is True when our own client gave up: the connection carried no bytes for longer than any live
    reply can be silent (label/harness.py SILENT_CONNECTION_LIMIT_S), so we closed it as dead. It is never the
    provider's failure, and the run record names it apart from one."""

    def __init__(self, message, *, http_status=None, verified_unpaid=False, rejection_code=None, stage=None,
                 client_timeout=False):
        super().__init__(message)
        self.http_status = http_status
        self.rejection_code = rejection_code
        self.verified_unpaid = verified_unpaid is True
        self.stage = stage
        self.client_timeout = client_timeout is True


class KeyExhausted(DispatchError):
    """A key that is out of credit or revoked. Only an explicit first-attempt HTTP rejection proves an unpaid,
    replay-safe refusal."""

    def __init__(self, message, *, http_status=None, first_attempt=False, rejection_code=None):
        super().__init__(message, http_status=http_status, rejection_code=rejection_code,
                         verified_unpaid=(type(http_status) is int and first_attempt is True
                                          and (http_status in (401, 402)
                                               or http_status == 429 and rejection_code == 'insufficient_quota')),
                         stage='rejected')


class Truncated(RuntimeError):
    """A billed reply cut off at the output limit, carrying the episode cost."""

    def __init__(self, message: str, cost: float):
        super().__init__(message)
        self.cost = cost


class SpendCap(RuntimeError):
    """The final request cannot fit under the remaining batch budget."""
