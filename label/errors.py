"""Shared exceptions for paid dispatch, independent of CLI module execution."""


class KeyExhausted(RuntimeError):
    """Only an explicit first-attempt HTTP rejection proves an unpaid, replay-safe refusal."""

    def __init__(self, message, *, http_status=None, first_attempt=False, rejection_code=None):
        super().__init__(message)
        self.http_status = http_status
        self.rejection_code = rejection_code
        self.verified_unpaid = (type(http_status) is int and first_attempt is True
                               and (http_status in (401, 402)
                                    or http_status == 429 and rejection_code == 'insufficient_quota'))


class Truncated(RuntimeError):
    """A billed reply cut off at the output limit, carrying the episode cost."""

    def __init__(self, message: str, cost: float):
        super().__init__(message)
        self.cost = cost


class SpendCap(RuntimeError):
    """The final request cannot fit under the remaining batch budget."""
