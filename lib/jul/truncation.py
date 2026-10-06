"""Inputs cut to a reading's limit: logged each time, and counted in `usage.truncated_tokens`.

Each reading takes a bounded input (a cross model the `max_length` / `max_state` of its `cross.json`, a
pointer model the `limits.max_state_tokens` of its `decision.json`, an encoder its positions, the onnx
backend `JUL_ONNX_MAX_TOKENS`...): the limit is read from the model's files, never fixed here.
Past it the end of the text is dropped, and an answer that sits there is never seen. The cut itself is
unchanged here; what this adds is that it is never silent:

- a call to `system_one` collects the cuts of all its passes; when it returns, one warning per reading is
  logged (one per run for `autotune` and `jul bench`, which wrap many calls) on the `jul.truncation` logger (printed by `jul serve`, and by Python's last-resort handler
  when nothing is configured), naming the reading, its limit and the tokens dropped;
- `usage.truncated_tokens` is the largest cut of the call (0 when every reading saw its whole input), so
  an HTTP client sees it too. The largest, not a sum: a Score read once per level cuts the same text
  once per level, and the question a client has is how much of its input a reading missed;
- a cut outside a call (a template run directly) is logged at once.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger("jul.truncation")


@dataclass
class Cuts:
    #: (reading, limit) -> [passes cut, largest cut in tokens]
    by_reading: dict[tuple[str, int], list[int]] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return max((largest for _, largest in self.by_reading.values()), default=0)


_current: ContextVar[Cuts | None] = ContextVar("jul_truncation", default=None)


def record(reading: str, limit: int, cut: int) -> None:
    """`cut` tokens of an input were dropped to fit `reading`'s `limit`."""
    if cut <= 0:
        return
    cuts = _current.get()
    if cuts is None:
        _log(reading, limit, 1, cut)
        return
    entry = cuts.by_reading.setdefault((reading, limit), [0, 0])
    entry[0] += 1
    entry[1] = max(entry[1], cut)


def _log(reading: str, limit: int, passes: int, cut: int) -> None:
    logger.warning("input cut: %s reads at most %d tokens, %d tokens past it were dropped%s",
                   reading, limit, cut, f" (on {passes} passes)" if passes > 1 else "")


@contextmanager
def tracking():
    """Collects the cuts of one call, logs them when it ends; `.tokens` is the largest. Nested in another
    tracking (a `jul bench` run, an `autotune`), the cuts are handed up and logged once, at the outermost."""
    cuts, parent = Cuts(), _current.get()
    token = _current.set(cuts)
    try:
        yield cuts
    finally:
        _current.reset(token)
        for (reading, limit), (passes, cut) in cuts.by_reading.items():
            if parent is None:
                _log(reading, limit, passes, cut)
                continue
            entry = parent.by_reading.setdefault((reading, limit), [0, 0])
            entry[0] += passes
            entry[1] = max(entry[1], cut)
