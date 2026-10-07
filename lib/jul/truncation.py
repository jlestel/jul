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

`system_one(..., on_long=)` decides what a cut of the **state** does (also `TypeSafeClient(on_long=)`,
`JUL_ON_LONG`): "cut" (default) answers on what was read; "error" refuses the call, naming the reading and
its limit, so nothing is decided on a text the model did not read. A cut of another part of the input (an
option description past a cross model's `max_option`) is logged and counted, never refused: the state was
read whole.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger("jul.truncation")

ON_LONG = ("cut", "error")
#: What a cut can drop: the state (refused by on_long="error"), or another part of the input (an option).
PARTS = ("state", "option")


@dataclass
class Cuts:
    #: (reading, limit) -> [passes cut, largest cut in tokens]
    by_reading: dict[tuple[str, int], list[int]] = field(default_factory=dict)
    #: the (reading, limit) keys whose cuts dropped part of the state
    state_readings: set[tuple[str, int]] = field(default_factory=set)
    #: set by a call refused by on_long="error": its cuts are neither logged nor handed up
    silent: bool = False

    @property
    def tokens(self) -> int:
        return max((largest for _, largest in self.by_reading.values()), default=0)

    def state_worst(self) -> tuple[str, int, int] | None:
        """(reading, limit, tokens cut) of the largest cut of the state; None when the state was read whole."""
        cut = [(key, self.by_reading[key][1]) for key in self.state_readings]
        if not cut:
            return None
        (reading, limit), tokens = max(cut, key=lambda kv: kv[1])
        return reading, limit, tokens


_current: ContextVar[Cuts | None] = ContextVar("jul_truncation", default=None)


def record(reading: str, limit: int, cut: int, part: str = "state") -> None:
    """`cut` tokens of an input were dropped to fit `reading`'s `limit`; `part` says which part of the
    input they belonged to ("state", or "option" for an option description)."""
    if cut <= 0:
        return
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, not {part!r}")
    cuts = _current.get()
    if cuts is None:
        _log(reading, limit, 1, cut)
        return
    entry = cuts.by_reading.setdefault((reading, limit), [0, 0])
    entry[0] += 1
    entry[1] = max(entry[1], cut)
    if part == "state":
        cuts.state_readings.add((reading, limit))


def _log(reading: str, limit: int, passes: int, cut: int) -> None:
    logger.warning("input cut: %s reads at most %d tokens, %d tokens past it were dropped%s",
                   reading, limit, cut, f" (on {passes} passes)" if passes > 1 else "")


@contextmanager
def tracking():
    """Collects the cuts of one call, logs them when it ends; `.tokens` is the largest. Nested in another
    tracking (a `jul bench` run, an `autotune`), the cuts are handed up and logged once, at the outermost.
    A call refused by `on_long="error"` sets `.silent`: it says so itself."""
    cuts, parent = Cuts(), _current.get()
    token = _current.set(cuts)
    try:
        yield cuts
    finally:
        _current.reset(token)
        for (reading, limit), (passes, cut) in (() if cuts.silent else cuts.by_reading.items()):
            if parent is None:
                _log(reading, limit, passes, cut)
                continue
            entry = parent.by_reading.setdefault((reading, limit), [0, 0])
            entry[0] += passes
            entry[1] = max(entry[1], cut)
            if (reading, limit) in cuts.state_readings:
                parent.state_readings.add((reading, limit))
