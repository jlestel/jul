"""on_long="cut|error": what a state over a reading's limit does (jul/truncation.py, client.py)."""

import logging

import pytest

from jul import truncation
from jul.client import TypeSafeClient
from jul.types import Choice, ChoiceAnswer, Noul, NoulAnswer, SystemOneResponse, Usage

LIMIT = 50      # characters stand for tokens in the fake reading below
OPTION_LIMIT = 10


def client(monkeypatch, seen):
    """A client whose reading takes LIMIT characters of state and OPTION_LIMIT of each option description,
    and says "true"/"billing" only if it sees "LATE"."""
    def fake(self, state, questions, *args):
        seen.append(state)
        truncation.record("fake reading", LIMIT, len(state) - LIMIT)
        for q in questions.values():
            for d in (getattr(q, "criteria", None) or {}).values() if isinstance(q, Choice) else ():
                truncation.record("fake reading (choice option)", OPTION_LIMIT, len(d) - OPTION_LIMIT, part="option")
        late = "LATE" in state[:LIMIT]
        answers = {}
        for name, q in questions.items():
            if isinstance(q, Noul):
                answers[name] = NoulAnswer(0.9 if late else 0.2)
            else:
                p = 0.8 if late else 0.3
                answers[name] = ChoiceAnswer("billing" if late else "tech", {"billing": p, "tech": 1 - p}, max(p, 1 - p))
        return SystemOneResponse(answers=answers, model="m", usage=Usage(input_tokens=min(len(state), LIMIT)),
                                 request_id="r")

    monkeypatch.setattr(TypeSafeClient, "_system_one", fake)
    c = TypeSafeClient.__new__(TypeSafeClient)
    c.on_long, c._preset = "cut", None
    return c


QS = {"late": Noul("Is it late?"), "team": Choice("Which team?", {"billing": "", "tech": ""})}
LONG = "word " * 40 + "the parcel is LATE " + "word " * 10


def test_cut_answers_on_what_was_read(monkeypatch, caplog):
    c = client(monkeypatch, [])
    with caplog.at_level(logging.WARNING, logger="jul.truncation"):
        r = c.system_one(LONG, QS)
    assert r.answers["late"].noul == 0.2 and r.usage.truncated_tokens == len(LONG) - LIMIT
    assert "input cut: fake reading" in caplog.text


def test_error_refuses_a_long_state_and_says_why(monkeypatch, caplog):
    c = client(monkeypatch, [])
    with caplog.at_level(logging.WARNING, logger="jul.truncation"), pytest.raises(ValueError, match="fake reading"):
        c.system_one(LONG, QS, on_long="error")
    assert "input refused (on_long=error)" in caplog.text and "input cut" not in caplog.text
    assert c.system_one("short LATE", QS, on_long="error").answers["late"].noul == 0.9   # under the limit: fine


def test_error_does_not_refuse_a_short_state_with_a_long_option(monkeypatch, caplog):
    """Only the state is refused: an option description cut to its own limit is logged and counted."""
    c = client(monkeypatch, [])
    qs = {"team": Choice("Which team?", {"billing": "payments, invoices, refunds and charges", "tech": "bugs"})}
    with caplog.at_level(logging.WARNING, logger="jul.truncation"):
        r = c.system_one("short LATE", qs, on_long="error")
    assert r.answers["team"].choice == "billing"
    assert r.usage.truncated_tokens == len("payments, invoices, refunds and charges") - OPTION_LIMIT
    assert "input cut: fake reading (choice option)" in caplog.text and "refused" not in caplog.text


def test_on_long_comes_from_the_client_then_the_environment(monkeypatch):
    monkeypatch.delenv("JUL_ON_LONG", raising=False)
    assert TypeSafeClient(model="fast").on_long == "cut"          # building a client loads nothing
    assert TypeSafeClient(model="fast", on_long="error").on_long == "error"
    monkeypatch.setenv("JUL_ON_LONG", "error")
    assert TypeSafeClient(model="fast").on_long == "error"
    c = client(monkeypatch, [])
    c.on_long = "error"
    with pytest.raises(ValueError, match="over the input limit"):
        c.system_one(LONG, QS)
    assert c.system_one(LONG, QS, on_long="cut").usage.truncated_tokens > 0   # the call overrides the client


def test_bad_values_are_refused(monkeypatch):
    c = client(monkeypatch, [])
    with pytest.raises(ValueError, match="on_long must be one of cut, error"):
        c.system_one("x", QS, on_long="chunk")
    with pytest.raises(ValueError, match="part must be one of"):
        truncation.record("r", 1, 2, part="question")


def test_a_refused_call_hands_nothing_up(monkeypatch, caplog):
    """In a bench run (an outer tracking), a refused call's cuts are not logged again at the end."""
    c = client(monkeypatch, [])
    with caplog.at_level(logging.WARNING, logger="jul.truncation"):
        with truncation.tracking() as outer:
            with pytest.raises(ValueError):
                c.system_one(LONG, QS, on_long="error")
            c.system_one("short", QS)
    assert outer.tokens == 0 and "input cut" not in caplog.text


def test_bench_counts_the_refused_rows_and_scores_the_others(monkeypatch):
    """jul bench --on-long error: a refused row is counted in refused_rows, the accuracy is over the others."""
    from jul_cli import bench

    c = client(monkeypatch, [])
    c.on_long = "error"
    task = bench.Task("late", "noul", "Is it late?", {"true": "", "false": ""},
                      test=[("short LATE", "true"), (LONG, "true"), ("short", "false")])
    out = bench.evaluate(c, task)
    assert (out["n"], out["correct"], out["accuracy"], out["refused_rows"]) == (2, 2, 1.0, 1)
    c.on_long = "cut"
    out = bench.evaluate(c, task)
    assert out["n"] == 3 and "refused_rows" not in out and out["truncated_rows"] == 1
    c.on_long = "error"
    every = bench.Task("late", "noul", "Is it late?", {"true": "", "false": ""}, test=[(LONG, "true")])
    assert bench.evaluate(c, every) == {"skipped": "every row refused: over the input limit (on_long=error)",
                                        "refused_rows": 1}


def test_the_refusal_is_input_too_long_a_value_error(monkeypatch):
    c = client(monkeypatch, [])
    with pytest.raises(truncation.InputTooLong) as e:
        c.system_one(LONG, QS, on_long="error")
    assert isinstance(e.value, ValueError) and (e.value.reading, e.value.limit) == ("fake reading", LIMIT)
    assert e.value.over == len(LONG) - LIMIT
