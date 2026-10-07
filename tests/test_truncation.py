"""An input cut to a reading's limit is logged and counted in usage.truncated_tokens (jul/truncation.py)."""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from jul import truncation
from jul.cross import CrossReader, CrossSpec
from jul.decision import DecisionSpec, PointerReader
from jul.types import Noul, Score, SystemOneResponse, Usage, options_of

D = 4
FIXTURES = Path(__file__).parent / "fixtures"


class StubTokenizer:
    """One token per character; <s> = 0, </s> = 2; the decision delimiters are single tokens."""

    special = {"<s>": 0, "</s>": 2, "<STATE>": 3, "<Q>": 4, "<O>": 5, "</O>": 6, "<D>": 7}

    def encode(self, text, add_special_tokens=True):
        ids = [10 + (ord(c) % 50) for c in text]
        return [0, *ids, 2] if add_special_tokens else ids

    def convert_tokens_to_ids(self, t):
        return self.special[t]

    def convert_ids_to_tokens(self, i):
        return {v: k for k, v in self.special.items()}[i]


class StubEncoder:
    architecture, name = "encoder", "stub"

    def __init__(self):
        self.tokenizer = StubTokenizer()

    def forward_batch(self, queries, layers=(), pools=None, prefix=None):
        return [{l: np.zeros(2 * D, dtype=np.float32) for l in layers} for _ in queries]


def cross_reader(tmp_path):
    zeros = lambda n: np.zeros((n, D), dtype=np.float32)
    np.savez(tmp_path / "cross_heads.npz", noul_weight=zeros(3), noul_bias=np.zeros(3, dtype=np.float32),
             choice_weight=zeros(1), choice_bias=np.zeros(1, dtype=np.float32),
             score_weight=zeros(1), score_bias=np.zeros(1, dtype=np.float32))
    (tmp_path / "cross.json").write_text(json.dumps({"method": "cross", "prefix": "q: ", "max_length": 40,
                                                     "layer": 3, "separator": [2, 2], "types": ["noul", "score"]}))
    return CrossReader(StubEncoder(), CrossSpec.load(tmp_path))


def pointer_reader(tmp_path, max_state=20):
    spec = json.loads((FIXTURES / "decision_minicpm5-2b.json").read_text())
    spec["tokens"] = {"state": "<STATE>", "question": "<Q>", "option_open": "<O>", "option_close": "</O>",
                      "decide": "<D>"}
    spec["limits"]["max_state_tokens"] = max_state
    spec.pop("escape_user_specials", None)
    (tmp_path / "decision.json").write_text(json.dumps(spec))
    return PointerReader(StubEncoder(), DecisionSpec.load(tmp_path))


def test_nothing_is_cut_nothing_is_said(tmp_path, caplog):
    reader = cross_reader(tmp_path)
    q = Noul("Late?")
    with caplog.at_level(logging.WARNING, logger="jul.truncation"), truncation.tracking() as cuts:
        reader.logits("short", "noul", q.instructions, options_of(q))
    assert cuts.tokens == 0 and caplog.records == []


def test_the_cross_cut_is_counted_and_logged_once_per_call(tmp_path, caplog):
    reader = cross_reader(tmp_path)
    q = Score("How urgent?", ["low", "high"])
    with caplog.at_level(logging.WARNING, logger="jul.truncation"), truncation.tracking() as cuts:
        reader.logits("x" * 100, "score", q.instructions, options_of(q))
    # 40 tokens minus <s>, </s></s>, </s>: 36 for the question and the text (100). The largest cut is the
    # pair of the longer level, "q: How urgent?\nhigh" (19 tokens): 83
    assert cuts.tokens == 100 + 19 - 36
    [record] = caplog.records
    assert record.getMessage() == ("input cut: stub cross reads at most 40 tokens, 83 tokens past it were "
                                   "dropped (on 2 passes)")


def test_the_pointer_cut_is_counted(tmp_path):
    reader = pointer_reader(tmp_path, max_state=20)
    with truncation.tracking() as cuts:
        ids = reader.encode_state("y" * 50)
    assert len(ids) == 20 and cuts.tokens == 1 + 50 - 20     # the state delimiter counts in the limit
    with truncation.tracking() as cuts:
        reader.encode_state("short")
    assert cuts.tokens == 0


def test_a_cut_outside_a_call_is_logged_at_once(caplog):
    with caplog.at_level(logging.WARNING, logger="jul.truncation"):
        truncation.record("e5 (onnx)", 512, 30)
    assert "e5 (onnx) reads at most 512 tokens, 30 tokens past it were dropped" in caplog.text


def test_system_one_reports_the_largest_cut_in_usage(monkeypatch, caplog):
    from jul.client import TypeSafeClient

    def fake(self, state, questions, *args):
        truncation.record("a cross", 320, 12)
        truncation.record("a cross", 320, 40)
        truncation.record("a pointer", 384, 7)
        return SystemOneResponse(answers={}, model="m", usage=Usage(input_tokens=9), request_id="r")

    monkeypatch.setattr(TypeSafeClient, "_system_one", fake)
    client = TypeSafeClient.__new__(TypeSafeClient)
    with caplog.at_level(logging.WARNING, logger="jul.truncation"):
        r = client.system_one("text", {"q": Noul("Late?")})
    assert r.usage.truncated_tokens == 40
    assert r.as_dict()["usage"]["truncated_tokens"] == 40
    assert [rec.getMessage() for rec in caplog.records] == [
        "input cut: a cross reads at most 320 tokens, 40 tokens past it were dropped (on 2 passes)",
        "input cut: a pointer reads at most 384 tokens, 7 tokens past it were dropped"]


@pytest.mark.parametrize("cut", [0, -5])
def test_no_cut_records_nothing(cut):
    with truncation.tracking() as cuts:
        truncation.record("r", 10, cut)
    assert cuts.by_reading == {} and cuts.tokens == 0


def test_a_nested_tracking_hands_its_cuts_up_and_logs_once(caplog):
    with caplog.at_level(logging.WARNING, logger="jul.truncation"):
        with truncation.tracking() as outer:
            for cut in (5, 9):
                with truncation.tracking() as inner:
                    truncation.record("r", 10, cut)
                assert inner.tokens == cut
    assert outer.tokens == 9
    assert [r.getMessage() for r in caplog.records] == [
        "input cut: r reads at most 10 tokens, 9 tokens past it were dropped (on 2 passes)"]


def test_bench_counts_the_rows_that_were_cut():
    from jul.types import NoulAnswer
    from jul_cli.bench import Task, evaluate

    class Client:
        def system_one(self, state, questions, **kw):
            truncation.record("r", 10, len(state) - 10)
            return SystemOneResponse(answers={"q": NoulAnswer(0.9)}, model="m",
                                     usage=Usage(truncated_tokens=max(0, len(state) - 10)), request_id="r")

    task = Task(name="t", kind="noul", instructions="Late?", options={"true": "", "false": ""},
                test=[("short", "true"), ("x" * 30, "true"), ("y" * 12, "false")])
    out = evaluate(Client(), task)
    assert out["truncated_rows"] == 2 and out["correct"] == 2      # NoulAnswer 0.9 -> "true" twice out of 3
