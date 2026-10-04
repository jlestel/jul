"""`jul bench`: reading the user's rows, the overlap check, scoring and the report. No model is loaded:
a fake client answers."""

import argparse
import json

import pytest

from jul import ChoiceAnswer, NoulAnswer, ScoreAnswer
from jul.types import SystemOneResponse, Usage
from jul_cli import bench
from jul_cli.main import build_parser

Q = "Which team?"
OPTS = {"billing": "payments", "tech": "bugs"}


def rows(*items, kind="choice", question=Q, options=OPTS):
    return [{"type": kind, "question": question, "options": options, "state": s, "answer": a} for s, a in items]


def write(path, data):
    path.write_text("".join(json.dumps(r) + "\n" for r in data))
    return path


class FakeClient:
    """Answers `billing` when the text says "charge", else `tech`; after autotune, always right."""

    model = "fake"

    def __init__(self, *_, **__):
        self.tuned = set()

    def system_one(self, state, questions, context=None):
        q = questions["q"]
        right = context is not None and id(context) in self.tuned
        if hasattr(q, "criteria") and isinstance(q.criteria, dict):
            pick = ("billing" if "charge" in state else "tech") if not right else self.gold[state]
            ans = ChoiceAnswer(choice=pick, probabilities={}, confidence=1.0)
        elif type(q).__name__ == "Noul":
            ans = NoulAnswer(noul=0.9 if "charge" in state else 0.1)
        else:
            ans = ScoreAnswer(score=1.2, legend={}, probabilities={}, confidence=1.0)
        return SystemOneResponse(answers={"q": ans}, model="fake", usage=Usage(), request_id="r")

    def autotune(self, ctx, questions, labeled, save=True):
        self.tuned.add(id(ctx))

        class R:
            activated, reason = True, "beats zero-shot"
        return {"q": R()}

    def close(self):
        pass


def args(test, train=None, **kw):
    base = dict(test=str(test), train=str(train) if train else None, models="fake", backend=None, output=None,
                json=True, near=0.8, drop_overlap=False, allow_overlap=False, quiet=True)
    base.update(kw)
    return argparse.Namespace(**base)


def test_rows_are_grouped_by_question(tmp_path):
    data = rows(("I was charged twice", "billing"), ("app crashes", "tech"))
    data += rows(("charge me", True), ("hello", "false"), kind="noul", question="Is it about money?", options=None)
    tasks = bench.load_tasks(write(tmp_path / "t.jsonl", data))
    assert [t.kind for t in tasks] == ["choice", "noul"]
    assert tasks[1].test == [("charge me", "true"), ("hello", "false")]


def test_csv_options_and_score_levels(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("type,question,options,state,answer\n"
                 "choice,Which team?,billing:payments|tech:bugs,charged twice,billing\n"
                 "score,How angry?,calm|annoyed|furious,THIS IS A JOKE,furious\n")
    choice, score = bench.load_tasks(p)
    assert choice.options == {"billing": "payments", "tech": "bugs"}
    assert score.options == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert score.test == [("THIS IS A JOKE", "2")]


def test_an_answer_outside_the_options_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="not one of the options"):
        bench.load_tasks(write(tmp_path / "t.jsonl", rows(("x", "sales"))))


def test_train_rows_must_answer_a_test_question(tmp_path):
    test = write(tmp_path / "test.jsonl", rows(("x", "tech")))
    train = write(tmp_path / "train.jsonl", rows(("y", "tech"), question="Another question?"))
    with pytest.raises(SystemExit, match="not in the test set"):
        bench.load_tasks(test, train)


def test_overlap_folds_case_punctuation_and_ids(tmp_path):
    test = write(tmp_path / "test.jsonl", rows(("Charge van ABC-123 at 11pm!", "billing"),
                                               ("Please charge the van tonight at eleven", "billing"),
                                               ("totally different words here", "tech")))
    train = write(tmp_path / "train.jsonl", rows(("charge van abc-987 at 10pm", "billing"),
                                                 ("Please charge the van tonight at eleven.", "tech")))
    ov = bench.find_overlap(bench.load_tasks(test, train))
    assert {o["test"] for o in ov["exact"]} == {"Charge van ABC-123 at 11pm!",
                                                "Please charge the van tonight at eleven"}
    assert ov["near"] == []


def test_near_duplicates_are_reported_not_blocking(tmp_path):
    test = write(tmp_path / "test.jsonl", rows(("my invoice for september shows a double charge", "billing")))
    train = write(tmp_path / "train.jsonl", rows(("my invoice for september shows a double charge again", "billing")))
    ov = bench.find_overlap(bench.load_tasks(test, train))
    assert ov["exact"] == [] and len(ov["near"]) == 1


def test_exact_overlap_stops_the_bench_unless_allowed(tmp_path):
    test = write(tmp_path / "test.jsonl", rows(("charged twice", "billing"), ("app crashes", "tech")))
    train = write(tmp_path / "train.jsonl", rows(("Charged twice.", "billing"), ("login fails", "tech")))
    with pytest.raises(SystemExit, match="also in train"):
        bench.run(args(test, train), make_client=FakeClient)
    report = bench.run(args(test, train, drop_overlap=True), make_client=_fake_with_gold(test))
    assert report["overlap"]["dropped"] == 1 and report["questions"][0]["n_test"] == 1


def test_duplicates_inside_test_are_reported(tmp_path):
    test = write(tmp_path / "t.jsonl", rows(("app crashes", "tech"), ("App crashes!", "tech")))
    report = bench.run(args(test), make_client=FakeClient)
    assert report["overlap"]["within_test"][0]["count"] == 2


def _fake_with_gold(*paths):
    gold = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            gold[r["state"]] = r["answer"]

    def make(*a):
        c = FakeClient()
        c.gold = gold
        return c
    return make


def test_zero_shot_then_autotune_scores_and_json_output(tmp_path):
    test = write(tmp_path / "test.jsonl", rows(("charge again", "billing"), ("charge please", "tech"),
                                               ("crash", "tech"), ("bug", "tech")))
    train = write(tmp_path / "train.jsonl", rows(("invoice wrong", "billing"), ("it froze", "tech")))
    out = tmp_path / "res.json"
    report = bench.run(args(test, train, output=str(out)), make_client=_fake_with_gold(test))
    q = report["results"][0]["questions"][Q]
    assert q["zero_shot"]["accuracy"] == 0.75 and q["autotune"]["accuracy"] == 1.0
    assert q["zero_shot"]["ci95"][0] < 0.75 < q["zero_shot"]["ci95"][1]
    assert json.loads(out.read_text())["recommendation"][Q]["pick"]["setting"] == "autotune"


def test_without_train_there_is_no_autotune(tmp_path):
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))
    report = bench.run(args(test), make_client=FakeClient)
    assert "autotune" not in report["results"][0]["questions"][Q]
    assert report["overlap"]["checked"] is False


def test_a_failing_model_does_not_lose_the_others(tmp_path):
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))

    def make(model, *_):
        if model == "broken":
            raise RuntimeError("no weights")
        return FakeClient()
    report = bench.run(args(test, models="broken,fake"), make_client=make)
    assert "no weights" in report["results"][0]["error"]
    assert report["results"][1]["questions"][Q]["zero_shot"]["n"] == 2
    assert report["recommendation"][Q]["pick"]["model"] == "fake"


def test_recommendation_prefers_the_fastest_when_not_separable():
    task = bench.Task("q", "choice", "q?", OPTS, test=[("a", "tech")] * 20)
    slow = {"model": "big", "questions": {"q": {"zero_shot": {"accuracy": 0.85, "ci95": [0.64, 0.95],
                                                               "latency_ms_p50": 90}}}}
    fast = {"model": "small", "questions": {"q": {"zero_shot": {"accuracy": 0.80, "ci95": [0.58, 0.92],
                                                                 "latency_ms_p50": 10}}}}
    rec = bench.recommend([slow, fast], [task])["q"]
    assert rec["best"]["model"] == "big" and rec["pick"]["model"] == "small" and not rec["separable"]


def test_the_plain_render_has_no_escape_codes(tmp_path):
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))
    report = bench.run(args(test), make_client=FakeClient)

    class Plain:
        on = False
        ph = dim = bold = staticmethod(lambda s: s)
        inv = staticmethod(lambda s: f"[{s}]")
    text = bench.render(report, Plain())
    assert "\x1b" not in text and "Which team?" in text and "100.0%" in text


def test_the_parser_knows_bench():
    a = build_parser().parse_args(["bench", "t.jsonl", "--train", "tr.csv", "--models", "fast,accurate",
                                   "-O", "r.json"])
    assert a.fn.__name__ == "cmd_bench" and a.output == "r.json" and a.train == "tr.csv"
