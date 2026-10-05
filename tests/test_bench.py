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

    def system_one(self, state, questions, context=None, method=None):
        if method == "letters":
            raise NotImplementedError("no logits")
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

    def autotune(self, ctx, questions, labeled, save=True, features="vector"):
        if features == "lexical":
            raise ImportError("scikit-learn is needed")
        self.tuned.add(id(ctx))

        class R:
            activated, reason = True, "beats zero-shot"
        return {"q": R()}

    def close(self):
        pass


def args(test, train=None, **kw):
    base = dict(test=str(test), train=str(train) if train else None, models="fake", backend=None, output=None,
                method=None, features=None,
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
    zs, tuned = report["results"][0]["questions"][Q]
    assert zs["setting"] == "zero-shot" and zs["accuracy"] == 0.75
    assert tuned["setting"] == "autotune" and tuned["accuracy"] == 1.0
    assert zs["ci95"][0] < 0.75 < zs["ci95"][1]
    assert json.loads(out.read_text())["recommendation"][Q]["pick"]["setting"] == "autotune"


def test_without_train_there_is_no_autotune(tmp_path):
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))
    report = bench.run(args(test), make_client=FakeClient)
    assert [r["setting"] for r in report["results"][0]["questions"][Q]] == ["zero-shot"]
    assert report["overlap"]["checked"] is False


def test_a_failing_model_does_not_lose_the_others(tmp_path):
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))

    def make(model, *_):
        if model == "broken":
            raise RuntimeError("no weights")
        return FakeClient()
    report = bench.run(args(test, models="broken,fake"), make_client=make)
    assert "no weights" in report["results"][0]["error"]
    assert report["results"][1]["questions"][Q][0]["n"] == 2
    assert report["recommendation"][Q]["pick"]["model"] == "fake"


def test_recommendation_prefers_the_fastest_when_not_separable():
    task = bench.Task("q", "choice", "q?", OPTS, test=[("a", "tech")] * 20)
    slow = {"model": "big", "questions": {"q": [{"setting": "zero-shot", "accuracy": 0.85, "ci95": [0.64, 0.95],
                                                 "latency_ms_p50": 90}]}}
    fast = {"model": "small", "questions": {"q": [{"setting": "zero-shot", "accuracy": 0.80, "ci95": [0.58, 0.92],
                                                   "latency_ms_p50": 10}]}}
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


def test_readings_are_compared_and_impossible_ones_reported(tmp_path):
    test = write(tmp_path / "test.jsonl", rows(("charge again", "billing"), ("charge me not", "tech"),
                                               ("crash", "tech")))
    train = write(tmp_path / "train.jsonl", rows(("invoice wrong", "billing"), ("it froze", "tech")))
    report = bench.run(args(test, train, method="vector,letters", features="vector,lexical"),
                       make_client=_fake_with_gold(test))
    runs = {r["setting"]: r for r in report["results"][0]["questions"][Q]}
    assert set(runs) == {"zero-shot:vector", "zero-shot:letters", "autotune:vector", "autotune:lexical"}
    assert "skipped" in runs["zero-shot:letters"] and "skipped" in runs["autotune:lexical"]
    assert runs["autotune:vector"]["accuracy"] == 1.0
    assert report["recommendation"][Q]["best"]["setting"] == "autotune:vector"


def test_auto_and_unknown_readings():
    assert bench._choices("auto", bench.METHODS, "--method") == ["vector", "letters", "cross"]
    with pytest.raises(SystemExit, match="unknown"):
        bench._choices("vector,guess", bench.METHODS, "--method")


def test_an_autotune_worse_on_test_is_not_picked_over_zero_shot():
    task = bench.Task("q", "choice", "q?", OPTS, test=[("a", "tech")] * 50)
    r = {"model": "m", "questions": {"q": [
        {"setting": "zero-shot", "accuracy": 0.9, "ci95": [0.79, 0.96], "latency_ms_p50": 10},
        {"setting": "autotune", "accuracy": 0.6, "ci95": [0.46, 0.72], "latency_ms_p50": 10, "activated": True}]}}
    assert bench.recommend([r], [task])["q"]["pick"]["setting"] == "zero-shot"


def test_non_latin_texts_are_not_folded_to_nothing(tmp_path):
    test = write(tmp_path / "test.jsonl", rows(("Мне дважды списали деньги", "billing"), ("应用崩溃了", "tech")))
    train = write(tmp_path / "train.jsonl", rows(("Приложение не запускается", "tech"), ("发票有误", "billing")))
    ov = bench.find_overlap(bench.load_tasks(test, train))
    assert ov["exact"] == [] and ov["within_test"] == []
    assert bench.normalize("Éléphant ÉTÉ") == "elephant ete"


def test_same_wording_with_other_options_stays_two_questions(tmp_path):
    data = rows(("charged", "billing")) + rows(("charged", "sales"), options={"sales": "s", "support": "t"})
    a, b = bench.load_tasks(write(tmp_path / "t.jsonl", data))
    assert a.name != b.name


def test_csv_with_a_byte_order_mark(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes("\ufefftype,question,options,state,answer\nnoul,Is it a bug?,,app crashes,yes\n".encode())
    (task,) = bench.load_tasks(p)
    assert task.kind == "noul" and task.test == [("app crashes", "true")]


def test_score_levels_round_half_up():
    class A:
        score = 1.5
    assert bench.predicted("score", A()) == "2"
    A.score = 2.5
    assert bench.predicted("score", A()) == "3"


def test_overlap_scales(tmp_path):
    import random
    import time
    random.seed(0)
    words = [f"w{i}" for i in range(3000)]
    def text():
        return " ".join(random.choice(words) for _ in range(12))
    test = write(tmp_path / "test.jsonl", rows(*[(text(), "tech") for _ in range(2000)]))
    train = write(tmp_path / "train.jsonl", rows(*[(text(), "tech") for _ in range(2000)]))
    tasks = bench.load_tasks(test, train)
    t = time.perf_counter()
    bench.find_overlap(tasks)
    assert time.perf_counter() - t < 20


# --- remote System One servers ------------------------------------------------------------------

def _fake_remote(monkeypatch, sent):
    from jul.escalate import SystemOneHTTP

    def answer(self, state, questions, **_):
        sent.append((self.url, self.model, self.api_key, state))
        return FakeClient().system_one(state, questions)
    monkeypatch.setattr(SystemOneHTTP, "system_one", answer)


def test_remote_targets_are_recognised():
    for m in ("typesafe", "ollama", "ollama:nimble", "cloudflare:clef", "https://kev.example#kev-4b",
              "http://localhost:8577"):
        assert bench.is_remote(m), m
    for m in ("fast", "jul-decision-e5-small", "minicpm5-2b", "laya", "laya:multilingual"):
        assert not bench.is_remote(m), m


def test_a_remote_server_is_benched_zero_shot_with_the_users_key(tmp_path, monkeypatch):
    sent = []
    _fake_remote(monkeypatch, sent)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k-ts")
    monkeypatch.setenv("KEV_KEY", "k-kev")
    test = write(tmp_path / "test.jsonl", rows(("charge again", "billing"), ("crash", "tech")))
    train = write(tmp_path / "train.jsonl", rows(("invoice wrong", "billing"), ("it froze", "tech")))
    report = bench.run(args(test, train, models="fake,typesafe,https://kev.example/v1#kev-4b",
                            remote_key_env="KEV_KEY", features="auto"),
                       make_client=_fake_with_gold(test))
    fake, jev, kev = report["results"]
    assert [r["setting"] for r in fake["questions"][Q]][:2] == ["zero-shot", "autotune:vector"]
    assert jev["remote"] == "https://api.typesafe.ai/v1/systemone"
    assert kev["remote"] == "https://kev.example/v1/systemone"
    for r in (jev, kev):
        runs = r["questions"][Q]
        assert runs[0]["setting"] == "zero-shot" and runs[0]["accuracy"] == 1.0
        assert all("remote server" in x["skipped"] for x in runs[1:]) and len(runs) == 4
    assert {(u, m, k) for u, m, k, _ in sent} == {
        ("https://api.typesafe.ai/v1/systemone", "jev-latest", "k-ts"),
        ("https://kev.example/v1/systemone", "kev-4b", "k-kev")}
    assert "the test rows were sent there" in bench.render(report, bench.Ink(open(tmp_path / "o", "w")))


def test_a_remote_server_without_its_key_fails_alone_and_sends_nothing(tmp_path, monkeypatch):
    sent = []
    _fake_remote(monkeypatch, sent)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))
    report = bench.run(args(test, models="typesafe,cloudflare:clef,fake"), make_client=FakeClient)
    jev, clef, fake = report["results"]
    assert "TYPESAFE_API_KEY is not set" in jev["error"]
    assert "CLOUDFLARE_ACCOUNT_ID" in clef["error"]
    assert fake["questions"][Q][0]["n"] == 2 and sent == []


def test_bench_does_not_ask_setup_for_a_remote_target(monkeypatch, tmp_path):
    import importlib
    m = importlib.import_module("jul_cli.main")
    seen = []
    monkeypatch.setattr("jul_cli.setup.require_setup", lambda name, backend: seen.append(name))
    monkeypatch.setattr("jul_cli.bench.run", lambda a: None)
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing")))
    m.main(["bench", str(test), "--models", "fake,typesafe,https://x.example#kev"])
    assert seen == ["fake"]


def test_url_credentials_never_reach_the_report_or_the_log(tmp_path, monkeypatch, capsys):
    _fake_remote(monkeypatch, [])
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))
    report = bench.run(args(test, models="https://u:sekret-pw@kev.example:8443#kev-4b", quiet=False),
                       make_client=FakeClient)
    r = report["results"][0]
    assert r["remote"] == "https://kev.example:8443/v1/systemone"
    assert r["model"] == "https://kev.example:8443#kev-4b"
    out = capsys.readouterr()
    assert "sekret" not in json.dumps(report) + out.out + out.err


def test_a_missing_url_key_does_not_echo_url_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))
    report = bench.run(args(test, models="https://u:sekret-pw@kev.example#kev", remote_key_env="NOPE_KEY"),
                       make_client=FakeClient)
    assert "NOPE_KEY is not set" in report["results"][0]["error"] and "sekret" not in json.dumps(report)


def test_cloudflare_honours_remote_key_env_like_serve(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "default-token")
    monkeypatch.setenv("MY_CF", "mine")
    assert bench.remote_client("cloudflare:clef", "MY_CF").api_key == "mine"
    assert bench.remote_client("cloudflare:clef").api_key == "default-token"
# --- cascade: local first, a remote server for the unsure answers ---------------------------------

class UnsureClient(FakeClient):
    """Sure only of `billing`; unsure of everything else."""

    def system_one(self, state, questions, context=None, method=None):
        r = super().system_one(state, questions, context, method)
        a = r.answers["q"]
        if isinstance(a, ChoiceAnswer) and a.choice != "billing":
            r.answers["q"] = ChoiceAnswer(choice=a.choice, probabilities={}, confidence=0.3)
        return r


def test_the_cascade_reports_accuracy_and_the_share_sent(tmp_path, monkeypatch):
    from jul.escalate import SystemOneHTTP
    sent = []

    def remote(self, state, questions, **_):      # the remote server knows the gold
        sent.append(state)
        return SystemOneResponse(answers={"q": ChoiceAnswer(choice=gold[state], probabilities={}, confidence=1.0)},
                                 model="jev", usage=Usage(), request_id="r")
    monkeypatch.setattr(SystemOneHTTP, "system_one", remote)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    gold = {"charge again": "billing", "crash": "tech", "bug charge": "tech", "froze": "billing"}
    test = write(tmp_path / "t.jsonl", rows(*gold.items()))
    report = bench.run(args(test, escalate_to="typesafe", min_confidence=0.8), make_client=lambda *a: UnsureClient())
    zs, casc = report["results"][0]["questions"][Q]
    assert zs["setting"] == "zero-shot" and zs["accuracy"] == 0.5 and "escalated" not in zs
    assert casc["setting"] == "cascade@0.8" and casc["escalate_to"] == "typesafe"
    assert casc["accuracy"] == 0.75                   # "bug charge" is answered billing, sure: never sent
    assert casc["escalated"] == 2 and casc["escalated_share"] == 0.5
    assert sorted(sent) == ["crash", "froze"]
    assert report["escalate"]["url"] == "https://api.typesafe.ai/v1/systemone"
    text = bench.render(report, bench.Ink(open(tmp_path / "o", "w")))
    assert "50% sent (2/4)" in text and "cascade: local first" in text


def test_the_cascade_needs_a_remote_target_and_its_key(tmp_path, monkeypatch):
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))
    with pytest.raises(SystemExit, match="not a remote server"):
        bench.run(args(test, escalate_to="fast", min_confidence=0.8), make_client=FakeClient)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    report = bench.run(args(test, escalate_to="typesafe", min_confidence=0.8), make_client=FakeClient)
    zs, casc = report["results"][0]["questions"][Q]
    assert zs["accuracy"] == 1.0 and "TYPESAFE_API_KEY is not set" in casc["skipped"]


def test_a_remote_model_is_not_cascaded(tmp_path, monkeypatch):
    _fake_remote(monkeypatch, [])
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))
    report = bench.run(args(test, models="typesafe", escalate_to="typesafe", min_confidence=0.8),
                       make_client=FakeClient)
    assert [r["setting"] for r in report["results"][0]["questions"][Q]] == ["zero-shot"]


def test_the_parser_knows_the_cascade():
    a = build_parser().parse_args(["bench", "t.jsonl", "--escalate-to", "typesafe", "--min-confidence", "0.7"])
    assert a.escalate_to == "typesafe" and a.min_confidence == 0.7


def test_a_rejected_remote_key_is_reported_not_hidden_in_the_cascade(tmp_path, monkeypatch):
    from jul.escalate import RemoteError, SystemOneHTTP

    def rejected(self, state, questions, **_):
        raise RemoteError(f"{self.url} answered HTTP 401: bad key")
    monkeypatch.setattr(SystemOneHTTP, "system_one", rejected)
    monkeypatch.setenv("TYPESAFE_API_KEY", "wrong")
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech"), ("froze", "tech")))
    report = bench.run(args(test, escalate_to="typesafe", min_confidence=0.8), make_client=lambda *a: UnsureClient())
    zs, casc = report["results"][0]["questions"][Q]
    assert "accuracy" not in casc and casc["skipped"].startswith("typesafe answered HTTP 401")
    assert "every escalated row" in casc["skipped"] and "api.typesafe.ai" not in casc["skipped"]
    assert report["recommendation"][Q]["pick"]["setting"] == "zero-shot"


def test_some_remote_failures_are_counted_on_the_row(tmp_path, monkeypatch):
    from jul.escalate import RemoteError, SystemOneHTTP

    def flaky(self, state, questions, **_):
        if state == "crash":
            raise RemoteError("timeout")
        return SystemOneResponse(answers={"q": ChoiceAnswer(choice="tech", probabilities={}, confidence=1.0)},
                                 model="jev", usage=Usage(), request_id="r")
    monkeypatch.setattr(SystemOneHTTP, "system_one", flaky)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech"), ("froze", "tech")))
    report = bench.run(args(test, escalate_to="typesafe", min_confidence=0.8), make_client=lambda *a: UnsureClient())
    casc = report["results"][0]["questions"][Q][1]
    assert casc["escalated"] == 2 and casc["remote_errors"] == 1 and "timeout" in casc["remote_error"]
    assert "1 failed there" in bench.render(report, bench.Ink(open(tmp_path / "o", "w")))


def test_a_local_model_failing_inside_the_cascade_is_not_shown_as_all_remote(tmp_path, monkeypatch):
    _fake_remote(monkeypatch, [])
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing"), ("crash", "tech")))
    report = bench.run(args(test, method="letters", escalate_to="typesafe", min_confidence=0.8),
                       make_client=FakeClient)
    zs, casc = report["results"][0]["questions"][Q]
    assert "no logits" in zs["skipped"] and "local model failed" in casc["skipped"]


def test_the_pick_line_shows_what_the_cascade_costs(tmp_path, monkeypatch):
    from jul.escalate import SystemOneHTTP
    gold = {"charge again": "billing", "crash": "tech", "froze": "billing"}

    def remote(self, state, questions, **_):
        return SystemOneResponse(answers={"q": ChoiceAnswer(choice=gold[state], probabilities={}, confidence=1.0)},
                                 model="jev", usage=Usage(), request_id="r")
    monkeypatch.setattr(SystemOneHTTP, "system_one", remote)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    test = write(tmp_path / "t.jsonl", rows(*gold.items()))
    report = bench.run(args(test, escalate_to="typesafe", min_confidence=0.8), make_client=lambda *a: UnsureClient())
    pick = report["recommendation"][Q]["pick"]
    assert pick["setting"] == "cascade@0.8" and pick["escalated_share"] == round(2 / 3, 4)
    assert "67% of rows sent to typesafe" in bench.render(report, bench.Ink(open(tmp_path / "o", "w")))


def test_min_confidence_must_be_a_confidence(tmp_path):
    test = write(tmp_path / "t.jsonl", rows(("charge", "billing")))
    for bad in (1.5, -0.1):
        with pytest.raises(SystemExit, match="between 0 and 1"):
            bench.run(args(test, escalate_to="typesafe", min_confidence=bad), make_client=FakeClient)
