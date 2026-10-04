"""Laya as a model: `TypeSafeClient(model="laya")` hands the questions to the `laya` package.

The fast tests stand a fake `laya` module in for the real one, answering in the shape laya 0.3.27
returns (recorded from a real call); the slow one loads the real checkpoint.
"""

import sys
import types

import pytest
from jul import Choice, Context, Escalation, Noul, NoulCriteria, Score, TypeSafeClient
from jul.types import ChoiceAnswer, NoulAnswer, ScoreAnswer

Q = {"team": Choice("Which team?", {"billing": "payments, invoices", "tech": "bugs, outages"}),
     "urgent": Noul("Is it urgent?", NoulCriteria(true="needs action today", false="can wait")),
     "anger": Score("How angry?", ["calm", "annoyed", "furious"]),
     "topic": Choice("Topic?", ["sport", "finance"])}

#: What laya 0.3.27's `Agent.system_one` returned for Q (extra fields included).
ANSWERS = {
    "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9615, "tech": 0.0385},
             "confidence": 0.7648, "answer_confidence": 0.9615, "action": {"act_probability": 1.0}},
    "urgent": {"type": "noul", "noul": 0.8927, "confidence": 0.8927, "answer_confidence": 0.8927,
               "action": {"act_probability": 1.0}},
    "anger": {"type": "score", "score": 1.4875, "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
              "probabilities": {"0": 0.0099, "1": 0.4927, "2": 0.4974}, "confidence": 0.3248,
              "answer_confidence": 0.4974, "action": {"act_probability": 1.0}},
    "topic": {"type": "choice", "choice": "finance", "probabilities": {"sport": 0.035, "finance": 0.965},
              "confidence": 0.7813, "answer_confidence": 0.965, "action": {"act_probability": 1.0}},
}


class FakeAgent:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return {"model": "laya-rl-agent", "answers": {n: self.answers[n] for n in questions},
                "usage": {"input_tokens": 171, "output_tokens": 0, "truncated": False}}


@pytest.fixture
def fake_laya(monkeypatch):
    module = types.SimpleNamespace(loads=[], agent=FakeAgent(ANSWERS))

    def load(repo, **kw):
        module.loads.append((repo, kw))
        return module.agent
    module.load = load
    monkeypatch.setitem(sys.modules, "laya", module)
    return module


def test_answers_come_back_typed_with_laya_s_own_numbers(fake_laya):
    client = TypeSafeClient(model="laya")
    r = client.system_one(state={"ticket": "charged twice"}, questions=Q)
    assert client.model == "laya" and r.model == "laya" and r.usage.input_tokens == 171
    assert r.choices["team"] == ChoiceAnswer("billing", {"billing": 0.9615, "tech": 0.0385}, 0.7648)
    assert r.nouls["urgent"] == NoulAnswer(0.8927)
    assert r.scores["anger"] == ScoreAnswer(1.4875, {"0": "calm", "1": "annoyed", "2": "furious"},
                                            {"0": 0.0099, "1": 0.4927, "2": 0.4974}, 0.3248)
    assert r.choices["topic"].choice == "finance"
    assert "answer_confidence" not in r.as_dict()["answers"]["team"]


def test_the_questions_reach_laya_in_the_system_one_body(fake_laya):
    TypeSafeClient(model="laya").system_one(state={"ticket": "x"}, questions=Q)
    state, sent = fake_laya.agent.calls[0]
    assert state == {"ticket": "x"}  # the state as given: Laya serialises it itself
    assert sent["team"] == {"type": "choice", "instructions": "Which team?",
                            "criteria": {"billing": "payments, invoices", "tech": "bugs, outages"}}
    assert sent["urgent"]["criteria"] == {"true": "needs action today", "false": "can wait"}
    assert sent["anger"] == {"type": "score", "instructions": "How angry?",
                             "criteria": ["calm", "annoyed", "furious"]}


def test_checkpoints_and_lazy_loading(fake_laya):
    client = TypeSafeClient(model="laya:multilingual")
    assert fake_laya.loads == []  # constructing loads nothing
    client.system_one(state="s", questions={"topic": Q["topic"]})
    client.system_one(state="s", questions={"topic": Q["topic"]})
    assert fake_laya.loads == [("convaiinnovations/laya", {"subfolder": "multilingual"})]
    TypeSafeClient(model="laya:./my-checkpoint").system_one(state="s", questions={"topic": Q["topic"]})
    assert fake_laya.loads[-1] == ("./my-checkpoint", {})
    with pytest.raises(ValueError, match="Unknown Laya checkpoint"):
        TypeSafeClient(model="laya:frenhc")


def test_what_does_not_apply_is_refused(fake_laya):
    with pytest.raises(ValueError, match="not on backend 'mlx'"):
        TypeSafeClient(model="laya", backend="mlx")
    client = TypeSafeClient(model="laya")
    with pytest.raises(ValueError, match="own runtime"):
        client.autotune(Context(name="t"), {"topic": Q["topic"]}, [("a", {"topic": "sport"}), ("b", {"topic": "finance"})])
    with pytest.raises(ValueError, match="create another one"):
        client.system_one(state="s", questions=Q, model="minicpm5-2b")
    with pytest.raises(ValueError, match="at least one question"):
        client.system_one(state="s", questions={})


def test_a_wrong_answer_type_is_an_error(fake_laya):
    fake_laya.agent.answers = {**ANSWERS, "topic": ANSWERS["urgent"]}
    with pytest.raises(RuntimeError, match="wrong type"):
        TypeSafeClient(model="laya").system_one(state="s", questions={"topic": Q["topic"]})


def test_missing_package_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", None)  # import laya -> ImportError
    with pytest.raises(ImportError, match=r"jul\[laya\]"):
        TypeSafeClient(model="laya").system_one(state="s", questions={"topic": Q["topic"]})


def test_laya_as_an_escalation_tier(fake_laya):
    class Unsure:
        def system_one(self, state, questions, **kw):
            from jul.types import SystemOneResponse, Usage
            return SystemOneResponse({n: ChoiceAnswer("tech", {"tech": 0.5}, 0.5) for n in questions},
                                     "local", Usage(), "r")
    r = Escalation([("local", Unsure()), ("laya", TypeSafeClient(model="laya"))]).system_one(
        "s", {"team": Q["team"]}, method="vector")
    assert r.choices["team"].choice == "billing" and r.escalation["team"]["tier"] == "laya"


def test_bench_reads_laya_with_its_default_only(fake_laya):
    from jul_cli.bench import readings
    methods, features, skipped = readings(TypeSafeClient(model="laya"), [None, "letters"], ["vector"])
    assert methods == [None] and features == []
    assert set(skipped) == {"zero-shot:letters", "autotune:vector"}


@pytest.mark.slow
def test_real_laya():
    pytest.importorskip("laya")
    r = TypeSafeClient(model="laya").system_one(state="I was charged twice, refund me now!", questions=Q)
    assert r.choices["team"].choice == "billing"
    assert set(r.answers) == set(Q) and 0 <= r.nouls["urgent"].noul <= 1
    assert abs(sum(r.scores["anger"].probabilities.values()) - 1) < 1e-3


def test_cli_preflight_asks_for_the_laya_package(monkeypatch):
    import importlib.util
    from jul_cli.setup import require_setup
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "laya" else real(n, *a))
    with pytest.raises(SystemExit, match="jul setup --model laya:multilingual"):
        require_setup("laya:multilingual", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: object() if n == "laya" else real(n, *a))
    require_setup("laya", None)  # nothing else to check: no backend, no preset
