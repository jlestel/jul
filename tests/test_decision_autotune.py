"""`autotune` on a decision model: heads trained on its vector fallback, judged against what answers without one.

End to end on a tiny random Qwen3 made into a decision model (a decision.json and a random pointer head), on
torch: the pointer path, the fallback swap and the heads all run for real. Only the tokenizer is downloaded.
"""

import importlib.util
import json
import os

import numpy as np
import pytest

from jul import Choice, Context, Noul, Score, TypeSafeClient
from jul.presets import ONE_WORD, QUESTION_OPTIONS, Preset, repo_fields, save_preset

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")

SPEC = {
    "method": "pointer",
    "tokens": {"state": "<|fim_prefix|>", "question": "<|fim_middle|>", "option_open": "<|box_start|>",
               "option_close": "<|box_end|>", "decide": "<|fim_suffix|>"},
    "layout": {"prefix": ["state", "{state}"], "branch": ["question", "{instructions}", "{options}", "decide"],
               "option": ["option_open", "{option}", "option_close"]},
    "option_text": {"with_description": "{name}: {description}", "without_description": "{name}"},
    "noul_options": ["no", "yes"],
    "add_special_tokens": False,
    "readout": {"layer": "last_normed", "question_token": "decide", "option_token": "option_close"},
    "head": {"file": "pointer_head.npz", "query": "q", "key": "k", "dim": 16, "temperature": 1.0},
    "limits": {"max_state_tokens": 128, "max_branch_tokens": 256},
}
ROUTING = {"above_options": 0, "tau": 0.05, "center": "options",
           "formulations": [{"name": "one_word", "template": ONE_WORD, "layer": 3},
                            {"name": "question_options", "template": QUESTION_OPTIONS, "layer": 2}]}
TEAM = Choice(instructions="Which team?", criteria={"billing": "payments, refunds", "tech": "bugs, crashes"})
WORDS = {"billing": ["refund", "invoice", "charged", "payment"], "tech": ["crash", "error", "bug", "freeze"]}


@pytest.fixture(scope="module")
def decision_dir(tmp_path_factory):
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    root = tmp_path_factory.mktemp("tiny-decision")
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("JUL_TEST_TOKENIZER", "Qwen/Qwen3-0.6B"))
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
    torch.manual_seed(0)
    Qwen3ForCausalLM(config).save_pretrained(root)
    tokenizer.save_pretrained(root)
    rng = np.random.default_rng(0)
    np.savez(root / "pointer_head.npz", q_weight=rng.normal(0, 0.1, (16, 64)).astype(np.float32),
             q_bias=np.zeros(16, np.float32), k_weight=rng.normal(0, 0.1, (16, 64)).astype(np.float32),
             k_bias=np.zeros(16, np.float32))
    (root / "decision.json").write_text(json.dumps(SPEC))
    os.environ.setdefault("JUL_DEVICE", "cpu")
    return root


def _client(decision_dir, tmp_path, monkeypatch, routing=ROUTING):
    import jul.presets
    monkeypatch.setattr(jul.presets, "PRESET_HOME", tmp_path / "presets")
    save_preset(Preset(name="tiny-decision", **repo_fields("torch", str(decision_dir)), backend="torch",
                       formulations=(), tau=1.0, latency_ms="?", quality="test", method="pointer", routing=routing),
                tmp_path / "presets")
    return TypeSafeClient(model="tiny-decision", backend="torch", context_home=tmp_path / "contexts")


def _labeled(n=40, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        team = ("billing", "tech")[i % 2]
        out.append((f"{rng.choice(WORDS[team])} please, ticket {i}", {"team": team, "urgent": bool(i % 3 == 0)}))
    return out


def test_a_head_on_a_decision_model_answers_through_its_vector_fallback(decision_dir, tmp_path, monkeypatch):
    client = _client(decision_dir, tmp_path, monkeypatch)
    labeled = _labeled()
    ctx = Context(name="t")
    report = client.autotune(ctx, {"team": TEAM}, labeled, save=False)["team"]
    assert report.n_examples == 40 and "pointer head" in report.reason
    engine = client._engine_for(None)
    # zero-shot is the pointer head's answers, one pass per state, as system_one reads them
    pointer = [engine.pointer.logits(s, [("choice", TEAM.instructions, [o for o in _options(TEAM)])])[0][0]
               for s, _ in labeled]
    keys = [o.key for o in _options(TEAM)]
    want = np.mean([keys[int(np.argmax(z))] == a["team"] for z, (_, a) in zip(pointer, labeled)])
    assert report.zero_shot_accuracy == pytest.approx(want)
    if not report.activated:
        pytest.skip(f"the safety net refused the head on random weights: {report.reason}")

    # with the head, the question is read through the fallback + head; the preset is restored after
    state = labeled[0][0]
    tuned = client.system_one(state=state, questions={"team": TEAM}, context=ctx).choices["team"]
    assert engine.preset.method == "pointer"
    from jul import tuning
    compiled = engine.compile("choice", TEAM.instructions, _options(TEAM), ctx, preset=client._fallback(engine))
    _, features, _ = engine.read(compiled, state)
    expected = tuning.apply(ctx.heads[client._digest("choice", TEAM, _options(TEAM))], features, state)
    assert np.allclose([tuned.probabilities[k] for k in keys], expected, atol=1e-4)
    # a question without a head keeps the pointer head
    urgent = Noul(instructions="Urgent?")
    plain = client.system_one(state=state, questions={"urgent": urgent}, context=ctx).nouls["urgent"].noul
    z = engine.pointer.logits(state, [("noul", urgent.instructions, _options(urgent))])[0][0]
    p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    assert plain == pytest.approx(float(p[[o.key for o in _options(urgent)].index("true")]), abs=1e-4)
    client.close()


def test_a_routed_type_is_judged_against_the_fallback_it_is_read_with(decision_dir, tmp_path, monkeypatch):
    client = _client(decision_dir, tmp_path, monkeypatch, routing={**ROUTING, "types": ["score"]})
    level = Score(instructions="How urgent?", criteria=["low", "medium", "high"])
    labeled = [(s, {"level": i % 3}) for i, (s, _) in enumerate(_labeled(45))]
    engine = client._engine_for(None)
    calls = []
    real = engine.pointer.logits
    engine.pointer.logits = lambda *a: calls.append(a) or real(*a)
    report = client.autotune(Context(name="t"), {"level": level}, labeled, save=False)["level"]
    assert calls == [] and "pointer" not in report.reason          # never read by the pointer head
    compiled = engine.compile("score", level.instructions, _options(level), preset=client._fallback(engine))
    scores, _ = engine.read_many(compiled, [s for s, _ in labeled])
    y = np.array([i % 3 for i in range(45)])
    assert report.zero_shot_accuracy == pytest.approx(float((scores.argmax(1) == y).mean()))
    client.close()


def test_the_shared_engine_preset_is_never_swapped(decision_dir, tmp_path, monkeypatch):
    """Concurrent calls share one engine (AsyncTypeSafeClient runs them in threads): reading a decision model's
    fallback must pass the preset along, never assign `engine.preset`, or another call reads with the wrong one."""
    client = _client(decision_dir, tmp_path, monkeypatch, routing={**ROUTING, "types": ["score"]})
    engine = client._engine_for(None)

    class Frozen(type(engine)):
        def __setattr__(self, name, value):
            if name == "preset":
                raise AssertionError("engine.preset was reassigned")
            super().__setattr__(name, value)

    engine.__class__ = Frozen
    monkeypatch.setattr(client, "_engine_for", lambda model: engine)
    level = Score(instructions="How urgent?", criteria=["low", "medium", "high"])
    ctx = Context(name="t")
    client.autotune(ctx, {"team": TEAM}, _labeled(), save=False)                     # fallback features
    client.system_one(state="refund please", questions={"team": TEAM, "level": level,  # routed + pointer (+ head)
                                                          "urgent": Noul("Urgent?")}, context=ctx)
    client.close()


def test_a_decision_model_without_a_fallback_says_why_it_cannot_be_tuned(decision_dir, tmp_path, monkeypatch):
    client = _client(decision_dir, tmp_path, monkeypatch, routing=None)
    with pytest.raises(ValueError, match="no vector reading fitted"):
        client.autotune(Context(name="t"), {"team": TEAM}, _labeled(), save=False)
    client.close()


def _options(question):
    from jul.types import options_of
    return options_of(question)
