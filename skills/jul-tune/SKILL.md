---
name: jul-tune
license: Apache-2.0
description: >
  Adapt JuL decisions to the user's own data: measure zero-shot accuracy on labeled
  examples, add a Context (unlabeled examples), train a head with `autotune`, pick
  `features` and `formulations`, or write synthetic labels with `jul synth`. Use when
  a JuL question is off on real data, when the user has (or can produce) labeled
  examples, or before deploying a fixed decision.
---

# Tune JuL on your data

Three tools, cheapest first: a `Context` (unlabeled examples), `autotune` (labeled
examples), `jul synth` (too few labels). None of them touches the model's weights, and
tuning takes seconds. Source of truth:
[tuning.md](https://raw.githubusercontent.com/usejul/jul/main/docs/tuning.md) and
[cli.md](https://raw.githubusercontent.com/usejul/jul/main/docs/cli.md) for file formats;
`jul autotune --help` for the installed flags.

## 0. Fix the questions first

Most misses are a wording problem, and a head trained on a bad question is stuck with it
(it is keyed on the exact instructions and option descriptions: change a word and it is
gone). Look at 20 errors before tuning: vague descriptions, overlapping options, a missing
no-match option, or a relation question the vector reading cannot answer (see the `jul`
skill). Fix those, then measure again.

## 1. Measure zero-shot, on held-out data

```bash
jul run questions.yaml --input dev.jsonl --output answers.jsonl     # dev.jsonl: {"state", "id"}
```

Compute accuracy per question in code against the labels. Keep a test split that no step
below ever sees, and report numbers on it only. With fewer than ~100 examples, say how wide
the error bars are (±5 to ±10 points).

## 2. A Context: ten to fifty real texts, no labels

```python
from jul import TypeSafeClient, Context
tickets = Context(examples=real_texts[:50])     # 10 captures most of the gain, 50 is best, 200 adds nothing
client = TypeSafeClient(context=tickets)
```

```bash
jul context create tickets --examples sample.txt       # reused by name: --context tickets / context="tickets"
```

- Their mean vector becomes the task's center. It helps most on topic-like tasks with a style
  of their own; it slightly hurt fine-grained intents. Measure with and without.
- Fewer than 10 examples is noise and can be worse than nothing (JuL warns).
- `description` is off by default because it measured harmful on average. Do not switch on
  `use_description=True` without a measured gain.
- Contexts live under `~/.jul/contexts/<name>/` (`JUL_HOME` moves it).

## 3. autotune: labeled examples

```python
reports = client.autotune("tickets", questions, labeled, features="hybrid")
# labeled: [(state, {"team": "billing", "is_bug": False, "frustration": 1}), ...]
print(reports["team"])       # zero-shot vs head, ACTIVE or not, and why
client.system_one(state, questions, context="tickets")   # the head is used automatically
```

```bash
jul autotune tickets --questions questions.yaml --labeled labeled.jsonl --features hybrid
```

- Labels: option key for a Choice, `true`/`false` for a Noul, level index (0 = lowest) for a Score.
- Below `max(20, 3 × options)` labels per question it only calibrates (temperature + per-option
  bias). Above, it trains a head and keeps it only if it beats zero-shot in cross-validation.
  An inactive head is a result, not a failure: report it.
- `features`: `vector` (default), `lexical` (TF-IDF, no model: the baseline a head must beat),
  `hybrid` (both, usually best; needs `pip install "jul[tune]"`).
- `formulations` (Python API) picks the prompts per question: `one_word` is one pass per message
  for every question (fastest), `question_options` helps with few options, `question` scales to many.
- A head is tied to the model, backend, instructions and options it was trained on. Change any of
  them and re-run `autotune`.
- Decision models (`minicpm5-2b-decision`) cannot be autotuned.
- With a cross model, a Noul or Score head must beat the cross model to replace it.

## 4. Too few labels: jul synth

```bash
jul synth questions.yaml --seeds sample.jsonl --per-option 30 --output synth.jsonl --writer <mlx-lm instruct repo, 7B+>
jul autotune tickets --questions questions.yaml --labeled synth.jsonl
```

Needs MLX and a generative model. Synthetic texts are cleaner than real ones: always judge a
head trained on them on real labeled examples, never on synthetic ones.

## Report

For each question: labels used, zero-shot and tuned accuracy on the held-out split, whether the
head is ACTIVE, the model and backend. Compare with the TF-IDF baseline (`features="lexical"`).
Never claim a gain measured on the examples the head was trained on.
