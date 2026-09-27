---
name: jul
license: Apache-2.0
description: >
  Build local typed decisions with JuL (Juste un LLM): a Python library that answers
  Choice, Noul (yes/no) and Score questions about a text with a probability, on the
  user's own machine, with the same API as TypeSafe's Jev SDK. Use when a feature needs
  a classifier, a router, a yes/no check, a grade or a ranking on text and should run
  without a hosted API, when an LLM prompt-and-parse step could become a typed decision,
  or when porting code written for `typesafe_sdk` to run on device.
---

# Build with JuL

JuL answers typed questions about a piece of text: pick an option (`Choice`), give the
probability of yes (`Noul`), place it on a scale (`Score`). The model never writes: the
answer is read from its hidden states, compared with the vectors of the option
*descriptions*. It runs locally (MLX, PyTorch or ONNX), no API key, no per-call bill.
Code owns the workflow; JuL supplies the judgments.

## Read the repo docs

The docs in the repo are the source of truth; this skill gives direction. Fetch the page you
need (raw Markdown) rather than guessing a flag or a field:

| Task | Read |
| --- | --- |
| Install, pick a backend, devices, batching | [installation.md](https://raw.githubusercontent.com/usejul/jul/main/docs/installation.md) |
| Choose or add a model, cross and decision models | [models.md](https://raw.githubusercontent.com/usejul/jul/main/docs/models.md) |
| `Context`, `autotune`, hybrid heads, `jul synth` | [tuning.md](https://raw.githubusercontent.com/usejul/jul/main/docs/tuning.md) (or the `jul-tune` skill) |
| Every CLI command and file format | [cli.md](https://raw.githubusercontent.com/usejul/jul/main/docs/cli.md) |
| `jul pack`, `jul serve`, AWS Lambda | the `jul-deploy` skill |
| What is measured, and what is not | [benchmarks.md](https://raw.githubusercontent.com/usejul/jul/main/docs/benchmarks.md) |

When the package is installed, the installed code wins over any doc: `python -c "import jul,
inspect; print(inspect.signature(jul.TypeSafeClient.system_one))"`, `jul <command> --help`.
Never invent a field: the shapes are fixed by the Jev SDK and listed below.

## Get it running

```bash
python -m venv .venv && . .venv/bin/activate      # Python >= 3.10, never sudo
pip install jul
jul setup      # picks MLX on Apple Silicon, else PyTorch; downloads the default model (2.6 GB); one timed decision
jul ask choice "Which team should handle this ticket?" \
    -o billing:"payments, invoices" -o technical:"bugs, errors" --state "I was charged twice"
```

If `jul setup` fails, show the error before trying something else. The first call loads the model
(seconds); measure latency on the calls after it.

## The API

```python
# from typesafe_sdk import TypeSafeClient, Choice, Noul, Score   # Jev: only this line changes
from jul import TypeSafeClient, Choice, Noul, Score, NoulCriteria

client = TypeSafeClient()          # default wemm-4b-4bit; model="minicpm5-2b" is the fast one
response = client.system_one(
    state={"ticket": ticket_text, "customer": {"plan": plan}},
    questions={
        "team": Choice(instructions="Which team should handle `ticket`?",
                       criteria={"billing": "payments, invoices, refunds",
                                 "technical": "bugs, errors, crashes",
                                 "other": "anything else: thanks, spam, off-topic"}),
        "is_bug": Noul(instructions="Does `ticket` report a software bug?",
                       criteria=NoulCriteria(true="software misbehaves", false="no defect is reported")),
        "frustration": Score(instructions="How frustrated is the customer?",
                             criteria=["Calm", "Frustrated but civil", "Very angry"]),
    },
)
response.choices["team"].choice          # "billing"
response.choices["team"].probabilities   # {"billing": 0.88, ...}; .confidence = top probability
response.nouls["is_bug"].noul            # probability of yes, in [0, 1]
response.scores["frustration"].score     # expected level, 0 .. len(criteria) - 1
response.scores["frustration"].probabilities, response.scores["frustration"].confidence
response.as_dict()                       # the Jev HTTP response shape, for json.dumps
```

- `state` is a string or anything JSON-serialisable (it is serialised with `json.dumps`).
- `AsyncTypeSafeClient` has the same API, awaitable. Remote-only arguments (`api_key`, `retry`,
  `base_url`, `response_model`…) are accepted and ignored, so Jev code runs unchanged.
- One model is held in memory per client; reuse the client, do not build one per call.
- From the shell: `jul ask` for one question, `jul run questions.yaml --input in.jsonl --output
  out.jsonl` for a file. Other languages: `jul serve` (see `jul-deploy`).

## Write questions the model can read

JuL compares the text with each option's **description**. The key is only what you get back.
This is the single biggest lever on accuracy, and the thing a human should review.

- Describe every option the way a colleague would understand it: `"billing": "payments, invoices,
  refunds"`, never a bare `"billing"` or a code like `"T2"`.
- Make options mutually exclusive and cover the space. Add a no-match option (`"other": "none of
  these: ..."`) when nothing may fit: a Choice always picks one.
- One narrow judgment per question. Several labels that may hold at once are several `Noul`s, not
  one `Choice`.
- `Score` levels go lowest first and each must describe a concrete situation on its own.
- Put every question and threshold in one file (`questions.yaml` in the format of
  [cli.md](https://raw.githubusercontent.com/usejul/jul/main/docs/cli.md), or one Python module) so they
  are easy to review; `jul run`, `jul autotune` and `jul pack` all read that file.
- Ask the questions about one state in a single `system_one` call: the state is read once and
  shared. Chain a second call only when an answer decides what to ask next.

## Know what it cannot do

- The default vector reading encodes the text and each option **apart**. A question about how two
  parts relate (is A a paraphrase of B, does the request name a place but not a date) is answered
  near chance by any embedding model. Use a model with a cross model (`jul-decision-e5-small`,
  `jul-decision-wemm-4b-4bit`, see models.md), split the judgment, or keep it in code.
- `Noul` is the weakest type on the vector reading (yes/no 0.762 on Kev's typed decisions, against
  0.841 with the cross model of `jul-decision-wemm-4b-4bit`). Check yes/no answers on real
  examples before thresholding them.
- It reads meaning, not exact facts: keep known rules, lookups, regexes and arithmetic in code, and
  ask JuL only where code would need to understand the text.
- Settings were fitted on English. Other languages and far-off domains (logs, chemistry) are not
  measured: measure before trusting.
- Long texts are cut to the model's positions. Select the relevant span in code first.
- A decision model (`minicpm5-2b-decision`) cannot be `autotune`d.

## Use the probabilities

- Calibrated (ECE 0.084 on the default model), so the pattern is: automate above a threshold, send
  the rest to a human or a larger model. Choose the threshold on labeled data from the real
  domain, from the cost of each kind of error, not from a round number.
- When only the best option matters, take `choice`; a threshold on `confidence` is for deciding
  whether to act at all.
- A `noul` near 0.5 means "undecided", not "medium".
- Keep raw answers; compose policy in code (weights, "any serious violation" rules, rankings), so
  changing a weight does not re-run the model.

## Verify before claiming it works

Typed output guarantees the interface, not the truth. Before proposing JuL in a codebase:

1. Collect 30 to 100 real texts, label them yourself or with the user.
2. Run them (`jul run`), compute accuracy per question and look at the errors: bad description,
   missing option, relation question, or model limit.
3. Compare with the cheap baseline: a TF-IDF + linear SVM beats every LLM on sorting by topic when
   there are 1000 labels (README, "The baseline worth remembering"). Say so if it wins.
4. If zero-shot is short, go to the `jul-tune` skill: a `Context`, then `autotune`.

Report the numbers you measured, on which model and backend, and how many examples. Never quote a
benchmark figure as the accuracy the user will get.
