# Quickstart

Ten minutes, from nothing to a file of texts answered on your machine. You need Python 3.10 or later and
about 3 GB of disk for the default model.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install jul
jul setup
```

`jul setup` does four things and says which: it picks the backend (MLX on an Apple Silicon Mac, PyTorch
anywhere else), installs it if missing, downloads the default model once from the Hugging Face Hub, and runs
one real decision with its latency.

```text
jul setup: jul-decision-wemm-4b on mlx
  backend  mlx (installed now)
  preset   jul-decision-wemm-4b (built-in): tau 0.0553, center options
  weights  usejul/WeMM-Embedding-4B-mlx-4bit -> ~/.cache/huggingface/...
  check    'I was charged twice' -> billing (0.88), 55 ms per decision (first call 4.1s with the load)
```

Nothing else leaves your machine after that download: no account, no key, no telemetry.

## 2. A first answer, from the shell

```bash
jul ask choice "Which team should handle this ticket?" \
    -o billing:"payments, invoices" -o technical:"bugs, errors" -o sales:"pricing, plans" \
    --state "I was charged twice"
```

```json
{"request_id": "…", "model": "jul-decision-wemm-4b",
 "usage": {"input_tokens": 24, "output_tokens": 0, "total_tokens": 24},
 "answers": {"choice": {"type": "choice", "choice": "billing",
                        "probabilities": {"billing": 0.88, "technical": 0.11, "sales": 0.01},
                        "confidence": 0.88}},
 "latency_ms": 55.3}
```

Read it this way:

- `choice` is the option with the highest probability, `confidence` is that probability;
- `probabilities` sum to 1 over **your** options: the model can only pick one of them;
- `output_tokens` is always 0: nothing is generated, the answer is read from the model's hidden state
  (see [Concepts](concepts.md#reading)).

The two other question types:

```bash
jul ask noul "Does the message report a software bug?" --state "The app crashes on export"
jul ask score "How frustrated is the customer?" -o "Calm" -o "Annoyed" -o "Very angry" \
    --state "Third time I write. Nobody answers."
```

A `noul` answers the probability of *yes* (`"noul": 0.93`). A `score` answers the expected level from 0 to the
number of levels minus one (`"score": 1.8`, close to "Very angry"), with a probability per level.

## 3. The same, in Python

```python
from jul import TypeSafeClient, Choice, Noul, Score

client = TypeSafeClient()              # the default model; TypeSafeClient(model="minicpm5-2b") for another

response = client.system_one(
    state={"ticket": "I was charged twice for my subscription this month."},
    questions={
        "team": Choice(instructions="Which team should handle this ticket?",
                       criteria={"billing": "payments, invoices, refunds",
                                 "technical": "bugs, errors, crashes",
                                 "sales": "pricing, plans, demos"}),
        "is_bug": Noul(instructions="Does the message report a software bug?"),
        "frustration": Score(instructions="How frustrated is the customer?",
                             criteria=["Calm", "Frustrated but civil", "Very angry"]),
    },
)

response.choices["team"].choice           # "billing"
response.nouls["is_bug"].noul             # 0.12
response.scores["frustration"].score      # 1.15
response.as_dict()                        # the JSON above
```

- `state` is the text to judge: a string, or any object (a dict is turned into JSON for the model).
- `questions` is a dict: you name each question, and its answer comes back under the same name.
- Every question of a call is about the same state. One call per text.
- The model loads on the first call (a few seconds) and stays in memory; the next calls take tens of
  milliseconds.

## 4. A whole file

Put the questions in a file once:

```yaml
# questions.yaml   (pip install "jul[yaml]", or write the same thing as JSON)
team:
  type: choice
  instructions: Which team should handle this ticket?
  criteria:
    billing: payments, invoices, refunds
    technical: bugs, errors, crashes
    sales: pricing, plans, demos
is_bug:
  type: noul
  instructions: Does the message report a software bug?
```

and the texts one per line, as a string or `{"state": …, "id": …}`:

```json
{"id": "T-1", "state": "I was charged twice"}
{"id": "T-2", "state": "The export button does nothing"}
"Do you have a yearly plan?"
```

```bash
jul run questions.yaml --input tickets.jsonl --output answers.jsonl
```

`answers.jsonl` has one line per text, in the shape above, with the `id` copied over.

## 5. Where to go next

| You want to… | Next |
| --- | --- |
| know why it answered that, and what the knobs are | [Concepts](concepts.md) |
| get better answers without labels | [Writing good questions](questions.md), then a [Context](tuning.md#context--what-the-data-looks-like) |
| get better answers with labels | [`autotune`](tuning.md#autotune--when-it-is-off-key) |
| use a smaller, faster or other model | [The hub](hub.md) |
| call it from JavaScript, Go, a shell script | [`jul serve`](serve.md) |
| ship it to a Lambda | [AWS Lambda](aws-lambda.md) |

**Already on Jev?** Change the import and run your code unchanged:

```python
# from typesafe_sdk import TypeSafeClient, Choice, Noul, Score
from jul import TypeSafeClient, Choice, Noul, Score
```

Arguments that only mean something for a remote API (`api_key`, `base_url`, `timeout`, `max_retries`, …) are
accepted and ignored.
