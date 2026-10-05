# Command line

```bash
jul ask choice "Which team should handle this ticket?" \
    -o billing:"payments, invoices" -o technical:"bugs, errors" \
    --state "I was charged twice" --model minicpm5-2b

jul run questions.yaml --input tickets.jsonl --output answers.jsonl --context tickets
jul context create tickets --description "Support tickets of an online bank" --examples sample.txt
jul context list                  # the saved contexts
jul context show tickets          # its description, example count, tuned and calibrated questions
jul context delete tickets
jul synth questions.yaml --seeds sample.jsonl --per-option 30 --output synth.jsonl
jul autotune tickets --questions questions.yaml --labeled labeled.jsonl --features hybrid
jul pack bundle/ --questions questions.yaml --context tickets --backend onnx
jul models
jul models add minicpm5-2b-decision --repo usejul/minicpm5-2b-decision-mlx-4bit   # a decision model
jul models add my-model --repo org/Some-Instruct-3B                                 # fits a preset
jul setup --model minicpm5-2b-decision    # backend, weights and one timed decision
jul serve --model minicpm5-2b --port 8577   # the Jev HTTP protocol on 127.0.0.1, see serve.md
```

## Bench on your own data

`jul bench` answers the question you will ask in production with every model you name, on rows you have
labeled, and says which one to use. One row per answer, JSONL or CSV, carrying its own question so a file can
mix several:

```json
{"type": "choice", "question": "Which team should handle this ticket?",
 "options": {"billing": "payments, invoices", "technical": "bugs, errors"},
 "state": "I was charged twice", "answer": "billing"}
```

In CSV the options are `key:description|key:description` (or `key|key`). `type` is `choice` by default; a
`noul` answers `true`/`false` and needs no options; a `score` lists its levels lowest first and answers the
level (its index or its text).

```bash
jul bench test.jsonl --models jul-decision-e5-small@onnx,minicpm5-2b@torch      # zero-shot
jul bench test.jsonl --train train.jsonl --models fast,accurate -O results.json   # + autotune
jul bench test.jsonl --train train.jsonl --models minicpm5-2b@torch --method auto --features auto
jul bench test.csv --json > results.json
```

- **Per question**: accuracy, its 95% Wilson interval, p50/p95 latency (and the mean error for a score).
- **The pick**: the fastest model whose interval still reaches the best accuracy. When several are within it,
  the report says they cannot be told apart on that many rows: that is the honest answer below ~50 rows.
- **Readings**: by default each model answers the way it does in production. `--method vector,letters,cross`
  (or `auto`) measures each zero-shot reading, `--features vector,lexical,hybrid` (or `auto`) each autotune
  head, the same names as `jul ask --method` and `jul autotune --features`. Every reading gets its own row and
  the pick is chosen among all of them, so the answer is "this model, read this way". A reading a model cannot
  take (letters on an embeddings API, anything but its pointer head on a decision model, cross without a cross
  model) is shown `n/a` with the reason. More readings cost more time, and the best of many on the same rows is
  slightly optimistic: the report says so.
- **Autotune** only with `--train`, a separate file you labeled: never a split of the test file. Each model is
  tuned in a throwaway context (nothing saved in `~/.jul`); a head that does not beat zero-shot on its own
  folds is shown with `*` and not counted, and an autotuned row is only picked if it does better on the test
  rows. Decision models (pointer) are measured zero-shot only.
- **Overlap check**, before anything runs: a test text also in train (after folding case, accents,
  punctuation and turning every number into 0, so `van ABC-123 at 11pm` matches `van abc-987 at 10pm`) stops
  the bench; `--drop-overlap` removes those rows from test, `--allow-overlap` runs anyway. Near duplicates
  (character 5-gram similarity above `--near`, 0.8 by default) and duplicates inside test are reported. What
  cannot be checked: a model that saw public data during its pretraining.
- **Remote servers**, next to JuL's models: `typesafe` (Jev, key in `TYPESAFE_API_KEY`), `ollama[:model]`
  (Nimble), `cloudflare:clef` or `cloudflare:clef-flash` (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`), or
  the URL of any `/v1/systemone` server as `URL#model` with its key in `--remote-key-env VAR` (Kev, llama.cpp,
  a hosted Laya, another `jul serve`). Each is asked with your own key, zero-shot only, and its latency
  includes the network. **The test rows are sent to that server.** A target whose key is missing is shown
  failed, before anything is sent, and the others still run.

  ```bash
  jul bench test.jsonl --models accurate,typesafe,cloudflare:clef,https://kev.example#kev-4b \
    --remote-key-env KEV_API_KEY
  ```

## File formats

**Questions** (`questions.yaml`, for `run`, `synth`, `autotune`, `pack`): YAML or JSON, one entry per
question, its name as key. `criteria` is `{key: description}` for a `choice`, the ordered levels for a
`score`, nothing for a `noul`.

```yaml
team:
  type: choice
  instructions: Which team should handle this ticket?
  criteria:
    billing: payments, invoices, refunds
    technical: bugs, errors, crashes
is_bug:
  type: noul
  instructions: Does the message report a software bug?
frustration:
  type: score
  instructions: How frustrated is the customer?
  criteria: [Calm, Frustrated but civil, Very angry]
```

**Labeled examples** (`labeled.jsonl`, for `autotune`): one JSON object per line, the text in `state`
(or `text`) and one answer per question in `answers` — the option key for a `choice`, `true`/`false` for
a `noul`, the index of the level for a `score` (0 = the first, lowest level). The answers may also sit at the top level, named after the questions.

```json
{"state": "I was charged twice for my subscription", "answers": {"team": "billing", "is_bug": false, "frustration": 1}}
{"state": "The app crashes every time I export", "answers": {"team": "technical", "is_bug": true, "frustration": 2}}
{"text": "Do you have a yearly plan?", "team": "billing", "is_bug": false, "frustration": 0}
```

**Seeds** (`sample.jsonl`, for `synth --seeds`): the same lines, with `answers` optional — a seed without
answers only informs the style of the texts written.

```json
{"state": "hi, my card got declined at the station again", "answers": {"team": "billing"}}
{"state": "ur app logged me out 3 times today"}
```

**Examples of a context** (`sample.txt`, for `context create --examples`): one text per line, or a
`.jsonl` with a `text` field per line.

```text
I was charged twice for my subscription
The app crashes every time I export
Do you have a yearly plan?
```

**Input of `run`** (`tickets.jsonl`): one message per line, a string or `{"state": ..., "id": ...}`; the
`id` is copied to the output. **Output** (`answers.jsonl`): one line per message, the shape of a Jev API
response:

```json
{"request_id": "c3b355de-…", "model": "e5-small", "usage": {"input_tokens": 9, "output_tokens": 0, "total_tokens": 9}, "answers": {"team": {"choice": "billing", "probabilities": {"billing": 0.91, "technical": 0.06, "sales": 0.03}, "confidence": 0.91}}, "id": "T-1042"}
```
