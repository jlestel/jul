---
name: jul-deploy
license: Apache-2.0
description: >
  Ship JuL decisions: serve them over HTTP with `jul serve` (the Jev protocol, for any
  language), freeze fixed questions into a bundle with `jul pack`, or run a tuned
  micro model inside an AWS Lambda. Use when JuL must be called from a non-Python
  service, a Jev client should point at a local server, or a fixed decision must run
  in production, a container or a Lambda.
---

# Deploy JuL

Pick by who calls and whether the questions change:

| Situation | Use |
| --- | --- |
| Python service, questions vary per call | `TypeSafeClient` in-process (the `jul` skill) |
| Another language, or existing Jev client code | `jul serve`, change only the base URL |
| Fixed questions, production, container | `jul pack` a bundle, load it with `Bundle` |
| One fixed decision per event, no server | a bundle on `jul-decision-e5-small` in AWS Lambda |

Source of truth: [serve.md](https://raw.githubusercontent.com/usejul/jul/main/docs/serve.md),
[deployment.md](https://raw.githubusercontent.com/usejul/jul/main/docs/deployment.md),
[aws-lambda.md](https://raw.githubusercontent.com/usejul/jul/main/docs/aws-lambda.md). Read the
relevant page before writing deployment code; `jul serve --help`, `jul pack --help` for flags.

## jul serve

```bash
jul serve                                  # 127.0.0.1:8577, default model, warmed up before it listens
curl -s http://127.0.0.1:8577/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest", "state": "I was charged twice",
  "questions": {"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                         "criteria": {"billing": "payments, invoices", "technical": "bugs, errors"}}}}'
```

- Routes: `POST /v1/systemone`, `GET /v1/models`, `GET /health`. The response is the Jev shape
  plus a `jul` object (latency). Extra request fields: `context`, `method`, `route_above`.
- `jev-*` model names mean the server's own model.
- One model in memory, calls run one at a time: size throughput accordingly, or run several.
- It binds to loopback. Before `--host 0.0.0.0`, set `JUL_API_KEY` (or `--api-key`); never expose
  it without a key. The request `state` is never logged.

## jul pack and Bundle

Everything that does not depend on the message is computed once: prompts, prefix caches,
option vectors, centers, a context's heads and calibrations. A bundle holds no weights; it
names its model.

```bash
jul pack bundle/ --questions questions.yaml --context tickets --model <preset> --backend <backend>
```

```python
from jul import Bundle
bundle = Bundle.load("bundle/")
bundle.system_one("I was charged twice").answers["team"].choice
bundle.system_one_batch(messages)
```

- Pack on the backend you deploy on: vectors differ between MLX (4-bit) and PyTorch (bf16).
- Tune (`jul-tune`) before packing: the bundle carries the context's heads as they are.
- The pack output says which models the bundle needs (`vector`, `cross`, or both). Ship only those.

## AWS Lambda

Follow [aws-lambda.md](https://raw.githubusercontent.com/usejul/jul/main/docs/aws-lambda.md) step by step:
`pip install "jul[onnx,tune,yaml,calibrate]"`, `jul models add jul-decision-e5-small --repo
usejul/jul-decision-e5-small-onnx --backend onnx`, autotune with `--features hybrid`, pack with
`--backend onnx`, build the zip for arm64 / Python 3.12, deploy with the AWS CLI or CDK.

- It is for one fixed decision per event, not a general endpoint. Zero-shot the micro model is a
  starting point (0.557 on the Jev bench); tuned on a few hundred real labels it is the product.
- Measured: 17 ms per message and $0.59 per million at 1,769 MB, 2.4 s cold start, 225 MB package.
  These are that function's numbers; measure yours.
- Long texts are cut to 512 positions (the cross model reads 256): select the span in code.
- Heads trained on another domain transfer badly: tune on the real inbox.

## Before calling it done

Run the deployed path end to end on a few real inputs (curl the server, invoke the function,
load the bundle) and compare the answers with the same questions run in-process. Report the
latency you measured, where, and with which model.
