# Changelog

Versions come from git tags (see [publishing](docs/publishing.md)): a `v*` tag releases to PyPI, every push to
`main` publishes a dev build to TestPyPI.

## Unreleased

### Added

- **`jul bench` compares JuL with remote servers**: `--models` also takes `typesafe` (Jev), `ollama[:model]`,
  `cloudflare:clef|clef-flash` and any `/v1/systemone` URL as `URL#model` (Kev, a hosted Laya…), each with the
  user's own key (`--remote-key-env` for a URL). Zero-shot only, network latency included, the report says the
  rows were sent there; a missing key fails that target alone, before anything is sent.
- **`jul bench --escalate-to TARGET`** measures each local model as a cascade (itself, then the remote server for
  the answers below `--min-confidence`, as `jul serve --escalate-to`): the accuracy of the pair and the share of
  rows sent to the server, also on the verdict line when the cascade is picked. The server's failures are
  counted on the row, and a server that failed on every escalated row (a rejected key) makes the row `n/a`.

### Fixed

- **EmbeddingGemma read as an encoder.** A model whose config sets `use_bidirectional_attention`
  (`google/embeddinggemma-300m`, a bidirectional `gemma3_text`) is now read like the encoders: mean over the
  text, its declared `query` prompt, no prefix cache. It was read as a decoder (last token, cached prefix):
  0.160 dev accuracy after `jul models add`, 0.655 now (multilingual-e5-small: 0.585, same data). Torch backend.

## 0.4.0 — 2026-10-05

### Added

- **Laya** as a model: `TypeSafeClient(model="laya")`, `--model laya` (also `laya:multilingual`,
  `laya:typed-decisions` or `laya:<hub repo or directory>`), with `pip install "jul[laya]"` or
  `jul setup --model laya`. Laya runs on its own package; JuL hands it the questions and returns its answers
  as JuL's, so `ask`, `run`, `serve`, `bench` and `Escalation` take it. JuL's readings, `autotune` and
  `pack` do not apply to it. See [Models > Laya](docs/models.md#laya).
- **`jul bench`: pick a model on your own data.** `jul bench test.jsonl --train train.jsonl --models
  jul-decision-e5-small@onnx,minicpm5-2b@torch -O results.json` answers every test row with each model and
  reports, per question, accuracy with its 95% interval, latency (p50/p95) and the model to pick: the fastest
  whose interval reaches the best. Each row carries its question (`question, options, state, answer`), so one
  file can mix questions and types. With `--train` (a separate file) each model is also autotuned and measured
  again on the same test rows; train and test are checked for overlap first (exact duplicates after folding
  case, punctuation and digits stop the bench, near duplicates are reported, `--drop-overlap` removes them).
  `--method` / `--features` (or `auto`) compare the zero-shot readings and autotune heads of each model, one
  row each. Tables in the site's amber on a terminal, `--json` or `-O file` for the full results.
- **Embeddings APIs as a backend** (`--backend api`): the vector reading over any embeddings endpoint, local or
  hosted, without the model's weights. `jul models add NAME --repo ollama:qwen3-embedding:0.6b --backend api`
  (also `openai:`, `mistral:`, `voyage:`, or `http(s)://host/v1#model` for any OpenAI-compatible server) fits
  the center and tau as for a local model. One layer, no logits (Noul and Score are read as vectors).
- **Escalation on confidence**: `jul.Escalation` chains deciders (a local client, then `SystemOneHTTP` for
  any `/v1/systemone` server: Jev, Ollama's Nimble, Kev, another `jul serve`); only the questions answered
  below the bar go on to the next one, and the response says per question which tier answered.
  `jul serve --escalate-to typesafe|ollama[:model]|URL --min-confidence 0.8` does the same over HTTP, with each
  provider's usual key variable (`TYPESAFE_API_KEY`).
- **Clef on Cloudflare Workers AI** as an escalation tier: `--escalate-to cloudflare:clef-flash` (or `:clef`),
  `jul.escalate.cloudflare_tier()` in Python, with `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.
- **`jul-decision-wemm-4b` is the default model** (alias `accurate`): `wemm-4b-4bit` plus LoRA adapters
  ([`usejul/jul-decision-wemm-4b`](https://huggingface.co/usejul/jul-decision-wemm-4b)) that read Noul,
  Score and Choice with the question and the text together. Decision bench (2,108 questions, PyTorch): 0.849,
  Jev 0.873. Attached on PyTorch; on MLX it reads as `wemm-4b-4bit` until the adapters are measured in
  4-bit. `wemm-4b-4bit` stays, vectors only.
- LoRA cross models read a Choice listwise when their `cross.json` has a `choice` entry (text, question and
  every option in one pass), mixed with the vector reading (`mix`).
- A preset's `cross` entry may name a repo for some backends only (`{"torch": ...}`); the others read with
  the vectors.
- **Contrastive heads on any backbone**: a reading mode where two projection heads (InfoNCE) read a frozen
  backbone, set by `contrastive.json` (layer, pooling, rendering, Noul texts, prefix), on MLX, torch or onnx,
  decoders and encoders. `jul models add <name> --repo <backbone> --train-heads rows.jsonl` trains them;
  `autotune` trains its usual head on the same embedding.
- CLM-8B is one profile of it: `jul models add clm-8b --repo Contrastive-LM/CLM-v0.1-8B` converts CLM's heads
  and runs them on their frozen Qwen3-8B (MLX 8-bit or torch bf16), without vLLM; CLM's reference answers
  reproduced within 0.025. docs/models.md#contrastive-heads-any-backbone-clm-8b.

## 0.3.0 — 2026-09-28

### Added

- **Cross models**: a second reading that takes the question and the text together, for the questions an
  embedding model answers near chance (paraphrase, inference, compositions, dates, "is it late?"). A preset
  gets one with `jul models add --cross`, or by itself from a repo that carries a `cross/` folder; its
  `cross.json` declares the types it reads (Noul and Score), the vectors read everything else.
  `method="vector"` or `method="cross"` forces one reading for a call.
- **LoRA cross models**: the cross reading as adapters on the preset's own model instead of a second
  encoder (`"method": "lora"` in `cross.json`), switched on only while a Noul or a Score is read, so one model
  sits in memory and the vector reading is unchanged. MLX and PyTorch.
  [`usejul/jul-decision-wemm-4b-4bit-mlx`](https://huggingface.co/usejul/jul-decision-wemm-4b-4bit-mlx):
  yes/no 0.841 on Kev's typed decisions, against 0.762 for the vectors of the default model.
- `jul pack` ships only the models a bundle's questions read with (vectors, cross, or both);
  `JUL_ONNX_CROSS_MODEL` points the cross graph at S3 as `JUL_ONNX_MODEL` does the vector one.
- Encoders read their input prefixes from the model's `config_sentence_transformers.json`, else from
  `assets/text_prefixes.json` (`$JUL_HOME/text_prefixes.json` extends it).
- Models: [`usejul/jul-decision-e5-small`](https://huggingface.co/usejul/jul-decision-e5-small) and its ONNX
  8-bit build, e5-small trained on jul decisions, with a cross model; a step-by-step AWS Lambda guide.

### Changed

- `autotune`: a head for a Noul or a Score must beat the cross model on the labeled examples to take the
  question; otherwise the question stays with the cross model.
- PyTorch: texts behind a cached prefix that cannot be repeated over a batch (the recurrent state of
  Qwen3.5 and WeMM-Embedding) are read in one batch, the prefix run again with each: about 150 texts/s
  instead of 12 for WeMM-Embedding-4B on an A10G. MLX keeps one at a time.

### Fixed

- `jul models add <built-in preset> --cross ...` saved the preset as `<name>@None.json`.

## 0.2.0 — 2026-09-26

### Added

- **onnx backend** (ONNX Runtime on CPU, no torch) and **encoders as backbones** (BERT, XLM-R,
  multilingual-e5): read as they were trained, mean-pooled, with their own input prefix. `JUL_HOME` moves
  everything jul writes.
- `jul pack` (first named `jul compile`): a fixed set of questions and its models as a bundle, for a
  deployment such as AWS Lambda; per-question formulations.
- `autotune` with lexical and hybrid heads (`features=`): e5-small reaches 0.790 on the Jev benchmark with a
  hybrid head, at 6 ms per text.
- `jul serve`: a local HTTP server that speaks the Jev protocol (`/v1/models` included), so the official SDK
  talks to it; single-option Choice.

### Changed

- The version comes from git (setuptools-scm): `main` publishes dev builds to TestPyPI, tags release.

## 0.1.1 — 2026-09-24

### Changed

- `wemm-4b-4bit` (WeMM-Embedding-4B, MLX 4-bit) is the new default model, in place of `minicpm5-2b`;
  `qwen3.5-9b` is dropped.
- Project links move to the `usejul` organization.

## 0.1.0 — 2026-09-24

First release: typed decisions (`Choice`, `Noul`, `Score`) with calibrated probabilities from one forward
pass of a local model, behind the interface of TypeSafe's (Jev) Python SDK.

- MLX and PyTorch backends; presets for `minicpm5-2b` (the default) and `qwen3.5-9b`; `jul models add` fits
  a preset for any model on the dev sets.
- Decision models read with their own pointer format, with a vector fallback and a measured routing
  threshold.
- `Context`, `autotune` (per-task heads), `jul synth` (synthetic labeled data), `jul setup`.
- Apache-2.0.
