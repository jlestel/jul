# Concepts

Six words explain JuL: **state**, **question**, **option**, **reading**, **context**, **head**. This page
gives each one in a paragraph, then how they fit together. The rest of the docs assumes them.

## State

The text you want a decision about: a support ticket, an email, a log line, a product review. One call,
one state. It can be a string, or any object: a dict is turned into JSON before the model reads it, so
`{"subject": …, "body": …}` works as is.

## Question

What you want to know about the state. Three types, the same as Jev's:

| Type | Asks | Answers | Example |
| --- | --- | --- | --- |
| `Choice` | which one of these options? | `choice` (a key), `probabilities` per option, `confidence` | which team handles this ticket? |
| `Noul` | yes or no? | `noul`, the probability of yes, from 0 to 1 | does it report a bug? |
| `Score` | how much, on this scale? | `score`, the expected level from 0 to n−1, `probabilities` per level, `confidence` | how angry is the customer? |

A call can ask many questions about the same state at once; each comes back under the name you gave it.

## Option

A question's possible answers. A `Choice` has a **key** (what you get back, `"billing"`) and a
**description** (what the model reads, `"payments, invoices, refunds"`). The model never sees your keys if
you give descriptions, so write descriptions a colleague would understand: see
[Writing good questions](questions.md). A `Score`'s options are its levels, lowest first. A `Noul`'s options
are yes and no, which you may describe (`NoulCriteria(true=…, false=…)`).

## Reading

How a model is turned into an answer. JuL never lets the model write: it runs the model over the text,
stops it **one step before its first word**, and reads the answer out of the hidden state, the vector the
model built to understand the text.

```text
STATE "I was charged twice"
               │
┌──────────────┴──────────────┐
│ the model (your machine)    │
│ layers 1 → 31, then stop.   │
└──────────────┬──────────────┘
               │ hidden state
               ▼
      cosine(state, option) ◄── options
               │         (encoded once,
               ▼              cached)
       softmax(cos / τ)
               │
               ▼
billing    ████████████████  0.88
technical  ██                0.11
sales      ▏                 0.01
```

That default is the **vector** reading: the state and each option go through the same prompt, and the
answer is the option whose vector is closest. Everything that does not depend on the state (the prompt
prefix, the option vectors) is computed once and cached, so a call only pays for its own tokens. A
temperature `τ`, fitted per model, turns the similarities into probabilities that mean what they say.

Other readings exist for other kinds of models and questions: a **cross** reading that reads the question
and the text together (better for yes/no about how two things relate), a **pointer** reading for models
trained to decide, a **contrastive** one for projection heads. You rarely choose: the model's preset does.
The full list is in [Models › Every reading](models.md#every-reading-and-every-setting).

## Preset

A model plus the settings that make it read well: which layer to read, which prompts, which temperature,
which center. The built-in presets were fitted on development data that the benchmarks never use.
`jul models add` fits one for any other model with the same protocol, so **adding a model is adding a
preset, not code**. See [The hub](hub.md).

## Backend

What runs the model: **mlx** (Apple Silicon), **torch** (PyTorch, on CUDA, CPU or MPS), **onnx** (ONNX
Runtime on CPU, for deployment), or **api** (an embeddings API: Ollama, OpenAI, Mistral, Voyage, any
OpenAI-compatible server). Same code, same answers' shape, whichever runs.

## Context

What your data looks like, given once and reused: a description, and a few dozen **unlabeled** texts of
your task. Their mean vector becomes the "center" the model compares against, which helps on texts with a
style of their own. A context is saved under a name (`~/.jul/contexts/<name>/`) and is also where tuned
heads live. Jev has no equivalent. See [Adapting to your data](tuning.md#context--what-the-data-looks-like).

## Head

A small classifier trained in seconds on **labeled** examples, on top of the vectors the model already
computes (`autotune`). The model is not changed and inference is as fast. JuL keeps a head only if it
beats zero-shot on examples it did not train on, so tuning cannot make things worse by surprise. See
[`autotune`](tuning.md#autotune--when-it-is-off-key).

## How they fit

```text
                questions (Choice / Noul / Score, their options)
                                  │
state ──► preset (model + reading + settings) on a backend ──► answers + probabilities
                                  ▲
                    context (unlabeled texts, tuned heads)
```

From the cheapest to the most effective way to get better answers:

1. **Better descriptions** of the options: free, often the biggest gain ([Writing good questions](questions.md)).
2. **Another model**: bigger, or trained for decisions ([The hub](hub.md)).
3. **A context** with ~50 unlabeled texts: helps on texts with their own style.
4. **`autotune`** with a few hundred labeled examples per question: the largest gain when you have labels.
5. **Escalation**: answer locally, and send only the unsure questions to a bigger decider
   ([Serving › Escalation](serve.md#escalation-local-first-a-bigger-decider-when-unsure)).

## What JuL is not

- **Not a chatbot.** It does not write text, explain itself or extract fields. It picks among options you
  give, or puts a number on a yes/no or a scale.
- **Not a knowledge base.** A question that needs facts the text does not contain ("is this company listed
  in Paris?") is out of reach of a small model reading one text.
- **Not magic on arithmetic.** Comparing two dates works; computing a gap (an age, a warranty in months)
  stays near chance. See [Models › LoRA cross model](models.md#a-cross-model-on-the-presets-own-weights-lora).
- **Fitted on English.** It answers French and other languages, measured lower: test on your data.
