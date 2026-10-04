# Writing good questions

The model compares your text with what you wrote for each option. It reads **descriptions**, not your
intentions: the same model can go from wrong to right by rewording an option. This page is the checklist,
cheapest gain first.

## Describe every option

The key is what you get back; the description is what the model reads. Without a description, the key
alone is read, and a key like `T2` or `escal` means nothing to it.

```python
# weak: the model reads "billing", "tech", "other"
Choice(instructions="Which team?", criteria=["billing", "tech", "other"])

# better: it reads what each team actually handles
Choice(instructions="Which team should handle this support ticket?",
       criteria={"billing": "payments, invoices, refunds, double charges",
                 "tech": "bugs, errors, crashes, the app not working",
                 "other": "anything else: partnerships, press, job applications"})
```

Write descriptions as a colleague would need them: a few words or a short sentence, the vocabulary your
texts actually use ("charged twice", not "duplicate transaction event").

## Make options distinct

Two options that describe overlapping things split the probability between them and both look unsure. If a
ticket can be about billing *and* a bug, either ask two questions (a `Choice` for the team, a `Noul` for
"is it a bug?") or define the boundary in the descriptions ("bugs in the payment page go to billing").

## A catch-all, described

A `Choice` always picks one of your options: it cannot say "none of these". Add an option for the rest and
describe it, or the closest wrong option wins.

## Pick the right type

| You want | Use | Not |
| --- | --- | --- |
| one label among several | `Choice` | several `Noul`s, which do not sum to 1 |
| a yes/no you will threshold | `Noul` | a two-option `Choice` (it works, but `Noul` is read better by the cross models) |
| an intensity, a grade, a priority | `Score`, levels lowest first | a `Choice` (it loses the order) |
| several independent facts | several questions in one call | one `Choice` mixing them |

A `Score` answers the **expected level**: with levels `["Calm", "Annoyed", "Very angry"]`, `1.8` means
"close to Very angry". Read `probabilities` when you need the distribution, `confidence` for how peaked it is.

## Describe yes and no when they are not obvious

```python
Noul(instructions="Is the delivery late?",
     criteria=NoulCriteria(true="it arrived after the promised date or has not arrived",
                           false="it arrived on time or early"))
```

Without criteria, yes and no are read as "Yes." and "No.".

## Use the probabilities

Every answer comes with a probability, and the temperature that produces it was fitted so that it means
what it says: on Jev's benchmark, the default model's answers at 0.9 are right close to 90% of the time
(expected calibration error 0.084, Jev 0.156). Use it:

- **automate above a threshold**, send the rest to a human;
- **escalate** the unsure ones to a bigger model ([Escalation](serve.md#escalation-local-first-a-bigger-decider-when-unsure));
- **check the threshold on your data**: calibration was measured on other texts than yours. Label 100 to 200
  of your own and look at what a given bar lets through.

## Measure, do not guess

Wording effects are real but uneven: listing the options in the prompt helped on topics and hurt on
sentiment; a context description helped one model and hurt another. Keep 100 to 200 labeled texts aside,
change one thing at a time, and keep what wins on them. `jul run` plus a few lines of Python is enough:

```python
import json
gold = {r["id"]: r["team"] for r in map(json.loads, open("gold.jsonl"))}
pred = {r["id"]: r["answers"]["team"]["choice"] for r in map(json.loads, open("answers.jsonl"))}
print(sum(pred[i] == gold[i] for i in gold) / len(gold))
```

## When wording is not enough

| Symptom | Try |
| --- | --- |
| right on most texts, wrong on a style (short, slangy, in-house jargon) | a [Context](tuning.md#context--what-the-data-looks-like) with ~50 real texts |
| systematically off on your labels | [`autotune`](tuning.md#autotune--when-it-is-off-key) with a few hundred labeled examples |
| yes/no about how two things relate (is it a paraphrase, does it follow, is it on time) near chance | a preset with a cross model ([Models](models.md#cross-models-reading-the-question-and-the-text-together)) |
| many options (50+), fine-grained | a bigger embedding model, then `autotune` |
| needs knowledge the text does not contain | a bigger decider, through [Escalation](serve.md#escalation-local-first-a-bigger-decider-when-unsure) |
