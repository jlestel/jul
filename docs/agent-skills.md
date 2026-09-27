# Agent skills

Three skills give a coding agent (Claude Code, Codex, Cursor…) what it needs to build with JuL
without inventing fields: the API and how to write questions, tuning on your data, and deployment.
They point to these docs for the details, so they stay short.

| Skill | Use it to |
| --- | --- |
| [`jul`](../skills/jul/SKILL.md) | install JuL, write `Choice` / `Noul` / `Score` questions, know what the vector reading cannot do, use the probabilities, measure before claiming |
| [`jul-tune`](../skills/jul-tune/SKILL.md) | measure zero-shot, add a `Context`, `autotune`, `jul synth` |
| [`jul-deploy`](../skills/jul-deploy/SKILL.md) | `jul serve`, `jul pack` and `Bundle`, AWS Lambda |

## Install

**Claude Code**, as a plugin:

```bash
claude plugin marketplace add usejul/jul
claude plugin install jul@usejul
```

Invoke one with `/jul:jul`, `/jul:jul-tune` or `/jul:jul-deploy`, or just ask to "use the JuL skill".

**Other agents**, with [skills.sh](https://skills.sh):

```bash
npx skills add usejul/jul --skill jul      # and jul-tune, jul-deploy
```

**By hand**: copy the directories under [`skills/`](../skills) into your agent's skills directory.

## Prompts to start with

```text
Using the JuL skill, find the places in this project where a regex, a keyword list or an LLM
prompt-and-parse step decides something about a text, and propose which could become a JuL question.
```

```text
Using the JuL and jul-tune skills, label 50 real examples from data/tickets.jsonl with me, measure
zero-shot accuracy per question, then tell me whether a Context or autotune is worth it.
```

## Good practice

- Keep the questions and thresholds in one file (`questions.yaml`): that is what a human reviews.
  Agents write poor option descriptions; edit them together.
- Ask for measured numbers on your own examples, never a benchmark figure.
