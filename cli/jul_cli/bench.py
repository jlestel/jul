"""`jul bench`: which model to use, measured on the user's own questions and data.

    jul bench test.jsonl --models jul-decision-e5-small,minicpm5-2b
    jul bench test.csv --train train.csv --models fast,accurate --output results.json

Each test row carries its question, so a model is judged per question ("which tool?" and "is it urgent?"
are not the same task):

    {"type": "choice", "question": "Which team?", "options": {"billing": "payments", "tech": "bugs"},
     "state": "I was charged twice", "answer": "billing"}

CSV has the same columns, options as `key:description|key:description` (or `key|key`). `type` defaults
to choice; a noul answers true/false, a score the level index (options are the levels, lowest first).

With `--train`, every model is measured zero-shot, then autotuned on the train rows and measured again on
the same test rows. Train and test are checked for overlap first: the same text in both (after folding
case, punctuation and digits) stops the bench, near duplicates are reported. Nothing is saved: the tuned
heads live in a temporary context.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from jul import Choice, Noul, Score

TYPES = ("choice", "noul", "score")
NEAR = 0.8          # character 5-gram Jaccard above which two texts count as near duplicates
Z = 1.96


# --- reading ------------------------------------------------------------------------------------

@dataclass
class Task:
    """One question of the user's data, with its test (and train) rows."""

    name: str
    kind: str
    instructions: str
    options: dict[str, str]          # key -> description (levels for a score: "0", "1", ...)
    test: list[tuple[str, str]] = field(default_factory=list)    # (state, answer key)
    train: list[tuple[str, str]] = field(default_factory=list)

    def question(self):
        if self.kind == "noul":
            return Noul(instructions=self.instructions)
        if self.kind == "score":
            return Score(instructions=self.instructions, criteria=list(self.options.values()))
        return Choice(instructions=self.instructions, criteria=dict(self.options))


def _options(raw, kind: str) -> dict[str, str]:
    if kind == "noul":
        return {"true": "", "false": ""}
    if isinstance(raw, str):
        raw = [p for p in (s.strip() for s in raw.split("|")) if p]
        if kind == "choice" and any(":" in p for p in raw):
            raw = dict((p.split(":", 1) + [""])[:2] for p in raw)
            raw = {k.strip(): v.strip() for k, v in raw.items()}
    if kind == "score":
        levels = list(raw.values()) if isinstance(raw, dict) else list(raw or [])
        return {str(i): str(v) for i, v in enumerate(levels)}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {str(k): "" for k in raw or []}


def _answer(value, kind: str, options: dict[str, str]) -> str:
    if kind == "noul":
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "yes", "1", "oui", "vrai"}:
                return "true"
            if v in {"false", "no", "0", "non", "faux"}:
                return "false"
            raise ValueError(f"noul answer {value!r}: expected true or false")
        return "true" if value else "false"
    key = str(value).strip()
    if kind == "score" and key not in options:
        # a level given by its text rather than its index
        by_text = {v: k for k, v in options.items()}
        key = by_text.get(key, key)
    if key not in options:
        raise ValueError(f"answer {value!r} is not one of the options {list(options)}")
    return key


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{n}: not JSON ({exc.msg})") from None
    return rows


def load_tasks(test_path, train_path=None) -> list[Task]:
    """Rows grouped by question (type, instructions, options): one Task each, in order of appearance."""
    tasks: dict[tuple, Task] = {}
    for split, path in (("test", test_path), ("train", train_path)):
        if path is None:
            continue
        for n, row in enumerate(read_rows(path), 1):
            where = f"{path}:{n}"
            kind = str(row.get("type") or "choice").strip().lower()
            if kind not in TYPES:
                raise SystemExit(f"{where}: unknown type {kind!r} (choice, noul or score)")
            instructions = str(row.get("question") or row.get("instructions") or "").strip()
            state = row.get("state", row.get("text"))
            if not instructions or state in (None, "") or row.get("answer") in (None, ""):
                raise SystemExit(f"{where}: each row needs a question, a state and an answer")
            if not isinstance(state, str):
                state = json.dumps(state, ensure_ascii=False, sort_keys=True)
            options = _options(row.get("options"), kind)
            if kind != "noul" and len(options) < 2:
                raise SystemExit(f"{where}: a {kind} needs at least two options")
            try:
                answer = _answer(row["answer"], kind, options)
            except ValueError as exc:
                raise SystemExit(f"{where}: {exc}") from None
            key = (kind, instructions, tuple(options.items()))
            if key not in tasks:
                if split == "train":
                    raise SystemExit(f"{where}: this question is not in the test set "
                                     f"({instructions!r}); train rows must match a test question")
                tasks[key] = Task(name=row.get("name") or instructions, kind=kind,
                                  instructions=instructions, options=options)
            getattr(tasks[key], split).append((state, answer))
    if not tasks:
        raise SystemExit(f"{test_path}: no rows")
    return list(tasks.values())


# --- overlap ------------------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Folded for comparison: case, accents, punctuation, and every number becomes 0 (an id, a plate)."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\d+", "0", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _grams(text: str, n: int = 5) -> set[str]:
    t = f" {text} "
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def find_overlap(tasks: list[Task], threshold: float = NEAR) -> dict:
    """Exact and near duplicates between train and test, and duplicates inside test. Texts, not labels:
    the same text under another answer is still a leak."""
    test = [(t.name, s) for t in tasks for s, _ in t.test]
    train = [(t.name, s) for t in tasks for s, _ in t.train]
    norm_train: dict[str, str] = {}
    for _, s in train:
        norm_train.setdefault(normalize(s), s)
    grams_train = [(s, _grams(normalize(s))) for s in dict.fromkeys(s for _, s in train)]

    exact, near = [], []
    for task, s in test:
        n = normalize(s)
        if n in norm_train:
            exact.append({"question": task, "test": s, "train": norm_train[n]})
            continue
        g = _grams(n)
        best = max(((jaccard(g, gt), st) for st, gt in grams_train), default=(0.0, None))
        if best[0] >= threshold:
            near.append({"question": task, "test": s, "train": best[1], "similarity": round(best[0], 3)})

    seen: dict[tuple, int] = {}
    for task, s in test:
        seen[(task, normalize(s))] = seen.get((task, normalize(s)), 0) + 1
    within = [{"question": q, "text": n, "count": c} for (q, n), c in seen.items() if c > 1]
    return {"exact": exact, "near": near, "within_test": within, "threshold": threshold}


def drop_overlap(tasks: list[Task], overlap: dict) -> int:
    leaked = {(o["question"], o["test"]) for o in overlap["exact"] + overlap["near"]}
    dropped = 0
    for t in tasks:
        before = len(t.test)
        t.test = [(s, a) for s, a in t.test if (t.name, s) not in leaked]
        dropped += before - len(t.test)
    return dropped


# --- measuring ----------------------------------------------------------------------------------

def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def predicted(kind: str, answer) -> str:
    if kind == "noul":
        return "true" if answer.noul >= 0.5 else "false"
    if kind == "score":
        return str(int(round(answer.score)))
    return answer.choice


def evaluate(client, task: Task, context=None) -> dict:
    question = task.question()
    hits, errors, times = 0, [], []
    for state, gold in task.test:
        t = time.perf_counter()
        response = client.system_one(state=state, questions={"q": question}, context=context)
        times.append((time.perf_counter() - t) * 1000)
        got = predicted(task.kind, response.answers["q"])
        hits += got == gold
        if task.kind == "score":
            errors.append(abs(response.answers["q"].score - int(gold)))
    n = len(task.test)
    lo, hi = wilson(hits, n)
    out = {"n": n, "correct": hits, "accuracy": round(hits / n, 4) if n else None,
           "ci95": [round(lo, 4), round(hi, 4)], "latency_ms_p50": round(_pct(times, 50), 1),
           "latency_ms_p95": round(_pct(times, 95), 1)}
    if errors:
        out["mae"] = round(sum(errors) / len(errors), 4)
    return out


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = (len(xs) - 1) * q / 100
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def bench_model(model: str, tasks: list[Task], backend=None, make_client=None, log=print) -> dict:
    """Zero-shot on every task, then (tasks with train rows) autotuned on a throwaway context."""
    from jul import Context
    if make_client is None:
        from jul import TypeSafeClient

        def default_client(model, backend, home):
            return TypeSafeClient(model=model, backend=backend, context_home=home)
        make_client = default_client
    home = Path(tempfile.mkdtemp(prefix="jul-bench-"))
    client = None
    result: dict = {"model": model, "questions": {}}
    try:
        client = make_client(model, backend, home)
        t0 = time.perf_counter()
        for task in tasks:
            log(f"  {model} · {task.name} · zero-shot ({len(task.test)})")
            entry = {"zero_shot": evaluate(client, task)}
            if task.train:
                ctx = Context(name="bench")
                try:
                    report = client.autotune(ctx, {"q": task.question()},
                                             [(s, {"q": _label(task.kind, a)}) for s, a in task.train],
                                             save=False)["q"]
                    log(f"  {model} · {task.name} · autotuned on {len(task.train)}")
                    entry["autotune"] = {**evaluate(client, task, context=ctx), "activated": report.activated,
                                         "reason": report.reason, "n_train": len(task.train)}
                except ValueError as exc:      # a decision model: no autotune
                    entry["autotune"] = {"skipped": str(exc)}
            result["questions"][task.name] = entry
        result["seconds"] = round(time.perf_counter() - t0, 1)
        result["resolved_model"] = getattr(client, "model", model)
    except Exception as exc:  # one model failing does not lose the others
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if client is not None:
                client.close()
        except Exception:
            pass
    return result


def _label(kind: str, key: str):
    if kind == "noul":
        return key == "true"
    if kind == "score":
        return int(key)
    return key


def best_of(entry: dict) -> tuple[str, dict]:
    """The setting a user would ship: autotuned when its head was activated, else zero-shot."""
    tuned = entry.get("autotune") or {}
    if tuned.get("activated"):
        return "autotune", tuned
    return "zero-shot", entry["zero_shot"]


def recommend(results: list[dict], tasks: list[Task]) -> dict:
    """Per question: the most accurate model, and the fastest one whose interval still reaches it."""
    out = {}
    for task in tasks:
        cands = []
        for r in results:
            if "error" in r or task.name not in r["questions"]:
                continue
            how, m = best_of(r["questions"][task.name])
            cands.append((r["model"], how, m))
        if not cands:
            continue
        top = max(cands, key=lambda c: (c[2]["accuracy"], -c[2]["latency_ms_p50"]))
        tied = [c for c in cands if c[2]["ci95"][1] >= top[2]["accuracy"]]
        fast = min(tied, key=lambda c: c[2]["latency_ms_p50"])
        out[task.name] = {
            "best": {"model": top[0], "setting": top[1], "accuracy": top[2]["accuracy"]},
            "pick": {"model": fast[0], "setting": fast[1], "accuracy": fast[2]["accuracy"],
                     "latency_ms_p50": fast[2]["latency_ms_p50"]},
            "separable": len(tied) == 1,
            "tied": sorted(c[0] for c in tied),
        }
    return out


# --- terminal -----------------------------------------------------------------------------------

class Ink:
    """The site's amber phosphor (#ffb000), dimmed and inverted; plain text when not a terminal."""

    def __init__(self, stream=sys.stdout):
        self.on = stream.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"

    def _w(self, code, s):
        return f"\x1b[{code}m{s}\x1b[0m" if self.on else s

    def ph(self, s):
        return self._w("38;2;255;176;0", s)

    def dim(self, s):
        return self._w("38;2;150;104;0", s)

    def inv(self, s):
        return self._w("48;2;255;176;0;38;2;10;6;0", s) if self.on else f"[{s}]"

    def bold(self, s):
        return self._w("1;38;2;255;176;0", s)


def _vis(s: str) -> int:
    return len(re.sub(r"\x1b\[[0-9;]*m", "", s))


def _pad(s: str, w: int) -> str:
    return s + " " * max(0, w - _vis(s))


def bar(p: float, lo: float, hi: float, width: int = 20) -> str:
    """█ the accuracy, ░ up to the top of its 95% interval, · the rest."""
    a, h = round(p * width), round(hi * width)
    return "█" * a + "░" * max(0, h - a) + "·" * max(0, width - max(a, h))


def window(title: str, lines: list[str], ink: Ink, width: int = 0) -> list[str]:
    """A section: the title inverted (or between brackets), its lines indented below. No right border, so a
    long question or model name never breaks the layout."""
    out = [ink.inv(f" {title} ") if ink.on else f"[ {title} ]"]
    out += [("  " + line) if line else "" for line in lines]
    return out


def render(report: dict, ink: Ink | None = None) -> str:
    ink = ink or Ink()
    width = 86
    out: list[str] = []
    data = report["data"]
    out.append(ink.bold("JuL bench") + ink.dim(f"  ·  {data['test_rows']} test rows"
                                               + (f", {data['train_rows']} train rows" if data["train_rows"] else "")
                                               + f"  ·  {len(report['questions'])} question(s)"
                                               + f"  ·  {len(report['models'])} model(s)"))
    out.append("")

    ov = report["overlap"]
    if ov.get("checked"):
        near = f"near duplicates (>= {ov['threshold']:.2f})"
        lines = [f"{'exact duplicates train/test':<30}{len(ov['exact']):>5}",
                 f"{near:<30}{len(ov['near']):>5}",
                 f"{'duplicates inside test':<30}{len(ov['within_test']):>5}"]
        if ov.get("dropped"):
            lines.append(ink.dim(f"{ov['dropped']} leaked test rows dropped (--drop-overlap)"))
        for o in (ov["exact"] + ov["near"])[:3]:
            lines.append(ink.dim(f"  · {o['test'][:width - 12]}"))
        out += window("OVERLAP", lines, ink, width)
        out.append("")

    for q in report["questions"]:
        lines = [ink.dim(f"{q['type']} · {len(q['options'])} options · {q['n_test']} test"
                         + (f" · {q['n_train']} train" if q["n_train"] else ""))]
        lines.append(ink.dim(f"{'model':<26}{'setting':<11}{'accuracy':<23}{'95% ci':<14}{'p50 ms':>8}"))
        rec = report["recommendation"].get(q["name"], {})
        for r in report["results"]:
            if "error" in r:
                lines.append(f"{r['model'][:25]:<26}" + ink.dim("failed: " + r["error"][:46]))
                continue
            entry = r["questions"].get(q["name"])
            if not entry:
                continue
            rows = [("zero-shot", entry["zero_shot"])]
            tuned = entry.get("autotune")
            if tuned and "accuracy" in tuned:
                rows.append(("autotune" + ("" if tuned["activated"] else "*"), tuned))
            for i, (how, m) in enumerate(rows):
                name = r["model"] if i == 0 else ""
                pick = rec.get("pick", {})
                star = pick.get("model") == r["model"] and how.rstrip("*") == pick.get("setting")
                acc = f"{m['accuracy'] * 100:5.1f}% " + bar(m["accuracy"], *m["ci95"], width=14)
                cell = (f"{name:<26}" + ("\n  " + " " * 26 if len(name) > 25 else "") + f"{how:<11}{acc:<23}"
                        f"{m['ci95'][0] * 100:4.0f}–{m['ci95'][1] * 100:3.0f}%     {m['latency_ms_p50']:>8.0f}")
                lines.append(ink.inv(cell) if star and ink.on else (cell + "  ◄" if star else cell))
            if tuned and "skipped" in tuned:
                lines.append(ink.dim(f"{'':<26}autotune   n/a (decision model)"))
        if rec:
            p, b = rec["pick"], rec["best"]
            lines.append("")
            if len(rec["tied"]) == 1 and len([r for r in report["results"] if "error" not in r]) == 1:
                lines.append(ink.ph(f"► {p['model']} ({p['setting']})"))
            elif rec["separable"]:
                lines.append(ink.ph(f"► {p['model']} ({p['setting']}): clearly ahead"))
            else:
                lines.append(ink.ph(f"► {p['model']} ({p['setting']}): fastest within the interval of the best")
                             )
                lines.append(ink.dim(f"  {len(rec['tied'])} models not separable on {q['n_test']} rows: "
                                     + ", ".join(rec["tied"])[:width - 30]))
        out += window(q["name"], lines, ink, width)
        out.append("")

    notes = []
    if any(r.get("questions", {}).get(q["name"], {}).get("autotune", {}).get("activated") is False
           for r in report["results"] for q in report["questions"]):
        notes.append("* autotune head not activated: it did not beat zero-shot on its own held-out folds")
    small = [q["name"] for q in report["questions"] if q["n_test"] < 50]
    if small:
        notes.append(f"under 50 test rows ({', '.join(small)[:40]}): intervals are wide, prefer ~50+ per question")
    notes.append("pretraining contamination cannot be checked: a model may have seen public data")
    out += [ink.dim("· " + n) for n in notes]
    return "\n".join(out)


# --- command ------------------------------------------------------------------------------------

def run(a, make_client=None, stream=sys.stdout) -> dict:
    from jul.presets import DEFAULT_MODEL
    tasks = load_tasks(a.test, a.train)
    models = [m.strip() for m in (a.models or DEFAULT_MODEL).split(",") if m.strip()]
    log = (lambda s: print(s, file=sys.stderr, flush=True)) if not a.quiet else (lambda s: None)

    overlap = {"checked": bool(a.train), "exact": [], "near": [], "within_test": [], "threshold": a.near}
    if a.train:
        overlap.update(find_overlap(tasks, a.near), checked=True)
        if a.drop_overlap:
            overlap["dropped"] = drop_overlap(tasks, overlap)
        elif overlap["exact"] and not a.allow_overlap:
            for o in overlap["exact"][:5]:
                print(f"  {o['question']}: {o['test'][:100]!r}", file=sys.stderr)
            raise SystemExit(f"{len(overlap['exact'])} test row(s) also in train (after folding case, punctuation "
                             "and digits): the autotune score would be inflated. Remove them, or pass "
                             "--drop-overlap (drop them from test) or --allow-overlap.")
    else:
        overlap["within_test"] = find_overlap(tasks, a.near)["within_test"]
    tasks = [t for t in tasks if t.test]

    results = []
    for m in models:
        name, _, backend = m.partition("@")
        r = bench_model(name, tasks, backend or a.backend, make_client, log)
        r["model"] = m
        results.append(r)
    report = {
        "data": {"test": str(a.test), "train": str(a.train) if a.train else None,
                 "test_rows": sum(len(t.test) for t in tasks), "train_rows": sum(len(t.train) for t in tasks)},
        "models": models,
        "questions": [{"name": t.name, "type": t.kind, "instructions": t.instructions, "options": t.options,
                       "n_test": len(t.test), "n_train": len(t.train)} for t in tasks],
        "overlap": overlap,
        "results": results,
        "recommendation": recommend(results, tasks),
    }
    if a.output:
        Path(a.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if a.json:
        print(json.dumps(report, indent=2, ensure_ascii=False), file=stream)
    else:
        print(render(report, Ink(stream)), file=stream)
        if a.output:
            print(Ink(stream).dim(f"\n→ {a.output}"), file=stream)
    return report
