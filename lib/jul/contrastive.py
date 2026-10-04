"""The contrastive reading: projection heads on a frozen model, for any backbone jul loads.

A typed question is answered without generating anything: the state (with the question's instructions
after a blank line) and every option are embedded by the frozen backbone, a state head and an action
head (two small MLPs trained with InfoNCE) project them to a shared space, the score of an option is
`scale * cos(state_head(s), action_head(o))` and a softmax over the options is the answer.

Nothing here is tied to a model. A contrastive preset is a backbone (any repo jul reads, on MLX, torch or
onnx) plus a directory holding:

    contrastive.json   how to read: backbone repo per backend, the layer and pooling of the embedding,
                       the token limit, how texts are written (render, Noul templates, prefix), the scale
    heads.npz          both MLPs as plain arrays: inference needs neither torch nor any model package

The heads only mean something on the backbone they were trained on. Two ways to get them:

* train them on any backbone: `jul models add <name> --repo <backbone> --train-heads <data.jsonl>`
  (`train_heads` below, needs torch for the training only);
* convert a published checkpoint: CLM-8B (`Contrastive-LM/CLM-v0.1-8B`, heads on a frozen Qwen3-8B) is one
  profile of this format, `convert` writes it; `jul models add clm-8b --repo Contrastive-LM/CLM-v0.1-8B`
  does it by itself.
"""

from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import NOUL_DEFAULTS, Option

SPEC_FILE = "contrastive.json"
HEADS_FILE = "heads.npz"

#: Noul options as texts the action head can tell apart: "Yes."/"No." alone carry nothing of the question.
#: CLM's own wording, used by default for trained heads too.
NOUL_TEMPLATES = {"true": "Yes. This is true: {instructions}", "false": "No. This is false: {instructions}"}

#: The CLM-8B profile, used by `convert` only. On MLX, 8-bit: on CLM's reference requests (M1 Pro) it answers
#: within 0.02 of bf16 vLLM (urgency 0.865 vs 0.84-0.85), while 4-bit moves the Noul to 0.713 (same argmax).
CLM_REPO = "Contrastive-LM/CLM-v0.1-8B"
CLM_FILE = "CLM_v0.1-8B.pt"
CLM_BACKBONE = {"torch": "Qwen/Qwen3-8B", "mlx": "mlx-community/Qwen3-8B-8bit"}
#: What vLLM's pooling endpoint returns for Qwen3, which CLM's heads were trained on.
CLM_READING = {"layer": "final", "pooling": "last", "max_tokens": 2048, "add_special_tokens": False,
               "render": "fields", "noul": NOUL_TEMPLATES, "prefix": ""}


# --- the texts ----------------------------------------------------------------------------------

def to_text(x: Any, indent: int = 0) -> str:
    """The "fields" rendering of a state (CLM's `schema.py`): an object becomes `key: value` fields (top
    level separated by a blank line), an array one `- item` line per element."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(x)
    pad = " " * indent
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}{k}:\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}{k}: {to_text(v)}")
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(x, (list, tuple)):
        parts = []
        for v in x:
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}-\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}- {to_text(v)}")
        return "\n".join(parts)
    return json.dumps(x, ensure_ascii=False)


def _render(x: Any, how: str) -> str:
    if how == "json":
        from .types import serialize_state
        return "" if x is None else serialize_state(x)
    return to_text(x)


def state_text(state: Any, instructions: str | None, render: str = "fields") -> str:
    """Context first, question last, a blank line between: the state head reads the question too."""
    s, i = _render(state, render).strip(), to_text(instructions).strip()
    return f"{s}\n\n{i}" if s and i else (s or i)


def option_texts(kind: str, instructions: str | None, options: list[Option],
                 noul: dict[str, str] | None) -> tuple[list[str], list[int]]:
    """(candidate texts, and for each the index of the jul option).

    choice: the description, else the key, verbatim. score: the level descriptions. noul: in the order
    (false, true), each `<key>: <description>`; jul's default "Yes."/"No." counts as no description, so
    the spec's Noul templates (built from the instructions) are used when it has some.
    """
    if kind == "noul":
        by_key = {o.key: o for o in options}
        ins = to_text(instructions).strip()
        texts = []
        for k in ("false", "true"):
            d = by_key[k].description
            if noul and (not d or d == NOUL_DEFAULTS[k]):
                d = noul[k].format(instructions=ins) if ins else k
            texts.append(f"{k}: {d or k}")
        return texts, [next(i for i, o in enumerate(options) if o.key == k) for k in ("false", "true")]
    if kind == "score":
        return [to_text(o.description) for o in options], list(range(len(options)))
    return [o.description if o.description not in (None, "") else o.key for o in options], list(range(len(options)))


# --- the spec and the heads ---------------------------------------------------------------------

@dataclass(frozen=True)
class ContrastiveSpec:
    """How a backbone is read for its heads, and where the heads are.

    layer: "final" = the last layer after the final norm (`Backbone.last_hidden`, decoders on MLX and
    torch), or a layer index read with the backbone's own features (any backend, encoders included).
    pooling: "last" token or "mean" over the tokens; an encoder is always read as its mean.
    render: "fields" (CLM's `key: value`) or "json" (jul's `serialize_state`) for a structured state.
    noul: templates for a Noul without descriptions ({instructions}), or None for "Yes."/"No.".
    prefix: prepended to every text (an embedding model's input convention, e5's "query: ").
    """

    backbone: dict[str, str]
    max_tokens: int
    scale: float
    directory: Path
    layer: str | int = "final"
    pooling: str = "last"
    add_special_tokens: bool = False
    render: str = "fields"
    noul: dict[str, str] | None = None
    prefix: str = ""
    heads_file: str = HEADS_FILE
    source: str = ""

    @classmethod
    def load(cls, directory: str | Path) -> "ContrastiveSpec":
        directory = Path(directory)
        d = json.loads((directory / SPEC_FILE).read_text())
        if d.get("method") != "contrastive":
            raise ValueError(f"{directory / SPEC_FILE}: unsupported method {d.get('method')!r}")
        if d.get("pooling", "last") not in ("last", "mean"):
            raise ValueError(f"{directory / SPEC_FILE}: pooling is 'last' or 'mean' (got {d['pooling']!r})")
        if d.get("render", "fields") not in ("fields", "json"):
            raise ValueError(f"{directory / SPEC_FILE}: render is 'fields' or 'json' (got {d['render']!r})")
        layer = d.get("layer", "final")
        return cls(backbone=dict(d["backbone"]), max_tokens=int(d["max_tokens"]), scale=float(d["scale"]),
                   directory=directory, layer=layer if layer == "final" else int(layer),
                   pooling=d.get("pooling", "last"), add_special_tokens=bool(d.get("add_special_tokens", False)),
                   render=d.get("render", "fields"), noul=d.get("noul"), prefix=d.get("prefix", ""),
                   heads_file=d.get("heads", HEADS_FILE), source=d.get("source", ""))


def write_spec(out: Path, backbone: dict[str, str], scale: float, source: str, reading: dict) -> None:
    spec = {"method": "contrastive", "source": source, "backbone": backbone, "scale": scale,
            **reading, "heads": HEADS_FILE}
    (out / SPEC_FILE).write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")


def _erf(x: np.ndarray) -> np.ndarray:
    """erf without scipy: Abramowitz-Stegun 7.1.26 is 1.5e-7 off, below float32 noise here."""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return (s * y).astype(x.dtype)


#: GELU is the exact (erf) one, torch's default, which CLM's heads use.
_ACT = {"gelu": lambda x: 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0))),
        "relu": lambda x: np.maximum(x, 0.0),
        "silu": lambda x: x / (1.0 + np.exp(-x))}


class Head:
    """`hidden -> width -> ... -> proj` MLP: CLM's `make_head`, in numpy (float32)."""

    def __init__(self, arrays: dict[str, np.ndarray], prefix: str, cfg: dict):
        g = lambda k: np.asarray(arrays[f"{prefix}.{k}"], dtype=np.float32)
        self.inp = (g("inp.weight"), g("inp.bias"))
        self.out = (g("out.weight"), g("out.bias"))
        n_hidden = int(cfg["depth"]) - 2
        self.hidden = [(g(f"hidden.{i}.weight"), g(f"hidden.{i}.bias")) for i in range(n_hidden)]
        self.norms = ([(g(f"norms.{i}.weight"), g(f"norms.{i}.bias")) for i in range(n_hidden)]
                      if cfg.get("layernorm") else [None] * n_hidden)
        self.act = _ACT[cfg.get("activation", "gelu")]
        self.residual = bool(cfg.get("residual", False))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(n, hidden) -> (n, proj), L2-normalised."""
        x = self.act(np.asarray(x, dtype=np.float32) @ self.inp[0].T + self.inp[1])
        for (w, b), norm in zip(self.hidden, self.norms):
            h = x @ w.T + b
            if norm is not None:
                mu = h.mean(-1, keepdims=True)
                var = ((h - mu) ** 2).mean(-1, keepdims=True)
                h = (h - mu) / np.sqrt(var + 1e-5) * norm[0] + norm[1]
            h = self.act(h)
            x = x + h if self.residual else h
        z = x @ self.out[0].T + self.out[1]
        return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)


def load_heads(spec: ContrastiveSpec) -> tuple[Head, Head]:
    with np.load(spec.directory / spec.heads_file) as f:
        arrays = {k: f[k] for k in f.files}
    cfg = json.loads(str(arrays.pop("cfg")))
    return Head(arrays, "state_head", cfg), Head(arrays, "action_head", cfg)


# --- reading ------------------------------------------------------------------------------------

class Embedder:
    """Texts -> (n, d) L2-normalised embeddings of the frozen backbone, read as the spec says.

    layer "final" reads `Backbone.last_hidden` (decoders on MLX and torch: the hidden states after the
    final norm, what an embedding server returns). An integer layer reads the backbone's own features
    (`forward_batch`, every backend, encoders included): [last token ; mean] for a decoder, the mean for
    an encoder.
    """

    def __init__(self, backbone, spec: ContrastiveSpec, batch: int = 16):
        self.backbone, self.spec, self.batch = backbone, spec, batch
        encoder = getattr(backbone, "architecture", "decoder") == "encoder"
        if encoder and spec.layer == "final":
            raise ValueError(f"{spec.directory}: an encoder is read at a layer index (\"layer\": <int>), "
                             "not \"final\"")
        if encoder and spec.pooling != "mean":
            raise ValueError(f"{spec.directory}: an encoder has no last token to read: pooling \"mean\"")
        self.encoder = encoder

    def tokens(self, text: str) -> list[int]:
        tok = self.backbone.tokenizer
        ids = tok.encode(self.spec.prefix + text, add_special_tokens=self.spec.add_special_tokens)
        return list(ids[-self.spec.max_tokens:]) or [tok.eos_token_id or 0]

    def __call__(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """(len(texts), d) unit vectors, and the tokens run."""
        seqs = [self.tokens(t) for t in texts]
        if self.spec.layer == "final":
            hs = [np.asarray(self.backbone.last_hidden(s), dtype=np.float32) for s in seqs]
            vs = [h[-1] if self.spec.pooling == "last" else h.mean(0) for h in hs]
        else:
            layer, vs = int(self.spec.layer), []
            for i in range(0, len(seqs), self.batch):
                chunk = seqs[i:i + self.batch]
                for f in self.backbone.forward_batch(chunk, layers=[layer], pools=[(0, len(q)) for q in chunk]):
                    f = np.asarray(f[layer], dtype=np.float32)
                    d = f.shape[0] // 2
                    # decoder: [last token ; mean over the text]; encoder: [mean over the sequence ; ...]
                    vs.append(f[:d] if self.encoder or self.spec.pooling == "last" else f[d:])
        e = np.stack(vs)
        return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12), sum(map(len, seqs))


class ContrastiveReader:
    """Embeds with the backbone, projects with the heads, scores `scale * cos`."""

    def __init__(self, backbone, spec: ContrastiveSpec, cache_size: int = 4096):
        self.backbone, self.spec = backbone, spec
        self.embedder = Embedder(backbone, spec)
        self.state_head, self.action_head = load_heads(spec)
        self._options: OrderedDict[str, np.ndarray] = OrderedDict()   # option text -> projected vector
        self._cache_size = cache_size

    def embed(self, text: str) -> tuple[np.ndarray, int]:
        """(d,) L2-normalised embedding of `text`, and its token count."""
        e, n = self.embedder([text])
        return e[0], n

    def _option_vectors(self, texts: list[str]) -> tuple[np.ndarray, int]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._options]
        spent = 0
        if missing:
            e, spent = self.embedder(missing)
            for t, z in zip(missing, self.action_head(e)):
                self._options[t] = z
        out = []
        for t in texts:
            self._options.move_to_end(t)
            out.append(self._options[t])
        while len(self._options) > max(self._cache_size, len(texts)):
            self._options.popitem(last=False)
        return np.stack(out), spent

    def read(self, state: Any, kind: str, instructions: str | None,
             options: list[Option]) -> tuple[np.ndarray, np.ndarray, int]:
        """(logits in jul's option order, the state's embedding, tokens run).

        The embedding is what `autotune` trains a head on: it already holds the question, since the
        state head reads `state + instructions`.
        """
        texts, index = option_texts(kind, instructions, options, self.spec.noul)
        e, spent = self.embed(state_text(state, instructions, self.spec.render))
        zc, more = self._option_vectors(texts)
        z = self.spec.scale * (zc @ self.state_head(e[None])[0])
        ordered = np.empty(len(options), dtype=np.float32)
        ordered[index] = z
        return ordered, e, spent + more


# --- training heads on any backbone ---------------------------------------------------------------

def typed_row(row: dict) -> tuple[str, Any, list[Option], int] | None:
    """(kind, question, options in jul's order, gold index) from one labeled row, None when unusable.

    A row: {"state", "type": choice|noul|score, "question" (the instructions), "options" (a list of
    option names, or {key: description}), "answer"}. choice: the gold option's index or key. noul: a
    bool, "yes"/"no"/"true"/"false" (or oui/non), or an index into options listed yes-first (the decision
    dataset's [yes, no, unknown]: unknown is skipped). score: the level index, or "score" in [0, 1].
    """
    from .types import Choice, Noul, Score, options_of
    kind, question, opts, answer = row["type"], row.get("question") or "", row.get("options"), row.get("answer")
    if kind == "noul":
        if isinstance(answer, bool):
            yes = answer
        elif isinstance(answer, str):
            a = answer.strip().lower()
            if a not in ("yes", "no", "true", "false", "oui", "non"):
                return None
            yes = a in ("yes", "true", "oui")
        elif answer in (0, 1):
            yes = answer == 0
        else:
            return None
        q = Noul(question)
        options = options_of(q)
        return kind, q, options, [o.key for o in options].index("true" if yes else "false")
    if kind == "score":
        if not opts or len(opts) < 2:
            return None
        q = Score(question, list(opts))
        n = len(opts)
        gold = (int(answer) if answer is not None and answer == answer
                else int(round(float(row["score"]) * (n - 1))))
        return kind, q, options_of(q), gold
    q = Choice(question, dict(opts) if isinstance(opts, dict) else {str(o): "" for o in opts})
    options = options_of(q)
    keys = [o.key for o in options]
    gold = keys.index(str(answer)) if str(answer) in keys and not isinstance(answer, int) else int(answer)
    return kind, q, options, gold


def rows_from_labeled(questions: dict, labeled: list) -> list[dict]:
    """autotune's (questions, [(state, {name: answer})]) as typed rows for `train_heads`."""
    from .types import Choice, Noul, NoulCriteria, Score
    rows = []
    for state, answers in labeled:
        for name, q in questions.items():
            if name not in answers:
                continue
            a = answers[name]
            if isinstance(q, Noul):
                c = q.criteria
                if isinstance(c, NoulCriteria) and (c.true or c.false):
                    rows.append({"state": state, "type": "choice", "question": q.instructions,
                                 "options": {"false": c.false or "No.", "true": c.true or "Yes."},
                                 "answer": "true" if str(a).lower() in ("true", "yes", "1") else "false"})
                    continue
                rows.append({"state": state, "type": "noul", "question": q.instructions,
                             "answer": str(a).lower() in ("true", "yes", "1")})
            elif isinstance(q, Score):
                rows.append({"state": state, "type": "score", "question": q.instructions,
                             "options": list(q.criteria), "answer": int(a)})
            else:
                c = q.criteria
                rows.append({"state": state, "type": "choice", "question": q.instructions,
                             "options": dict(c) if isinstance(c, dict) else list(c), "answer": str(a)})
    return rows


def default_reading(backbone, max_tokens: int = 512) -> dict:
    """How to read a backbone that has no heads yet: the final norm at the last token for a decoder on
    MLX/torch, the mean of the last layer for an encoder (with its input convention), the last layer's
    [last token] for a backend without `last_hidden` (onnx)."""
    from .backbone import Backbone
    if getattr(backbone, "architecture", "decoder") == "encoder":
        layer, pooling = backbone.n_layers - 1, "mean"
    elif type(backbone).last_hidden is Backbone.last_hidden or backbone.backend == "onnx":
        layer, pooling = backbone.n_layers - 1, "last"
    else:
        layer, pooling = "final", "last"
    return {"layer": layer, "pooling": pooling, "max_tokens": max_tokens, "add_special_tokens": False,
            "render": "json", "noul": NOUL_TEMPLATES, "prefix": getattr(backbone, "text_prefix", "") or ""}


def train_heads(backbone, rows: list[dict], out: str | Path, backbone_repos: dict[str, str],
                reading: dict | None = None, width: int | None = None, depth: int = 3, proj: int = 512,
                epochs: int = 40, batch: int = 256, lr: float = 1e-3, weight_decay: float = 0.01,
                val_frac: float = 0.1, patience: int = 6, in_batch: float = 0.5, seed: int = 0,
                log=print) -> dict:
    """Train a state head and an action head on the frozen `backbone` and write them to `out`.

    The loss is CLM's InfoNCE with the question's own options as the candidates (cross-entropy over
    `scale * cos`), plus `in_batch` x the same over every distinct option text of the batch, the other
    questions' options as extra negatives. The scale is learnt. The epoch kept is the one with the best
    accuracy on `val_frac` of the rows held out; the plain cosine of the backbone's embeddings (no heads)
    on the same rows is reported next to it. Needs torch, for the training only.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("training contrastive heads needs torch: pip install torch "
                          "(inference afterwards does not)") from exc
    reading = dict(reading or default_reading(backbone))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = ContrastiveSpec(backbone=backbone_repos, max_tokens=int(reading["max_tokens"]), scale=1.0,
                           directory=out, layer=reading["layer"], pooling=reading["pooling"],
                           add_special_tokens=bool(reading.get("add_special_tokens", False)),
                           render=reading.get("render", "json"), noul=reading.get("noul"),
                           prefix=reading.get("prefix", ""))
    embed = Embedder(backbone, spec)

    items, state_ids, option_ids = [], {}, {}
    for row in rows:
        t = typed_row(row)
        if t is None:
            continue
        kind, q, options, gold = t
        texts, index = option_texts(kind, q.instructions, options, spec.noul)
        st = state_text(row["state"], q.instructions, spec.render)
        cands = [option_ids.setdefault(x, len(option_ids)) for x in texts]
        items.append((state_ids.setdefault(st, len(state_ids)), cands, index.index(gold), kind))
    if len(items) < 20:
        raise ValueError(f"{len(items)} usable rows: training heads needs at least 20")
    log(f"embedding {len(state_ids)} states and {len(option_ids)} option texts")
    S, _ = embed(list(state_ids))
    A, _ = embed(list(option_ids))
    hidden = S.shape[1]
    width = width or min(1536, hidden)

    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    n_val = max(1, int(len(items) * val_frac))
    val, train = order[:n_val], order[n_val:]
    torch.manual_seed(seed)

    def make():
        layers = {"inp": torch.nn.Linear(hidden, width), "out": torch.nn.Linear(width, proj)}
        m = torch.nn.ModuleDict(layers)
        m.hidden = torch.nn.ModuleList(torch.nn.Linear(width, width) for _ in range(depth - 2))
        m.norms = torch.nn.ModuleList(torch.nn.LayerNorm(width) for _ in range(depth - 2))
        return m

    def run(m, x):
        x = F.gelu(m["inp"](x))
        for lin, nrm in zip(m.hidden, m.norms):
            x = F.gelu(nrm(lin(x)))
        return F.normalize(m["out"](x), dim=-1)

    sh, ah = make(), make()
    log_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    params = list(sh.parameters()) + list(ah.parameters()) + [log_scale]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    St, At = torch.from_numpy(S), torch.from_numpy(A)

    def batch_loss(idx, train_mode=True):
        rows_ = [items[i] for i in idx]
        uniq = sorted({c for _, cands, _, _ in rows_ for c in cands})
        pos = {c: j for j, c in enumerate(uniq)}
        zs = run(sh, St[[r[0] for r in rows_]])
        za = run(ah, At[uniq])
        logits = log_scale.exp().clamp(max=100.0) * zs @ za.t()
        own = torch.full_like(logits, float("-inf"))
        gold = torch.tensor([pos[cands[g]] for _, cands, g, _ in rows_])
        for i, (_, cands, _, _) in enumerate(rows_):
            own[i, [pos[c] for c in cands]] = 0
        loss = F.cross_entropy(logits + own, gold)
        if in_batch:
            loss = loss + in_batch * F.cross_entropy(logits, gold)
        hit = ((logits + own).argmax(1) == gold).float().sum().item()
        return loss, hit

    def accuracy(idx):
        with torch.no_grad():
            return sum(batch_loss(idx[i:i + batch])[1] for i in range(0, len(idx), batch)) / len(idx)

    def cosine_accuracy(idx):
        hit = 0
        for i in idx:
            s, cands, g, _ = items[i]
            hit += int(np.argmax(A[cands] @ S[s]) == g)
        return hit / len(idx)

    base = cosine_accuracy(val)
    log(f"{len(train)} train / {len(val)} held-out rows; plain cosine on the held-out rows: {base:.3f}")
    steps = max(1, math.ceil(len(train) / batch)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        rng.shuffle(train)
        total = 0.0
        for i in range(0, len(train), batch):
            loss, _ = batch_loss(train[i:i + batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        acc = accuracy(val)
        history.append({"epoch": epoch, "loss": total / max(1, math.ceil(len(train) / batch)),
                        "val_accuracy": acc, "scale": float(log_scale.detach().exp().clamp(max=100.0))})
        log(f"epoch {epoch}: loss {history[-1]['loss']:.3f}, held-out accuracy {acc:.3f}, "
            f"scale {history[-1]['scale']:.1f}")
        if acc > best:
            best, bad = acc, 0
            best_state = ({k: v.detach().clone() for k, v in sh.state_dict().items()},
                          {k: v.detach().clone() for k, v in ah.state_dict().items()},
                          float(log_scale.detach().exp().clamp(max=100.0)))
        else:
            bad += 1
            if bad >= patience:
                break
    s_state, a_state, scale = best_state
    arrays = {}
    for side, state in (("state_head", s_state), ("action_head", a_state)):
        for k, v in state.items():
            arrays[f"{side}.{k}"] = v.float().numpy()
    cfg = {"width": width, "depth": depth, "activation": "gelu", "layernorm": True, "residual": False,
           "hidden_size": hidden, "projection_dim": proj}
    np.savez(out / HEADS_FILE, cfg=np.array(json.dumps(cfg)), **arrays)
    report = {"rows": len(items), "train": len(train), "held_out": len(val), "cosine_accuracy": base,
              "heads_accuracy": best, "epochs": history}
    source = f"trained by jul on {getattr(backbone, 'repo', '?')}: {len(items)} rows, held-out " \
             f"{best:.3f} (plain cosine {base:.3f})"
    write_spec(out, backbone_repos, scale, source, {k: reading[k] for k in reading})
    (out / "training.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


# --- converting a published checkpoint (CLM-8B) --------------------------------------------------

def convert(checkpoint: str, out: str | Path, backbone: dict[str, str] | None = None,
            max_tokens: int = 2048) -> Path:
    """Write contrastive.json + heads.npz from a CLM-format `.pt` (a path, or a Hub repo holding one):
    `state_head` / `action_head` state dicts, `logit_scale`, `cfg`. The reading is CLM's (`CLM_READING`).

    Needs torch, once: the checkpoint is a pickled torch dict.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("converting a CLM checkpoint reads a torch .pt: pip install torch "
                          "(inference afterwards does not need it)") from exc
    path = Path(checkpoint)
    source = str(checkpoint)
    if not path.is_file():
        from huggingface_hub import HfApi, hf_hub_download
        repo = checkpoint
        files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".pt")]
        if not files:
            raise ValueError(f"{repo}: no .pt checkpoint in the repo")
        info = HfApi().model_info(repo)
        path = Path(hf_hub_download(repo, CLM_FILE if CLM_FILE in files else files[0]))
        source = f"{repo}@{info.sha}/{path.name}"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    for k in ("hidden_size", "projection_dim"):
        if k in ck:
            cfg[k] = ck[k]
    arrays = {f"{side}.{k}": v.float().numpy() for side in ("state_head", "action_head")
              for k, v in ck[side].items()}
    scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))
    if backbone is None:
        base = cfg.get("model", CLM_BACKBONE["torch"])
        backbone = {"torch": base, **({"mlx": CLM_BACKBONE["mlx"]} if base == CLM_BACKBONE["torch"] else {})}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / HEADS_FILE, cfg=np.array(json.dumps(cfg)), **arrays)
    write_spec(out, backbone, scale, source, {**CLM_READING, "max_tokens": max_tokens})
    return out


def is_heads_source(repo: str) -> bool:
    """A directory holding a contrastive.json, a `.pt` file, or a Hub repo whose config.json says
    `model_type: clm` (a published checkpoint `convert` reads)."""
    if Path(repo).is_dir():
        return (Path(repo) / SPEC_FILE).exists()
    if Path(repo).suffix == ".pt":
        return Path(repo).is_file()
    try:
        from huggingface_hub import hf_hub_download
        config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    except Exception:
        return False
    return config.get("model_type") == "clm"


def read_rows(path: str | Path) -> list[dict]:
    """Typed rows from a JSONL file or a parquet file (the decision dataset's columns)."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd
        return [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
                for r in pd.read_parquet(path).to_dict("records")]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m jul.contrastive",
                                 description="Contrastive heads for jul: convert a published checkpoint, or "
                                             "train heads on any backbone.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="a CLM-format .pt (or the Hub repo holding one) -> contrastive.json + heads.npz")
    c.add_argument("checkpoint", help=f"a .pt file or a Hub repo (e.g. {CLM_REPO})")
    c.add_argument("out")
    c.add_argument("--backbone-torch")
    c.add_argument("--backbone-mlx")
    c.add_argument("--max-tokens", type=int, default=2048)
    t = sub.add_parser("train", help="train heads on a frozen backbone from typed rows (JSONL or parquet)")
    t.add_argument("repo", help="the backbone: a Hub repo or a local directory jul loads")
    t.add_argument("rows", nargs="+", help="typed rows: {state, type, question, options, answer}")
    t.add_argument("out")
    t.add_argument("--backend")
    t.add_argument("--limit", type=int, default=0, help="rows kept per file (random), 0 = all")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--width", type=int)
    t.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        backbone = {k: v for k, v in (("torch", a.backbone_torch), ("mlx", a.backbone_mlx)) if v} or None
        print(convert(a.checkpoint, a.out, backbone, a.max_tokens))
        return
    from .backbone import Backbone, resolve_backend
    backend = resolve_backend(a.backend)
    rows = []
    for f in a.rows:
        got = read_rows(f)
        if a.limit and len(got) > a.limit:
            got = random.Random(0).sample(got, a.limit)
        rows += got
    bb = Backbone(a.repo, backend)
    report = train_heads(bb, rows, a.out, {backend: a.repo}, default_reading(bb, a.max_tokens),
                         width=a.width, epochs=a.epochs)
    print(f"{a.out}: held-out accuracy {report['heads_accuracy']:.3f} "
          f"(plain cosine {report['cosine_accuracy']:.3f}, {report['held_out']} rows)")


if __name__ == "__main__":
    main()
