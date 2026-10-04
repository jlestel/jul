"""Escalation: answer locally, and hand only the unsure questions to the next decider.

    from jul import TypeSafeClient, Escalation, SystemOneHTTP
    client = Escalation(
        tiers=[("local", TypeSafeClient()),                                  # JuL, on this machine
               ("jev", SystemOneHTTP("https://api.typesafe.ai", api_key=...))],  # or Nimble, Kev...
        min_confidence=0.8)
    response = client.system_one(state=..., questions=...)
    response.escalation["team"]   # {"tier": "local", "confidence": 0.93, "tried": ["local"]}

Each tier is anything with a `system_one(state, questions, ...)` returning a `SystemOneResponse`: a
local `TypeSafeClient`, or `SystemOneHTTP` for any server speaking `POST /v1/systemone` (Jev, Ollama's
Nimble, Kev, another `jul serve`). The first tier answers every question; a question whose answer is
below the bar goes to the next tier, alone with the other unsure ones, and so on. The answer kept is
the one from the last tier that answered, whatever its confidence (confidences of different models are
not comparable), so every question gets one; `met_bar` in the trace says whether it cleared the bar.

How sure an answer is, per type:

- Choice and Score: their `confidence`;
- Noul: `max(noul, 1 - noul)`, the probability of the more likely side (a noul at 0.5 is a coin flip).

`min_confidence` is one bar for every question, or a map from question name to its own bar (a question
absent from the map uses `default`). A tier that fails (network, HTTP error, refusal) is skipped, and
its questions go to the next one; the failure is recorded in the trace, never raised, unless no tier
could answer at all.

The bar is a policy, not a measurement: a confidence is not an accuracy. Measure what a given bar
lets through on labeled examples of your own before relying on it.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
import uuid
from typing import Any, Mapping, Sequence

from .types import (Choice, ChoiceAnswer, Noul, NoulAnswer, NoulCriteria, Question, Score, ScoreAnswer,
                    SystemOneResponse, Usage, options_of, serialize_state)

__all__ = ["Escalation", "EscalatedResponse", "RemoteError", "SystemOneHTTP", "certainty"]


def certainty(answer: Any) -> float:
    """How sure an answer is, in [0, 1]: `confidence`, or the likelier side of a Noul."""
    value = (max(answer.noul, 1.0 - answer.noul) if isinstance(answer, NoulAnswer)
             else float(answer.confidence))
    return value if math.isfinite(value) else 0.0  # a NaN is unsure, never a pass


class EscalatedResponse(SystemOneResponse):
    """A `SystemOneResponse` plus, per question, which tier answered and which were tried."""

    def __init__(self, answers, model, usage, request_id, escalation):
        super().__init__(answers=answers, model=model, usage=usage, request_id=request_id)
        self.escalation: dict[str, dict] = escalation

    def as_dict(self) -> dict:
        out = super().as_dict()
        out["jul"] = {"escalation": self.escalation}
        return out


class Escalation:
    """Several deciders in order; a question goes on to the next one only while its answer is unsure."""

    def __init__(self, tiers: Sequence[tuple[str, Any]], min_confidence: float | Mapping[str, float] = 0.8,
                 default: float = 0.8):
        if not tiers:
            raise ValueError("Escalation needs at least one tier")
        names = [n for n, _ in tiers]
        if len(set(names)) != len(names):
            raise ValueError(f"tier names must be unique: {names}")
        self.tiers = list(tiers)
        self.min_confidence = min_confidence
        self.default = default

    def bar(self, name: str) -> float:
        if isinstance(self.min_confidence, Mapping):
            return float(self.min_confidence.get(name, self.default))
        return float(self.min_confidence)

    @property
    def model(self) -> str:
        return "+".join(n for n, _ in self.tiers)

    def system_one(self, state: Any, questions: Mapping[str, Question], **kwargs: Any) -> EscalatedResponse:
        if not questions:
            raise ValueError("system_one needs at least one question")
        pending = dict(questions)
        answers: dict[str, Any] = {}
        trace: dict[str, dict] = {n: {"tried": []} for n in questions}
        raw: dict[str, float] = {}
        tokens = 0
        errors: list[str] = []
        last = len(self.tiers) - 1

        for i, (tier, decider) in enumerate(self.tiers):
            if not pending:
                break
            try:
                # Options such as `method` or `context` mean something to a local client only.
                extra = kwargs if i == 0 else {}
                response = decider.system_one(state=state, questions=pending, **extra)
            except (ValueError, TypeError):
                # A malformed request (unknown model, a Score with one level...) is the caller's error:
                # raise it rather than ship the state to the next tier.
                raise
            except Exception as e:  # noqa: BLE001 - a failing tier is skipped, never fatal by itself
                errors.append(f"{tier}: {type(e).__name__}: {e}")
                for name in pending:
                    trace[name]["tried"].append(tier)
                    trace[name].setdefault("errors", []).append(f"{tier}: {type(e).__name__}")
                continue
            tokens += response.usage.input_tokens
            still: dict[str, Question] = {}
            for name, question in pending.items():
                trace[name]["tried"].append(tier)
                answer = response.answers.get(name)
                if answer is None:
                    still[name] = question
                    continue
                sure = certainty(answer)
                # The later tier's answer wins: confidences from different models are not comparable,
                # and a tier is escalated to because it is trusted more. A tier that fails keeps the last.
                answers[name] = answer
                raw[name] = sure
                trace[name].update(tier=tier, confidence=round(sure, 4))
                if sure < self.bar(name) and i < last:
                    still[name] = question
            pending = still

        missing = [n for n in questions if n not in answers]
        if missing:
            raise RuntimeError(f"no tier answered {missing}: " + "; ".join(errors))
        for name in questions:
            trace[name]["met_bar"] = raw[name] >= self.bar(name)
        return EscalatedResponse(answers={n: answers[n] for n in questions}, model=self.model,
                                 usage=Usage(input_tokens=tokens), request_id=str(uuid.uuid4()),
                                 escalation=trace)


# --- a remote decider over HTTP -------------------------------------------------------------------

def _question_payload(question: Question) -> dict:
    if isinstance(question, Noul):
        payload: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
        c = question.criteria
        given = (c.true or c.false) if isinstance(c, NoulCriteria) else bool(c)
        if given:  # the defaults ("Yes."/"No.") stay the receiving server's own
            options = {o.key: o.description for o in options_of(question)}
            payload["criteria"] = {"true": options["true"], "false": options["false"]}
        return payload
    if isinstance(question, Score):
        return {"type": "score", "instructions": question.instructions, "criteria": list(question.criteria)}
    if isinstance(question, Choice):
        return {"type": "choice", "instructions": question.instructions,
                "criteria": {o.key: (o.description or None) for o in options_of(question)}}
    raise TypeError(f"Questions must be Choice, Noul or Score (got {type(question).__name__})")


class RemoteError(RuntimeError):
    """A remote tier failed: network, HTTP status, or a response that is not a System One answer."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would carry the Authorization header to another host."""

    def redirect_request(self, *args, **kwargs):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _answer(data: Mapping) -> Any:
    kind = data.get("type")
    if kind == "noul":
        return NoulAnswer(noul=float(data["noul"]))
    if kind == "score":
        return ScoreAnswer(score=float(data["score"]), legend=dict(data.get("legend") or {}),
                           probabilities={str(k): float(v) for k, v in (data.get("probabilities") or {}).items()},
                           confidence=float(data.get("confidence") or 0.0))
    if kind == "choice":
        return ChoiceAnswer(choice=str(data["choice"]),
                            probabilities={str(k): float(v) for k, v in (data.get("probabilities") or {}).items()},
                            confidence=float(data.get("confidence") or 0.0))
    raise ValueError(f"unknown answer type {kind!r}")


#: `--escalate-to` shorthands: the server, its default model, and the variable its key is usually in.
PROVIDERS = {
    "typesafe": ("https://api.typesafe.ai", "jev-latest", "TYPESAFE_API_KEY"),
    "ollama": ("http://localhost:11434", "nimble", None),
}


def remote_tier(target: str, model: str | None = None, key_env: str | None = None,
                timeout: float = 30.0) -> SystemOneHTTP:
    """A tier from `typesafe`, `ollama:nimble` or a URL. The key is read from the provider's own variable
    (TYPESAFE_API_KEY for Jev), or from `key_env` for any other server; never passed as a value."""
    import os
    name, _, suffix = target.partition(":")
    if name in PROVIDERS and not target.startswith(("http://", "https://")):
        url, default_model, default_env = PROVIDERS[name]
        model = model or suffix or default_model
        key_env = key_env or default_env
    else:
        url, model = target, model or "jev-latest"
    key = os.environ.get(key_env) if key_env else None
    return SystemOneHTTP(url, model=model, api_key=key or None, timeout=timeout)


_ANSWER_TYPE = {Choice: ChoiceAnswer, Noul: NoulAnswer, Score: ScoreAnswer}


class SystemOneHTTP:
    """Any server speaking `POST /v1/systemone`: TypeSafe's Jev, Ollama (Nimble, Tev1), Kev, `jul serve`.

    `base_url` is the server root, with or without `/v1` (`https://api.typesafe.ai`,
    `http://localhost:11434`). The state is sent to that server: a remote tier is a third party.
    """

    def __init__(self, base_url: str, model: str = "jev-latest", api_key: str | None = None,
                 timeout: float = 30.0):
        root = base_url.rstrip("/")
        if root.endswith("/v1/systemone"):
            self.url = root
        else:
            self.url = (root[:-3] if root.endswith("/v1") else root) + "/v1/systemone"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def system_one(self, state: Any, questions: Mapping[str, Question], **_ignored: Any) -> SystemOneResponse:
        if not isinstance(state, (str, dict, list)):
            state = serialize_state(state)  # the text the local tier read, not a repr
        body = {"model": self.model, "state": state,
                "questions": {n: _question_payload(q) for n, q in questions.items()}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=json.dumps(body, default=str).encode(), headers=headers)
        try:
            with _opener.open(request, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:200].decode("utf-8", "replace")
            raise RemoteError(f"{self.url} answered HTTP {e.code}: {detail}") from e
        except (OSError, ValueError) as e:  # network, timeout, a body that is not JSON
            raise RemoteError(f"{self.url}: {type(e).__name__}: {e}") from e
        if not isinstance(data, dict):
            raise RemoteError(f"{self.url}: the response is not a JSON object")
        answers = {}
        for name, a in (data.get("answers") or {}).items():
            if name not in questions:
                continue
            try:
                answer = _answer(a)
            except (KeyError, TypeError, ValueError) as e:
                raise RemoteError(f"{self.url}: malformed answer for {name!r}: {e}") from e
            if _ANSWER_TYPE[type(questions[name])] is not type(answer):
                raise RemoteError(f"{self.url}: {name!r} was answered with the wrong type")
            answers[name] = answer
        usage = data.get("usage") or {}
        return SystemOneResponse(answers=answers, model=str(data.get("model") or self.model),
                                 usage=Usage(input_tokens=int(usage.get("input_tokens") or 0)),
                                 request_id=str(data.get("request_id") or uuid.uuid4()))
