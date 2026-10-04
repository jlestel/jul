"""Escalation: unsure answers go to the next decider, and the trace says who answered."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from jul import Choice, Escalation, Noul, NoulCriteria, Score, SystemOneHTTP
from jul.escalate import certainty
from jul.types import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage


class Fixed:
    """A decider answering every question with the same confidence; records what it was asked."""

    def __init__(self, confidence, choice="a", noul=None, fail=False):
        self.confidence, self.choice, self.noul, self.fail = confidence, choice, noul, fail
        self.asked = []

    def system_one(self, state, questions, **kwargs):
        self.asked.append((sorted(questions), kwargs))
        if self.fail:
            raise ConnectionError("down")
        answers = {}
        for name, q in questions.items():
            if isinstance(q, Noul):
                answers[name] = NoulAnswer(self.noul if self.noul is not None else self.confidence)
            elif isinstance(q, Score):
                answers[name] = ScoreAnswer(1.0, {}, {"0": 0.0, "1": 1.0}, self.confidence)
            else:
                answers[name] = ChoiceAnswer(self.choice, {self.choice: self.confidence}, self.confidence)
        return SystemOneResponse(answers=answers, model="fixed", usage=Usage(input_tokens=3), request_id="r")


Q = {"team": Choice("Which team?", {"a": "A", "b": "B"}), "urgent": Noul("Urgent?")}


def test_a_sure_answer_stays_local():
    local, remote = Fixed(0.95), Fixed(0.99, choice="b")
    r = Escalation([("local", local), ("remote", remote)], min_confidence=0.8).system_one("s", Q)
    assert r.choices["team"].choice == "a"
    assert remote.asked == []
    assert r.escalation["team"] == {"tried": ["local"], "tier": "local", "confidence": 0.95, "met_bar": True}


def test_only_unsure_questions_escalate():
    local = Fixed(0.6, noul=0.97)            # unsure on the choice, sure the noul is true
    remote = Fixed(0.9, choice="b")
    r = Escalation([("local", local), ("remote", remote)], min_confidence=0.8).system_one("s", Q)
    assert remote.asked[0][0] == ["team"]
    assert r.choices["team"].choice == "b" and r.escalation["team"]["tier"] == "remote"
    assert r.escalation["urgent"]["tier"] == "local"
    assert r.usage.input_tokens == 6


def test_noul_certainty_is_the_likelier_side():
    assert certainty(NoulAnswer(0.03)) == pytest.approx(0.97)
    assert certainty(NoulAnswer(0.5)) == 0.5


def test_the_last_tier_answers_even_when_unsure_and_says_so():
    r = Escalation([("local", Fixed(0.5)), ("remote", Fixed(0.6))], min_confidence=0.8).system_one("s", Q)
    assert r.escalation["team"]["tier"] == "remote" and r.escalation["team"]["met_bar"] is False


def test_the_escalated_tier_wins_even_less_confident():
    """Confidences of two models are not comparable: the tier escalated to is the one trusted."""
    r = Escalation([("local", Fixed(0.7)), ("remote", Fixed(0.4, choice="b"))], min_confidence=0.8).system_one("s", Q)
    assert r.choices["team"].choice == "b" and r.escalation["team"]["tried"] == ["local", "remote"]


def test_a_failing_last_tier_keeps_the_earlier_answer():
    r = Escalation([("local", Fixed(0.7)), ("jev", Fixed(0, fail=True))], min_confidence=0.8).system_one("s", Q)
    assert r.choices["team"].choice == "a" and r.escalation["team"]["tier"] == "local"
    assert r.escalation["team"]["met_bar"] is False


def test_a_failing_tier_is_skipped_and_recorded():
    r = Escalation([("local", Fixed(0.5)), ("jev", Fixed(0, fail=True)), ("kev", Fixed(0.9, choice="b"))],
                   min_confidence=0.8).system_one("s", Q)
    assert r.choices["team"].choice == "b"
    assert r.escalation["team"]["tried"] == ["local", "jev", "kev"]
    assert r.escalation["team"]["errors"] == ["jev: ConnectionError"]


def test_no_tier_at_all_raises():
    with pytest.raises(RuntimeError, match="no tier answered"):
        Escalation([("a", Fixed(0, fail=True))]).system_one("s", Q)


def test_per_question_bars():
    local, remote = Fixed(0.85, noul=0.85), Fixed(0.99)
    Escalation([("local", local), ("remote", remote)],
               min_confidence={"urgent": 0.95}, default=0.8).system_one("s", Q)
    assert remote.asked[0][0] == ["urgent"]


def test_local_options_reach_the_first_tier_only():
    local, remote = Fixed(0.5), Fixed(0.9)
    Escalation([("local", local), ("remote", remote)]).system_one("s", Q, method="letters")
    assert local.asked[0][1] == {"method": "letters"} and remote.asked[0][1] == {}


def test_response_serializes_with_the_trace():
    out = Escalation([("local", Fixed(0.95))]).system_one("s", Q).as_dict()
    assert out["jul"]["escalation"]["team"]["tier"] == "local"
    assert json.dumps(out)


# --- SystemOneHTTP against a fake Jev server -------------------------------------------------------

@pytest.fixture
def jev_server():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen["path"], seen["auth"] = self.path, self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = {"model": "jev-1.13.0", "usage": {"input_tokens": 42, "output_tokens": 0},
                   "answers": {"team": {"type": "choice", "choice": "b", "probabilities": {"a": 0.1, "b": 0.9},
                                        "confidence": 0.88},
                               "urgent": {"type": "noul", "noul": 0.12},
                               "level": {"type": "score", "score": 1.7, "legend": {"0": "low", "1": "mid", "2": "high"},
                                         "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75}, "confidence": 0.6}}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()
    srv.server_close()


def test_http_tier_speaks_the_jev_protocol(jev_server):
    url, seen = jev_server
    questions = {"team": Choice("Which team?", {"a": "A", "b": ""}),
                 "urgent": Noul("Urgent?", NoulCriteria(true="now", false="later")),
                 "level": Score("How bad?", ["low", "mid", "high"])}
    r = SystemOneHTTP(url + "/v1", api_key="k").system_one({"ticket": "x"}, questions)
    assert seen["path"] == "/v1/systemone" and seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "jev-latest" and seen["body"]["state"] == {"ticket": "x"}
    assert seen["body"]["questions"]["team"] == {"type": "choice", "instructions": "Which team?",
                                                 "criteria": {"a": "A", "b": None}}
    assert seen["body"]["questions"]["urgent"]["criteria"] == {"true": "now", "false": "later"}
    assert seen["body"]["questions"]["level"]["criteria"] == ["low", "mid", "high"]
    assert r.choices["team"].choice == "b" and r.nouls["urgent"].noul == 0.12
    assert r.scores["level"].confidence == 0.6 and r.usage.input_tokens == 42 and r.model == "jev-1.13.0"


def test_a_noul_without_criteria_sends_none(jev_server):
    url, seen = jev_server
    SystemOneHTTP(url).system_one("s", {"urgent": Noul("Urgent?")})
    assert "criteria" not in seen["body"]["questions"]["urgent"]


def test_local_then_http(jev_server):
    url, _ = jev_server
    r = Escalation([("local", Fixed(0.5)), ("jev", SystemOneHTTP(url))]).system_one("s", {"team": Q["team"]})
    assert r.choices["team"].choice == "b" and r.escalation["team"]["tier"] == "jev"


# --- review fixes ------------------------------------------------------------------------------------

class Refuses(Fixed):
    def system_one(self, state, questions, **kwargs):
        raise ValueError("Unknown model 'nope'")


def test_a_malformed_request_is_raised_not_escalated():
    remote = Fixed(0.9)
    with pytest.raises(ValueError, match="nope"):
        Escalation([("local", Refuses(0)), ("remote", remote)]).system_one("s", Q)
    assert remote.asked == []


def test_nan_is_unsure_and_the_trace_stays_json():
    r = Escalation([("local", Fixed(float("nan"))), ("remote", Fixed(0.9))]).system_one("s", {"team": Q["team"]})
    assert r.escalation["team"]["tier"] == "remote"
    json.loads(json.dumps(r.as_dict(), allow_nan=False))


def test_met_bar_uses_the_raw_confidence():
    r = Escalation([("local", Fixed(0.79996))], min_confidence=0.8).system_one("s", {"team": Q["team"]})
    assert r.escalation["team"]["met_bar"] is False


def test_url_forms():
    for u in ("http://h", "http://h/", "http://h/v1", "http://h/v1/systemone"):
        assert SystemOneHTTP(u).url == "http://h/v1/systemone"


def test_provider_shorthands(monkeypatch):
    from jul.escalate import remote_tier
    monkeypatch.setenv("TYPESAFE_API_KEY", "tk")
    t = remote_tier("typesafe")
    assert t.url == "https://api.typesafe.ai/v1/systemone" and t.model == "jev-latest" and t.api_key == "tk"
    o = remote_tier("ollama:clef-flash")
    assert o.url == "http://localhost:11434/v1/systemone" and o.model == "clef-flash" and o.api_key is None
    monkeypatch.setenv("KEV_KEY", "kk")
    k = remote_tier("http://kev:8009", "kev-latest", key_env="KEV_KEY")
    assert k.model == "kev-latest" and k.api_key == "kk"


@pytest.fixture
def raw_server():
    """Answers each POST with the (status, body, headers) queued in `replies`."""
    replies, seen = [], []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self):
            seen.append((self.path, self.headers.get("Authorization")))
            status, body, headers = replies.pop(0)
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._reply()

        do_GET = _reply

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", replies, seen
    srv.shutdown()
    srv.server_close()


@pytest.mark.parametrize("status,body,match", [
    (401, b'{"error": "bad key"}', "401.*bad key"),
    (200, b"not json", "JSONDecodeError"),
    (200, b"[1, 2]", "not a JSON object"),
    (200, b'{"answers": {"team": {"type": "choice"}}}', "malformed answer"),
    (200, b'{"answers": {"team": {"type": "noul", "noul": 0.5}}}', "wrong type"),
])
def test_http_failures_are_remote_errors(raw_server, status, body, match):
    from jul.escalate import RemoteError
    url, replies, _ = raw_server
    replies.append((status, body, {}))
    with pytest.raises(RemoteError, match=match):
        SystemOneHTTP(url, api_key="SECRET").system_one("s", {"team": Q["team"]})


def test_a_redirect_is_not_followed(raw_server):
    from jul.escalate import RemoteError
    url, replies, seen = raw_server
    replies.append((302, b"", {"Location": url + "/elsewhere"}))
    with pytest.raises(RemoteError, match="302"):
        SystemOneHTTP(url, api_key="SECRET").system_one("s", {"team": Q["team"]})
    assert len(seen) == 1


# --- Clef on Cloudflare Workers AI ---------------------------------------------------------------------

CLEF_RESULT = {"model": "clef-flash", "usage": {"input_tokens": 57, "output_tokens": 0},
               "answers": {"team": {"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.8},
                                    "confidence": 0.8},
                           "urgent": {"type": "noul", "noul": 0.93}}}


def test_cloudflare_tier_url_and_keys(monkeypatch):
    from jul.escalate import remote_tier
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cft")
    t = remote_tier("cloudflare:clef")
    assert t.url == "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef"
    assert t.model == "clef" and t.api_key == "cft"
    assert remote_tier("cloudflare").model == "clef-flash"


def test_cloudflare_tier_refuses_unknown_models_and_a_missing_account(monkeypatch):
    from jul.escalate import cloudflare_tier
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        cloudflare_tier("clef")
    with pytest.raises(ValueError, match="decision models"):
        cloudflare_tier("llama-3", account_id="acc")


def test_cloudflare_envelope_is_unwrapped(raw_server):
    url, replies, seen = raw_server
    replies.append((200, json.dumps({"result": CLEF_RESULT, "success": True, "errors": [],
                                     "messages": []}).encode(), {}))
    route = url + "/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef-flash"
    r = SystemOneHTTP(route, model="clef-flash", api_key="cft", exact_url=True).system_one("s", Q)
    assert seen[0] == ("/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef-flash", "Bearer cft")
    assert r.choices["team"].choice == "b" and r.nouls["urgent"].noul == 0.93
    assert r.model == "clef-flash" and r.usage.input_tokens == 57


def test_cloudflare_failure_is_a_remote_error(raw_server):
    from jul.escalate import RemoteError
    url, replies, _ = raw_server
    replies.append((200, json.dumps({"result": None, "success": False,
                                     "errors": [{"code": 5006, "message": "bad input"}]}).encode(), {}))
    with pytest.raises(RemoteError, match="bad input"):
        SystemOneHTTP(url + "/run", exact_url=True).system_one("s", Q)


def test_cloudflare_as_an_escalation_tier(raw_server):
    url, replies, _ = raw_server
    replies.append((200, json.dumps({"result": CLEF_RESULT, "success": True}).encode(), {}))
    clef = SystemOneHTTP(url + "/run", model="clef-flash", exact_url=True)
    r = Escalation([("local", Fixed(0.5)), ("clef", clef)], min_confidence=0.8).system_one("s", Q)
    assert r.escalation["team"]["tier"] == "clef" and r.escalation["team"]["met_bar"] is True
