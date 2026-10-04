"""The api backend: the vector reading over an embeddings endpoint, against a fake OpenAI-style server."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from jul import Choice, Noul, TypeSafeClient
from jul.backbone import Backbone
from jul.backends.api import APIBackbone, Embeddings, EmbeddingsError, parse_repo
from jul.presets import Formulation, Preset, save_preset

WORDS = {"billing": 0, "charged": 0, "refund": 0, "invoice": 0, "technical": 1, "crash": 1, "crashes": 1,
         "bug": 1, "error": 1, "yes": 2, "no": 3}


def fake_vector(text: str) -> list[float]:
    """A bag of known words, plus a little text-specific noise so that no two texts are equal."""
    v = np.zeros(8)
    for w in text.lower().replace(",", " ").replace(".", " ").replace(":", " ").split():
        if w in WORDS:
            v[WORDS[w]] += 1.0
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v[4:] = np.random.default_rng(seed).normal(0, 0.05, 4)
    return v.tolist()


@pytest.fixture
def server():
    seen, replies = [], []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            seen.append((self.path, self.headers.get("Authorization"), body))
            if replies:
                status, raw = replies.pop(0)
            else:
                rows = [{"object": "embedding", "index": i, "embedding": fake_vector(t)}
                        for i, t in reversed(list(enumerate(body["input"])))]   # out of order on purpose
                status, raw = 200, json.dumps({"object": "list", "data": rows, "model": body["model"]}).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1", seen, replies
    srv.shutdown()
    srv.server_close()


def test_parse_repo(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert parse_repo("ollama:qwen3-embedding:0.6b") == ("http://localhost:11434/v1", "qwen3-embedding:0.6b", None)
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.2:11434")
    assert parse_repo("ollama:nomic")[0] == "http://10.0.0.2:11434/v1"
    assert parse_repo("openai:text-embedding-3-small") == ("https://api.openai.com/v1", "text-embedding-3-small",
                                                          "OPENAI_API_KEY")
    assert parse_repo("http://h:8000/v1/embeddings#bge") == ("http://h:8000/v1", "bge", None)
    for bad in ("nope:model", "openai", "openai:", "http://h/v1"):
        with pytest.raises(ValueError):
            parse_repo(bad)


def test_a_hosted_provider_needs_its_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Embeddings("openai:text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert Embeddings("openai:text-embedding-3-small").api_key == "sk-test"


def test_embeddings_are_returned_in_input_order(server):
    url, seen, _ = server
    out = Embeddings(f"{url}#m")(["billing", "crash"])
    assert out.shape == (2, 8) and out[0, 0] == 1 and out[1, 1] == 1
    assert seen[0][0] == "/v1/embeddings" and seen[0][2] == {"model": "m", "input": ["billing", "crash"]}
    assert seen[0][1] is None   # no key sent to a server that needs none


@pytest.mark.parametrize("status,raw,match", [
    (401, b'{"error": "bad key"}', "HTTP 401"),
    (200, b"not json", "not JSON"),
    (200, b'{"data": []}', "expected 1 embeddings"),
    (200, b'{"data": [{"embedding": "x"}]}', "not lists of numbers"),
    (200, b'{"data": [{"embedding": [1, NaN]}]}', "finite"),
])
def test_api_failures_are_embeddings_errors(server, status, raw, match):
    url, _, replies = server
    replies.append((status, raw))
    with pytest.raises(EmbeddingsError, match=match):
        Embeddings(f"{url}#m")(["x"])


def test_the_key_never_appears_in_an_error(server, monkeypatch):
    url, seen, replies = server
    monkeypatch.setitem(__import__("jul.backends.api", fromlist=["PROVIDERS"]).PROVIDERS, "fake",
                        (url, "FAKE_KEY"))
    monkeypatch.setenv("FAKE_KEY", "sk-secret")
    replies.append((500, b"boom"))
    with pytest.raises(EmbeddingsError) as e:
        Embeddings("fake:m")(["x"])
    assert "sk-secret" not in str(e.value) and seen[0][1] == "Bearer sk-secret"


def test_backbone_reads_one_layer_and_sends_whole_prompts(server):
    url, seen, _ = server
    bb = Backbone(f"{url}#m", "api")
    assert isinstance(bb, APIBackbone) and bb.n_layers == 1 and bb.layer_indices() == [0]
    feats = bb.forward_batch([bb.encode("crash")], layers=[0], prefix=bb.cache_prefix(bb.encode("ticket: ")))
    assert seen[-1][2]["input"] == ["ticket: crash"]
    assert feats[0][0].shape == (16,)
    with pytest.raises(NotImplementedError, match="letters"):
        bb.forward(bb.encode("x"), layers=[0], logits=True)
    with pytest.raises(ValueError, match="one layer"):
        bb.forward_batch([bb.encode("x")], layers=[3])


def test_a_choice_and_a_noul_through_the_client(server, tmp_path, monkeypatch):
    url, seen, _ = server
    monkeypatch.setenv("JUL_HOME", str(tmp_path))
    repo = f"{url}#m"
    t = "{state}"
    preset = Preset(name="fake-api", repo="", api_repo=repo, backend="api",
                    formulations=(Formulation("one_word", t, 0),), tau=0.1, center="options",
                    latency_ms="?", quality="test")
    import jul.presets as P
    monkeypatch.setattr(P, "PRESET_HOME", tmp_path / "presets")
    save_preset(preset, tmp_path / "presets")
    client = TypeSafeClient(model="fake-api", backend="api")
    q = {"team": Choice(instructions="Which team?", criteria={"billing": "", "technical": ""})}
    assert client.system_one("I was charged twice, refund", q).answers["team"].choice == "billing"
    assert client.system_one("the app crashes with an error", q).answers["team"].choice == "technical"
    calls = len(seen)
    client.system_one("another crash", q)
    assert len(seen) == calls + 1   # the options are embedded once per question, a state costs one call
    r = client.system_one("yes", {"ok": Noul(instructions="Is it ok?")})
    assert 0.0 <= r.answers["ok"].noul <= 1.0


def test_long_inputs_are_sent_in_small_requests(server):
    url, seen, _ = server
    out = Embeddings(f"{url}#m", batch=2)(["billing", "crash", "refund", "bug", "yes"])
    assert out.shape == (5, 8) and [len(b["input"]) for _, _, b in seen] == [2, 2, 1]
    assert out[2, 0] == 1 and out[3, 1] == 1


# --- review fixes -----------------------------------------------------------------------------------------

def test_rows_that_are_not_objects_are_an_embeddings_error(server):
    url, _, replies = server
    replies.append((200, b'{"data": [1]}'))
    with pytest.raises(EmbeddingsError, match="not objects"):
        Embeddings(f"{url}#m")(["x"])


def test_an_echoed_key_is_masked(server, monkeypatch):
    import jul.backends.api as api
    url, _, replies = server
    monkeypatch.setitem(api.PROVIDERS, "fake", (url, "FAKE_KEY"))
    monkeypatch.setenv("FAKE_KEY", "sk-secret")
    replies.append((401, b'{"error": "bad Authorization: Bearer sk-secret"}'))
    with pytest.raises(EmbeddingsError) as e:
        Embeddings("fake:m")(["x"])
    assert "sk-secret" not in str(e.value) and "***" in str(e.value)


def test_long_and_empty_texts_are_bounded(server):
    url, seen, _ = server
    emb = Embeddings(f"{url}#m")
    emb.max_chars = 10
    with pytest.warns(UserWarning, match="JUL_API_MAX_CHARS"):
        emb(["x" * 50, ""])
    assert seen[-1][2]["input"] == ["x" * 10, " "]


def test_letters_say_why_they_cannot_run(server):
    url, _, _ = server
    bb = Backbone(f"{url}#m", "api")
    from jul.backbone import PromptTemplate
    with pytest.raises(NotImplementedError, match="letters reading"):
        PromptTemplate.from_user_message(bb, "{input}")
