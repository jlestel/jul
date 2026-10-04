"""The vector reading over an embeddings API: any model, local or hosted, without its weights.

The other backends read a hidden state inside the model. An embeddings API only returns its final
vector, so the model is read as an encoder is (jul/encoder.py): the state's vector is the embedding of
the prompt that holds it, the options' are the embeddings of the options, and the decision is their
centered cosine, with the layers, center and tau that `jul models add` fits for this model:

    jul models add qwen3-emb --repo ollama:qwen3-embedding:0.6b --backend api
    jul models add oai-small --repo openai:text-embedding-3-small --backend api
    jul ask choice "Which team?" -o billing -o technical --state "I was charged twice" \
        --model qwen3-emb --backend api

The repo names the provider and its model, `provider:model`:

| provider | endpoint                                    | key            |
| ollama   | $OLLAMA_HOST or http://localhost:11434/v1   | none           |
| openai   | https://api.openai.com/v1                   | OPENAI_API_KEY |
| mistral  | https://api.mistral.ai/v1                   | MISTRAL_API_KEY |
| voyage   | https://api.voyageai.com/v1                 | VOYAGE_API_KEY |

or any OpenAI-compatible server, `http(s)://host/v1#model` (no key). Every one is called on
`POST {endpoint}/embeddings` with `{"model", "input": [texts]}`.

What changes against a model read locally:
- one layer: the API's output. `jul models add` fits the center and tau only;
- no prefix cache: the whole prompt is sent with each text, and every state costs one API call per
  formulation (the option vectors are computed once per question, as on the other backends);
- no logits: Noul and Score are read as vectors (their default), never with letters;
- the token counts in `usage` are characters: the API's tokenizer is not known here;
- the state leaves the machine for a hosted provider;
- texts go in requests of JUL_API_BATCH (16), each cut to JUL_API_MAX_CHARS (8000) characters, with a
  timeout of JUL_API_TIMEOUT (300) seconds. Voyage's `input_type` is not sent.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import warnings

import numpy as np

from ..backbone import Backbone

#: provider -> (base URL, environment variable holding its key or None)
PROVIDERS = {
    "ollama": (None, None),   # resolved at call time from OLLAMA_HOST
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "voyage": ("https://api.voyageai.com/v1", "VOYAGE_API_KEY"),
}


class EmbeddingsError(RuntimeError):
    """The embeddings API failed or answered something else than embeddings. Never holds the key."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # the key must never follow a redirect to another host
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _ollama_base() -> str:
    host = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/") + "/v1"


def parse_repo(repo: str) -> tuple[str, str, str | None]:
    """(base URL, model, key variable) from `provider:model` or `http(s)://host/v1#model`."""
    if repo.startswith(("http://", "https://")):
        url, sep, model = repo.partition("#")
        if not sep or not model:
            raise ValueError(f"{repo!r}: name the model after the URL, as in http://host/v1#my-model")
        url = url.rstrip("/")
        return url.removesuffix("/embeddings"), model, None
    provider, sep, model = repo.partition(":")
    if provider not in PROVIDERS or not sep or not model:
        raise ValueError(f"{repo!r}: expected provider:model with a provider among {', '.join(PROVIDERS)}, "
                         "or http(s)://host/v1#model")
    base, key_env = PROVIDERS[provider]
    return base or _ollama_base(), model, key_env


class Embeddings:
    """`POST {base}/embeddings`, OpenAI's format, which Ollama, Mistral and Voyage also speak."""

    def __init__(self, repo: str, timeout: float | None = None, batch: int | None = None):
        self.base, self.model, self.key_env = parse_repo(repo)
        self.api_key = os.environ.get(self.key_env) if self.key_env else None
        if self.key_env and not self.api_key:
            raise ValueError(f"{repo}: set {self.key_env}")
        # a local server on CPU can take minutes on a batch of long texts: generous defaults, small requests
        self.timeout = timeout or float(os.environ.get("JUL_API_TIMEOUT") or 300)
        self.batch = batch or int(os.environ.get("JUL_API_BATCH") or 16)
        #: longest text sent, in characters (~2,000 tokens): the end of a longer state is cut, with a warning
        self.max_chars = int(os.environ.get("JUL_API_MAX_CHARS") or 8000)

    def __call__(self, texts: list[str]) -> np.ndarray:
        """(len(texts), d), sent in requests of at most `batch` texts."""
        texts = [self._fit(t) for t in texts]
        if len(texts) <= self.batch:
            return self._post(texts)
        return np.concatenate([self._post(texts[i:i + self.batch]) for i in range(0, len(texts), self.batch)])

    def _fit(self, text: str) -> str:
        if not text.strip():
            return " "   # OpenAI rejects empty inputs; a blank reads as "no content" everywhere
        if len(text) > self.max_chars:
            warnings.warn(f"a text of {len(text)} characters is over JUL_API_MAX_CHARS={self.max_chars}: "
                          "its end is cut", stacklevel=4)
            return text[: self.max_chars]
        return text

    def _mask(self, text: str) -> str:
        """A server may echo the Authorization header in its error: the key is never repeated."""
        return text.replace(self.api_key, "***") if self.api_key else text

    def _post(self, texts: list[str]) -> np.ndarray:
        url = self.base + "/embeddings"
        body = json.dumps({"model": self.model, "input": texts}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with _opener.open(urllib.request.Request(url, body, headers), timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = self._mask(e.read()[:300].decode("utf-8", "replace"))
            raise EmbeddingsError(f"{url}: HTTP {e.code} {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise EmbeddingsError(f"{url}: {getattr(e, 'reason', e)}") from None
        except ValueError:
            raise EmbeddingsError(f"{url}: the response is not JSON") from None
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise EmbeddingsError(f"{url}: expected {len(texts)} embeddings, got {str(data)[:200]}")
        if not all(isinstance(r, dict) for r in rows):
            raise EmbeddingsError(f"{url}: the embeddings are not objects")
        rows = sorted(rows, key=lambda r: r.get("index", 0) if isinstance(r.get("index", 0), int) else 0)
        try:
            out = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            raise EmbeddingsError(f"{url}: the embeddings are not lists of numbers") from None
        if out.ndim != 2 or not np.isfinite(out).all():
            raise EmbeddingsError(f"{url}: the embeddings are not finite vectors of one size")
        return out


class _Chars:
    """A tokenizer over characters: prompts are built from text and sent as text, so the template
    machinery (prefix, suffix, pools) works on characters and the API sees the joined string."""

    chat_template = None
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(map(chr, ids))

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError("An embeddings API has no chat template nor logits: the letters reading "
                                  "needs the mlx or torch backend (Noul and Score are read as vectors by default)")


class APIBackbone(Backbone):
    backend = "api"
    architecture = "embedding"
    n_layers = 1

    def __init__(self, name: str, backend: str | None = None, embed: Embeddings | None = None):
        super().__init__(name)
        self.embed = embed or Embeddings(self.repo)
        self.tokenizer = _Chars()
        self.text_prefix = ""

    def layer_indices(self, fractions=()) -> list[int]:
        return [0]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def forward(self, tokens, layers=(), logits=False, pool=None, prefix=None):
        if logits:
            raise NotImplementedError("An embeddings API has no logits: the letters reading needs the mlx or "
                                      "torch backend")
        return self.forward_batch([tokens], layers, [pool], prefix)[0], None

    def forward_batch(self, queries, layers=(), pools=None, prefix=None):
        if not layers:
            return [{} for _ in queries]
        if set(layers) - {0}:
            raise ValueError(f"an embeddings API has one layer, 0; asked for {sorted(layers)}")
        head = list(prefix or [])
        vectors = self.embed([self.tokenizer.decode(head + list(q)) for q in queries])
        # [sequence ; pool], the layout of the other backends: the vector reading keeps the first half
        return [{0: np.concatenate([v, v])} for v in vectors]

    def cache_prefix(self, tokens) -> list[int]:
        return list(tokens)

    def last_hidden(self, tokens, prefix=None):
        raise NotImplementedError("The pointer method reads a decoder's weights, not an embeddings API")

    @property
    def model_dir(self):
        raise NotImplementedError("An embeddings API has no local weights")
