"""The prefix cache of an MLX batch: copied once per row, whatever mlx-lm's `KVCache.state` returns."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx_lm") is None, reason="needs mlx-lm")


def test_a_prefix_cache_is_repeated_once_per_row():
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from jul.backends.mlx import _repeat_into

    prefix = KVCache()
    keys = mx.arange(1 * 2 * 5 * 4, dtype=mx.float32).reshape(1, 2, 5, 4)
    prefix.update_and_fetch(keys, keys + 1000)          # 5 tokens in a buffer allocated past them
    new = KVCache()
    _repeat_into(new, prefix, 3)
    assert new.offset == 5
    k, v = new.keys[..., : new.offset, :], new.values[..., : new.offset, :]
    assert k.shape == (3, 2, 5, 4)
    for row in range(3):
        assert np.array_equal(np.array(k[row]), np.array(keys[0]))
        assert np.array_equal(np.array(v[row]), np.array(keys[0] + 1000))


def test_a_query_never_writes_into_the_prefix_snapshot():
    """A hybrid model (Qwen3.5: Gated DeltaNet + attention) keeps a snapshot of its state after the prefix,
    restored for every query. The query's writes (ArraysCache `cache[1] = ...`, KVCache past the offset)
    must not reach the snapshot, or the next query starts from the previous one's state."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    from jul.backends.mlx import _restore

    arrays, kv = ArraysCache(2), KVCache()
    arrays[0], arrays[1] = mx.zeros((1, 3)), mx.ones((1, 3))
    x = mx.zeros((1, 1, 5, 2))
    kv.update_and_fetch(x, x)
    snapshot = [tuple(c.state) for c in (arrays, kv)]
    before = [np.array(a) for a in (arrays[0], arrays[1], kv.keys)]

    for _ in range(2):                                   # two queries on the same prefix
        restored = [ArraysCache(2), KVCache()]
        _restore(restored, snapshot)
        assert restored[1].offset == 5
        restored[0][0], restored[0][1] = mx.full((1, 3), 7.0), mx.full((1, 3), 9.0)
        y = mx.full((1, 1, 1, 2), 7.0)
        restored[1].update_and_fetch(y, y)

    restored = [ArraysCache(2), KVCache()]
    _restore(restored, snapshot)
    after = [np.array(a) for a in (restored[0][0], restored[0][1], restored[1].keys)]
    assert all(np.array_equal(a, b[..., : a.shape[-2], :] if a.ndim == 4 else b) for a, b in zip(after, before))
    assert np.array_equal(np.array(kv.keys), before[2])  # nothing written past the offset either
