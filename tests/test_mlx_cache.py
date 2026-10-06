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
