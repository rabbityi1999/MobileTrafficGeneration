# -*- coding: utf-8 -*-
"""Role-limited fold operations; no evaluation labels or target data."""
import numpy as np

INPUT_KEYS = {'x_outer_train', 'y_outer_train', 'rows_outer_train', 'x_query', 'rows_query'}
ARMS = ['zero', 'satellite', 'all']


def condition(x, arm):
    assert arm in ARMS
    out = x.clone()
    if arm == 'zero':
        out.zero_()
    elif arm == 'satellite':
        out[:, :2] = 0
    return out


def source_moments(y):
    assert y.ndim == 3 and y.shape[1:] == (2, 168)
    means, factors, errors = [], [], []
    for c in range(2):
        bank = (y[:, c] * 2 - 1).astype(np.float64)
        mean = bank.mean(0)
        cov = (bank - mean).T @ (bank - mean) / len(bank)
        ev, vec = np.linalg.eigh(cov)
        assert ev.min() > -1e-10
        factor = (vec * np.sqrt(np.maximum(ev, 0))).astype(np.float32)
        error = float(np.max(np.abs(factor.astype(np.float64) @ factor.astype(np.float64).T - cov)))
        assert error < 1e-6
        means.append(mean.astype(np.float32))
        factors.append(factor)
        errors.append(error)
    return np.stack(means), np.stack(factors), errors


def load_packet(path):
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == INPUT_KEYS | {'means', 'factors'}
        d = {key: z[key].copy() for key in z.files}
    n = len(d['rows_outer_train'])
    assert d['x_outer_train'].shape == (n, 130)
    assert d['y_outer_train'].shape == (n, 2, 168)
    assert d['x_query'].shape == (128, 130)
    assert d['rows_query'].shape == (128,)
    assert d['means'].shape == (2, 168) and d['factors'].shape == (2, 168, 168)
    assert not np.intersect1d(d['rows_outer_train'], d['rows_query']).size
    assert all(np.isfinite(v).all() for v in d.values())
    assert d['y_outer_train'].min() >= 0 and d['y_outer_train'].max() <= 1
    return d
