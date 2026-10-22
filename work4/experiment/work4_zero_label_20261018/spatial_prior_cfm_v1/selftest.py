# -*- coding: utf-8 -*-
"""Synthetic checks for input interventions, source moments and prior semantics."""
from pathlib import Path
import json
import importlib.util
import numpy as np
import torch
from spatial_ops import condition, source_moments

B = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


ref = module('oldmodel', B.parent / 'train_stage0.py')
prior = module('oldprior', B.parent / 'gaussian_prior_cfm_v1/prior_ops.py')
torch.set_num_threads(4)
rng = np.random.RandomState(20261021)
x = torch.tensor(rng.normal(size=(7, 130)).astype(np.float32))
original = x.clone()
assert torch.equal(condition(x, 'all'), x)
assert torch.count_nonzero(condition(x, 'zero')) == 0
sat = condition(x, 'satellite')
assert torch.count_nonzero(sat[:, :2]) == 0
assert torch.equal(sat[:, 2:], x[:, 2:]) and torch.equal(x, original)
perturbed = x.clone()
perturbed[:, :2] += 999
assert torch.equal(condition(perturbed, 'satellite'), sat)
y = rng.uniform(size=(64, 2, 168)).astype(np.float32)
means, factors, errors = source_moments(y)
for c in range(2):
    bank = (y[:, c] * 2 - 1).astype(np.float64)
    # Independent np.cov population normalization.
    np.testing.assert_allclose(factors[c].astype(np.float64) @ factors[c].astype(np.float64).T,
                               np.cov(bank, rowvar=False, bias=True), atol=1e-7)
    np.testing.assert_allclose(means[c], bank.mean(0), atol=1e-7)
torch.manual_seed(1)
model = ref.CFM()
target = torch.tensor(y[:7, 0] * 2 - 1)
torch.manual_seed(23)
a = ref.loss(model, x, target)
torch.manual_seed(23)
b = prior.loss(model, x, target, torch.zeros(168), torch.eye(168))
assert torch.equal(a, b)
s0 = ref.generate(model, x, samples=3, steps=2, seed=97)
s1 = prior.generate(model, x, torch.zeros(168), torch.eye(168), seed=97, members=3, steps=2)
assert np.array_equal(s0, s1)
v = prior.loss(model, sat, target, torch.tensor(means[0]), torch.tensor(factors[0]))
v.backward()
assert torch.isfinite(v) and all(torch.isfinite(p.grad).all() for p in model.parameters())
with (B / 'SYNTHETIC_CHECK.json').open('x', encoding='utf-8') as f:
    json.dump({'status': 'PASS', 'synthetic_only': True, 'gpu_used': False,
               'checks': ['all input identity', 'zero input', 'satellite coordinate invariance',
                          'no mutation of original features', 'independent population covariance',
                          'identity prior recovers original loss and sampler exactly',
                          'finite gradients with nontrivial source prior'],
               'covariance_errors': errors}, f, indent=2)
print('SPATIAL_PRIOR_SYNTHETIC_PASS')
