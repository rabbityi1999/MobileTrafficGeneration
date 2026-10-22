# -*- coding: utf-8 -*-
from pathlib import Path
import os
import time
import json
import hashlib
import numpy as np
import torch
from model_reference import CFM
from prior_ops import loss, generate
from spatial_ops import load_packet, condition, ARMS

B = Path(__file__).resolve().parent
CH = ['call_in', 'call_out']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, d):
    with p.open('x', encoding='utf-8') as f:
        json.dump(d, f, indent=2)


def main():
    seal = json.loads((B / 'PACKAGE_SEAL.json').read_text(encoding='utf-8'))
    for n, h in seal['files'].items():
        assert sha(B / n) == h, n
    config = json.loads((B / 'CONFIG.json').read_text(encoding='utf-8'))
    assert config['target_label_budget'] == 0 and config['training']['new_fits'] == 24
    out = B / 'run'
    out.mkdir(exist_ok=False)
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    write(out / 'STARTED.json', {
        'pid': os.getpid(), 'gpu_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'time_unix': time.time(), 'package_sha256': sha(B / 'PACKAGE_SEAL.json'),
        'torch': torch.__version__, 'target_data_read': False, 'query_truth_deployed': False})
    records, checkpoints = [], []
    for fold in range(4):
        d = load_packet(B / ('fold%d.npz' % fold))
        x = torch.tensor(d['x_outer_train'], device=device)
        xq = torch.tensor(d['x_query'], device=device)
        for c, ch in enumerate(CH):
            mean = torch.tensor(d['means'][c], device=device)
            factor = torch.tensor(d['factors'][c], device=device)
            target = torch.tensor(d['y_outer_train'][:, c] * 2 - 1, device=device)
            for arm in ARMS:
                # Reset all training streams: sampling never advances training RNG.
                torch.manual_seed(1)
                torch.cuda.manual_seed_all(1)
                rng = np.random.RandomState(330001)
                model = CFM().to(device)
                assert sum(p.numel() for p in model.parameters()) == 167016
                opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
                xx, query = condition(x, arm), condition(xq, arm)
                started = time.monotonic()
                for step in range(1, 20001):
                    model.train()
                    idx = torch.tensor(rng.choice(len(x), 128, replace=False), device=device)
                    opt.zero_grad(set_to_none=True)
                    v = loss(model, xx[idx], target[idx], mean, factor)
                    assert torch.isfinite(v)
                    v.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    assert torch.isfinite(norm)
                    opt.step()
                    if step % 200 == 0:
                        with (out / 'history.jsonl').open('a', encoding='utf-8') as f:
                            f.write(json.dumps({'fold': fold, 'channel': ch, 'arm': arm,
                                                'step': step, 'loss': float(v.detach()),
                                                'training_seconds': time.monotonic() - started}) + '\n')
                stem = 'fold%d__%s__%s' % (fold, ch, arm)
                name = stem + '.pt'
                torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, out / name)
                checkpoints.append({'fold': fold, 'channel': ch, 'arm': arm, 'file': name,
                                    'sha256': sha(out / name), 'steps': 20000, 'parameters': 167016,
                                    'training_seconds': time.monotonic() - started})
                for seed in [6101901, 6101902, 6101903]:
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    parts = [generate(model, query[i:i+16], mean, factor, seed+i)
                             for i in range(0, 128, 16)]
                    torch.cuda.synchronize()
                    seconds = time.monotonic() - t0
                    s = np.concatenate(parts)
                    assert s.shape == (128, 200, 168) and np.isfinite(s).all()
                    name = stem + '__seed%d.npz' % seed
                    np.savez_compressed(out / name, samples=s, rows_query=d['rows_query'])
                    records.append({'fold': fold, 'channel': ch, 'arm': arm, 'seed': seed,
                                    'file': name, 'sha256': sha(out / name),
                                    'generation_seconds_excluding_serialization': seconds})
                print('SEALED_SPATIAL_ARM', fold, ch, arm, flush=True)
    assert len(records) == 72 and len(checkpoints) == 24
    write(out / 'COMPLETE.json', {
        'status': 'COMPLETE_72_FORECASTS_BEFORE_SCORING', 'records': records,
        'checkpoints': checkpoints, 'target_data_read': False,
        'outer_query_labels_read': False, 'model_selection': False})
    write(B / 'OUTPUT_SEAL.json', {'package_sha256': sha(B / 'PACKAGE_SEAL.json'),
                                 'time_unix': time.time(),
                                 'files': {p.name: sha(p) for p in out.iterdir()}})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(B / 'FAILED.json', {'error_type': type(exc).__name__, 'message': str(exc),
                                  'no_auto_retry': True})
        raise
