"""Run synthetic, CPU-only checks in disposable directories."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = list((ROOT/'work4').rglob('*.py'))
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    manifest = json.loads((ROOT/'work4/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        assert hashlib.sha256((ROOT/row['published']).read_bytes()).hexdigest() == row['sha256']
    with tempfile.TemporaryDirectory(prefix='mobiletraffic-check-') as tmp:
        base = Path(tmp)
        exp = base/'experiment'
        shutil.copytree(ROOT/'work4/experiment/work4_zero_label_20261018', exp,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'SYNTHETIC_CHECK.json'))
        commands = [
            [sys.executable, str(exp/'train_stage0.py'), '--smoke-output', str(base/'smoke.json')],
            [sys.executable, str(exp/'spatial_prior_cfm_v1/selftest.py')],
        ]
        objective = base/'objective_check.py'
        shutil.copyfile(ROOT/'work4/research/work4_cross_city_adaptation/ars_pipeline/phase1_scoping/objective_check.py', objective)
        commands.append([sys.executable, str(objective)])
        for command in commands:
            subprocess.run(command, check=True, cwd=base)
        import numpy as np
        spec = importlib.util.spec_from_file_location('metrics', exp/'spatial_prior_cfm_v1/assessment_v1/metrics.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        score, per = mod.score(np.zeros((2,168)), np.ones((2,3,168)))
        assert score['sample_crps'] == score['fair_crps'] == score['sample_mean_mae'] == 1
        assert score['coverage90'] == score['width90'] == score['variogram_mean'] == 0
        assert np.array_equal(per, np.ones(2))
    print(json.dumps({'status': 'PASS', 'syntax_files': len(sources), 'source_hashes': 'PASS',
                      'cfm_smoke': 'PASS', 'spatial_prior_checks': 'PASS',
                      'objective_checks': 29, 'metric_point_mass_identity': 'PASS',
                      'real_data_accessed': False, 'gpu_training_started': False}))


if __name__ == '__main__':
    main()
