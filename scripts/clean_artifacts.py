import os
from pathlib import Path

files = [
    Path('data/processed/train_processed_pipeline.csv'),
    Path('data/processed/test_processed_pipeline.csv'),
    Path('data/processed/test_predictions.csv'),
]
for f in files:
    try:
        if f.exists():
            f.unlink()
    except Exception:
        pass

# remove model pickle files
models_dir = Path('models')
if models_dir.exists():
    for p in models_dir.glob('*.pkl'):
        try:
            p.unlink()
        except Exception:
            pass

# delete files under reports but keep .gitkeep
reports = Path('reports')
if reports.exists():
    for p in reports.rglob('*'):
        if p.is_file() and p.name != '.gitkeep':
            try:
                p.unlink()
            except Exception:
                pass

# delete files under reports/figures but keep .gitkeep
figs = Path('reports/figures')
if figs.exists():
    for p in figs.rglob('*'):
        if p.is_file() and p.name != '.gitkeep':
            try:
                p.unlink()
            except Exception:
                pass

print('Clean complete.')
