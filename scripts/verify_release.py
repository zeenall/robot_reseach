#!/usr/bin/env python3
"""Lightweight integrity and syntax verifier. Standard library only."""
import csv,hashlib,sys,py_compile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'MANIFEST.sha256'
errors=[]
for entry in manifest.read_text(encoding='utf-8').splitlines():
    if not entry.strip(): continue
    expected,rel=entry.split('  ',1)
    path=ROOT/rel
    if not path.is_file():errors.append('MISSING '+rel);continue
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=expected:errors.append('SHA256 '+rel)
for n,needle in [('benchmark/weights/stage_a_best_model.pt','df3401df311973312e4b07ab6ae559167c668f6e3a57393289804025fabada74'),('benchmark/weights/AttentionNet_published.pt','dd9821f630613432704da6ced89f3fdeae6c91ba689016cb07e7ea3c36046bab')]:
    p=ROOT/n
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=needle:errors.append('CHECKPOINT '+n)
for file in ROOT.rglob('*.py'):
    if any(x in file.parts for x in ('__pycache__','outputs')):continue
    try:py_compile.compile(str(file),cfile=str(Path('/tmp')/('paper2_validate_'+hashlib.sha1(str(file).encode()).hexdigest()+'.pyc')),doraise=True)
    except Exception as exc:errors.append('SYNTAX '+str(file.relative_to(ROOT))+': '+str(exc))
with (ROOT/'results/results37/per_case_direct.csv').open(newline='',encoding='utf8') as f:
    rows=list(csv.DictReader(f))
if len(rows)!=4806:errors.append('Results37 per-case row count is not 4806')
print(f'Checked {len(manifest.read_text().splitlines())} files, 2 checkpoint SHA256 values, {len(rows)} Results37 rows.')
if errors:
    print('FAILED:');print('\n'.join(errors));sys.exit(1)
print('Release OK (integrity + Python syntax). Scientific rerun requires the Python dependencies.')
