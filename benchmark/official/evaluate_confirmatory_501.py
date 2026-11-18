#!/usr/bin/env python3
"""Locked, NO-TRAINING confirmatory benchmark for Paper 2.

Pre-specified: three fixed models, 167 new cases in each of 3 modes,
K=4 shared guided/scarcity candidate protocol with K=1 included as candidate 0.
Every winning policy trace is replay-verified in an independent TaskEnv copy.
Existing author/Stage A/Uniform weights are never modified or selected on test.
The published AttentionNet weights are evaluated under OUR shared evaluator;
this does not claim an exact reproduction of the authors' native paper protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import train_selective_distillation as sd

MODES = ('onehot', 'multihot', 'hard')
MODEL_NAMES = ('stage_a', 'uniform_kl_20261112', 'attention_original')
# Pre-specified before looking at new results. Disjoint from prior test seeds.
TEST_SEED_BASE = 3406110000
CANDIDATE_OFFSET = 99331
BOOTSTRAP_SEED = 100117
CONFIG_VERSION = 'paper2_fixed_confirmatory_501_v1'


def say(*xs):
    print(*xs, flush=True)


def shasum(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for blob in iter(lambda: file.read(1 << 20), b''):
            h.update(blob)
    return h.hexdigest()


def atomic_json(data, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(tmp, path)


def atomic_csv(rows, path, fieldnames):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


ROW_COLUMNS = (
    'mode', 'case_index', 'case_seed', 'model', 'method', 'K', 'search_seed',
    'complete', 'makespan', 'penalized_makespan', 'waiting_time',
    'native_reward', 'search_elapsed_s', 'replay_elapsed_s',
    'replay_verified', 'winner_candidate', 'winner_trace_sha256',
    'n_decisions', 'n_candidate_rollouts', 'n_completed_candidates'
)


def load_models(stage_path, uniform_path, attention_path):
    for p in (stage_path, uniform_path, attention_path):
        if not p.is_file():
            raise FileNotFoundError(f'Missing frozen weight file: {p}')
    stage = sd.load_stage_a(stage_path).eval().cpu()

    uck = torch.load(uniform_path, map_location='cpu', weights_only=True)
    required = ('weights', 'arm', 'run_seed', 'selection_method', 'initial_stage_a_sha')
    if any(k not in uck for k in required):
        raise RuntimeError('Uniform checkpoint metadata missing, wrong file?')
    if (uck['arm'] != 'teacher_uniform_kl' or int(uck['run_seed']) != 20261112
            or uck['selection_method'] != 'K4'):
        raise RuntimeError('Uniform checkpoint is not locked best_K4, seed 20261112')
    if str(uck['initial_stage_a_sha']) != shasum(stage_path):
        raise RuntimeError('Uniform checkpoint was trained from a DIFFERENT Stage A!')
    uniform = sd.load_stage_a(stage_path).eval().cpu()
    uniform.load_state_dict(uck['weights'], strict=True)
    uniform.eval()

    # sd.load_attention requires the weight at ROOT/weights/AttentionNet_published.pt.
    expected = ROOT / 'weights' / 'AttentionNet_published.pt'
    if attention_path.resolve() != expected.resolve():
        raise RuntimeError('Original attention weight path is not shared-project reference path')
    attention = sd.load_attention().eval().cpu()
    return {'stage_a': stage,
            'uniform_kl_20261112': uniform,
            'attention_original': attention}


def spec(args, model_hashes):
    return {
        'schema': CONFIG_VERSION,
        'n_per_mode': args.per_mode,
        'modes': list(MODES),
        'total_cases': len(MODES) * args.per_mode,
        'seed_base': TEST_SEED_BASE,
        'candidate_seed_offset': CANDIDATE_OFFSET,
        'candidate_seeds_delta': [0, 7919, 2*7919, 3*7919],
        'K': 4,
        'candidate_schedule': [
            'guided_greedy', 'guided_sample', 'scarcity_greedy', 'scarcity_sample'
        ],
        'models': list(MODEL_NAMES),
        'model_shas': model_hashes,
        'hyperparameters': {
            'stage_a': {'alpha': sd.DEFAULT_ALPHA['stage_a'], 'beta': sd.DEFAULT_BETA['stage_a']},
            'uniform_kl_20261112': {'alpha': sd.DEFAULT_ALPHA['stage_a'], 'beta': sd.DEFAULT_BETA['stage_a']},
            'attention_original': {'alpha': sd.DEFAULT_ALPHA['attention'], 'beta': sd.DEFAULT_BETA['attention']},
        },
        'selection': 'FIXED, no test-based weight/parameter selection',
        'incomplete_penalty': 200.,
        'confidence_interval': 'paired stratified bootstrap by mode (percentile 95%, 20000 draws)',
        'primary_endpoint': 'penalized_makespan using guided+scarcity K=4',
        'primary_pair': 'uniform_kl_20261112 minus attention_original (negative is better)',
        'cpu_threads': args.cpu_threads,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'python_version': platform.python_version(),
    }


def fingerprint_trace(winner):
    payload = json.dumps(winner['trace'], ensure_ascii=False, sort_keys=True,
                         separators=(',', ':')).encode('utf8')
    return hashlib.sha256(payload).hexdigest()


def validate_partial(rows, args):
    # Only complete cases with all 3 models x 2 budgets can be resumed.
    groups = defaultdict(list)
    for r in rows:
        groups[(r['mode'], int(r['case_index']))].append(r)
    accepted = []
    for (mode, case_idx), subset in groups.items():
        expected = {(name, meth) for name in MODEL_NAMES
                    for meth in ('guided_K1', 'guided_scarcity_K4')}
        observed = {(r['model'], r['method']) for r in subset}
        expected_seed = sd.seed_for(mode, case_idx, TEST_SEED_BASE)
        if (mode not in MODES or not 0 <= case_idx < args.per_mode
                or len(subset) != 6 or observed != expected
                or any(int(r['case_seed']) != expected_seed for r in subset)
                or any(int(r['replay_verified']) != 1 for r in subset)):
            raise RuntimeError(f'Corrupt/incomplete CSV case for {mode}/{case_idx}: manual inspection required')
        accepted += subset
    return accepted


def evaluate(args, out, models):
    path = out / 'evaluation_per_case.csv'
    previous = []
    if path.exists():
        with path.open('r', encoding='utf8', newline='') as f:
            previous = validate_partial(list(csv.DictReader(f)), args)
        say(f'[RESUME] {len(previous)//6} complete cases; model hashes/config locked')
    done = {(r['mode'], int(r['case_index'])) for r in previous}
    rows = list(previous)
    for mode in MODES:
        for case_idx in range(args.per_mode):
            if (mode, case_idx) in done:
                continue
            case_seed = sd.seed_for(mode, case_idx, TEST_SEED_BASE)
            env = sd.mixed_env(case_seed, mode)
            env.init_state()
            temporary = []
            for model_name in MODEL_NAMES:
                model = models[model_name]
                label = 'attention' if model_name == 'attention_original' else 'stage_a'
                alpha = sd.DEFAULT_ALPHA[label]
                beta = sd.DEFAULT_BETA[label]
                search_seed = case_seed + CANDIDATE_OFFSET
                t0 = time.perf_counter()
                winner, baseline, candidates = sd.choose_candidates(
                    env, model, search_seed, alpha, beta
                )
                elapsed = time.perf_counter()-t0
                # EXACTLY K=4, K1 baseline is already included in those four.
                if len(candidates) != 4:
                    raise AssertionError('Unexpected additional/hidden candidate rollouts')
                for method, candidate, cost_s, k in (
                    ('guided_K1', baseline[1], float(baseline[1]['result']['runtime_s']), 1),
                    ('guided_scarcity_K4', winner[1], elapsed, 4),
                ):
                    replay_s = sd.g.replay_and_verify(env, candidate)
                    r = candidate['result']
                    # Non-complete schedules remain in analysis, penalized uniformly.
                    complete = int(bool(r['done']))
                    idx = next(i for i, item in enumerate(candidates) if item[1] is candidate)
                    temporary.append({
                        'mode': mode, 'case_index': case_idx, 'case_seed': case_seed,
                        'model': model_name, 'method': method, 'K': k,
                        'search_seed': search_seed, 'complete': complete,
                        'makespan': float(r['makespan']),
                        'penalized_makespan': float(r['makespan']) if complete else 200.,
                        'waiting_time': float(r['waiting_time']),
                        'native_reward': float(r['native_reward']),
                        'search_elapsed_s': cost_s,
                        'replay_elapsed_s': replay_s,
                        'replay_verified': 1,
                        'winner_candidate': candidates[idx][0],
                        'winner_trace_sha256': fingerprint_trace(candidate),
                        'n_decisions': len(candidate['trace']),
                        'n_candidate_rollouts': k,
                        'n_completed_candidates': sum(bool(c[1]['result']['done'])
                                                      for c in candidates[:k]),
                    })
            if len(temporary) != 6:
                raise AssertionError('Case must have 6 rows')
            rows.extend(temporary)
            # Atomic append-by-rewrite protects from notebook interruptions.
            atomic_csv(rows, path, ROW_COLUMNS)
            if (case_idx + 1) % args.log_every == 0 or case_idx + 1 == args.per_mode:
                say(f'[TEST] {mode} {case_idx + 1}/{args.per_mode}; '
                    '3 frozen models; K1 and K4; all plans replay-verified')
    if len(rows) != args.per_mode * len(MODES) * 6:
        raise RuntimeError(f'Expected {args.per_mode * len(MODES) * 6} rows, found {len(rows)}')
    return rows


def metrics_and_pairs(rows, out, bootstrap):
    summary=[]
    for model in MODEL_NAMES:
        for method in ('guided_K1','guided_scarcity_K4'):
            for mode in ('ALL',)+MODES:
                rs = [r for r in rows if r['model'] == model and r['method'] == method
                      and (mode == 'ALL' or r['mode'] == mode)]
                if not rs:
                    raise RuntimeError(f'No observations {model}/{method}/{mode}')
                vals = np.asarray([float(r['penalized_makespan']) for r in rs])
                summary.append({
                    'model': model, 'method': method, 'mode': mode,
                    'n': len(rs), 'success': sum(int(r['complete']) for r in rs),
                    'mean_penalized_makespan': float(vals.mean()),
                    'mean_waiting': float(np.mean([float(r['waiting_time']) for r in rs])),
                    'mean_search_elapsed_s': float(np.mean([float(r['search_elapsed_s']) for r in rs]))
                })
    atomic_csv(summary, out/'evaluation_summary.csv',
               ['model','method','mode','n','success','mean_penalized_makespan','mean_waiting','mean_search_elapsed_s'])

    comparisons = [('uniform_kl_20261112','attention_original'),
                   ('stage_a','attention_original'),
                   ('uniform_kl_20261112','stage_a')]
    output = []
    for method in ('guided_scarcity_K4','guided_K1'):
        for a,b in comparisons:
            a_map={(r['mode'],int(r['case_index'])):float(r['penalized_makespan'])
                   for r in rows if r['model']==a and r['method']==method}
            b_map={(r['mode'],int(r['case_index'])):float(r['penalized_makespan'])
                   for r in rows if r['model']==b and r['method']==method}
            if a_map.keys()!=b_map.keys():raise RuntimeError('Paired case-key mismatch')
            for mode in ('ALL',)+MODES:
                keys=sorted(k for k in a_map if mode=='ALL' or k[0]==mode)
                diff=np.array([a_map[k]-b_map[k] for k in keys],dtype=np.float64)
                mean=float(diff.mean())
                rng=np.random.default_rng(BOOTSTRAP_SEED)
                if mode=='ALL':
                    strats=[np.array([i for i,key in enumerate(keys) if key[0]==m]) for m in MODES]
                else:
                    strats=[np.arange(len(keys))]
                boots=np.empty(bootstrap,dtype=np.float64)
                # Vectorized batches avoid large allocations on Kaggle.
                for begin in range(0,bootstrap,1000):
                    count=min(1000,bootstrap-begin)
                    sampled = np.concatenate([rng.choice(ix, size=(count,len(ix)), replace=True)
                                              for ix in strats],axis=1)
                    boots[begin:begin+count]=diff[sampled].mean(axis=1)
                ci_low,ci_high=np.quantile(boots,[.025,.975])
                output.append({
                    'method':method,'mode':mode,'model_a':a,'model_b':b,
                    'n_paired':len(diff),'mean_difference_a_minus_b':mean,
                    'ci95_lower':float(ci_low),'ci95_upper':float(ci_high),
                    'a_better_cases':int((diff < -1e-8).sum()),
                    'tie_cases':int((np.abs(diff) <= 1e-8).sum()),
                    'b_better_cases':int((diff > 1e-8).sum()),
                    'bootstrap_draws':bootstrap
                })
    atomic_csv(output,out/'paired_statistics.csv',[
        'method','mode','model_a','model_b','n_paired','mean_difference_a_minus_b',
        'ci95_lower','ci95_upper','a_better_cases','tie_cases','b_better_cases','bootstrap_draws'
    ])
    return summary,output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=str, default='/kaggle/working/PAPER2_CONFIRM_501')
    ap.add_argument('--per-mode',type=int,default=167)
    ap.add_argument('--bootstrap',type=int,default=20000)
    ap.add_argument('--log-every',type=int,default=5)
    ap.add_argument('--cpu-threads',type=int,default=1)
    ap.add_argument('--smoke',action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.per_mode=2
        args.bootstrap=200
        say('[SMOKE] 2 cases per mode; ONLY wiring check, NOT confirmatory evidence')
    elif args.per_mode != 167 or args.bootstrap != 20000:
        raise RuntimeError('Locked final protocol requires exactly --per-mode 167 --bootstrap 20000')

    torch.set_num_threads(args.cpu_threads)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    stage=ROOT/'reference'/'stage_a_best_model.pt'
    uniform=ROOT/'reference'/'uniform_kl_seed20261112_bestK4.pt'
    att=ROOT/'weights'/'AttentionNet_published.pt'
    hashes={ 'stage_a':shasum(stage),
             'uniform_kl_20261112':shasum(uniform),
             'attention_original':shasum(att) }
    locked = spec(args,hashes)
    locked_path = out/'locked_protocol.json'
    if locked_path.exists():
        original=json.loads(locked_path.read_text(encoding='utf8'))
        if original!=locked:
            raise RuntimeError('Protocol or weights changed: use the original dataset and a new --out directory')
    else:
        atomic_json(locked,locked_path)
    say('[LOCK]',json.dumps({k:locked[k] for k in ('total_cases','models','primary_pair','K')},ensure_ascii=False))
    models=load_models(stage,uniform,att)
    say('[MODEL] frozen weights loaded; CPU; no training or tuning')
    rows=evaluate(args,out,models)
    sums,pairs=metrics_and_pairs(rows,out,args.bootstrap)
    atomic_json({'all_test_cases':len(MODES)*args.per_mode,
                 'all_replay_verified':all(int(r['replay_verified'])==1 for r in rows),
                 'evaluated_models':list(MODEL_NAMES),
                 'rows':len(rows),
                 'finished_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},out/'completion.json')
    say('[DONE] confirmation finished')
    for r in sums:
        if r['mode']=='ALL':
            say(f"[RESULT] {r['model']:<24} {r['method']:<22} Cmax={r['mean_penalized_makespan']:.4f} success={r['success']}/{r['n']} elapsed={r['mean_search_elapsed_s']:.3f}s")
    for r in pairs:
        if r['method']=='guided_scarcity_K4' and r['mode']=='ALL':
            say(f"[PAIR] {r['model_a']} - {r['model_b']}: {r['mean_difference_a_minus_b']:.4f}, 95% CI [{r['ci95_lower']:.4f}, {r['ci95_upper']:.4f}]")


if __name__ == '__main__':
    main()
