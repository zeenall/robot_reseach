#!/usr/bin/env python3
# coding: utf-8
"""Paper 2 — locked 300-case cross-DATASET benchmark.

SADCHER ORIGINAL zero-precedence generator -> unchanged ORIGINAL HeteroMRTA
TaskEnv -> frozen Stage A and original authors' AttentionNet weights.

It is not a native SADCHER simulator experiment. SADCHER and HeteroMRTA
have different coalition start semantics (verified in prior PROBE), so
all scores and replay checks are made in identical HeteroMRTA simulator.
No training, no test-driven model selection, no fifth rollout.

Published generator: https://github.com/jakbichler/Sadcher
Original HeteroMRTA: https://github.com/marmotlab/HeteroMRTA
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import paper2_sadcher_crossdataset as bridge
import train_selective_distillation as sd

VERSION='paper2_sadcher_crossdataset_frozen300_v1'
FULL_SEED_BASE=2026121200  # disjoint from 10-case PROBE (2026111200 + i*1009)
SMOKE_SEED_BASE=2026090900
EXPECTED_PROBE_GENERATOR_SHA='4a885558d682e3295ebac0cfce6af5454b87dd3df45148f7a80efa16ae87ed59'
EXPECTED_PROBE_CASE0_SHA='e3f7b59c0d583f1e7056d3e6cdb80e9845b4aeb9618f0e28cad837bc09b1a018'
EXPECTED_WEIGHTS={
 'stage_a':'df3401df311973312e4b07ab6ae559167c668f6e3a57393289804025fabada74',
 'attention_original':'dd9821f630613432704da6ced89f3fdeae6c91ba689016cb07e7ea3c36046bab',
}
MODELS=('stage_a','attention_original')
METHODS=(('guided_K1',1),('guided_scarcity_K4',4))
K4_CANDIDATES=('guided_greedy','guided_sample','scarcity_greedy','scarcity_sample')
COLUMNS=(
 'case_index','case_seed','case_sha256','source','model','method','K',
 'complete','makespan','penalized_makespan','makespan_source_units','waiting_time',
 'search_elapsed_s','replay_elapsed_s','replay_verified','search_seed',
 'n_candidate_rollouts','completed_candidates','winner_candidate',
 'n_decisions','trace_sha256','n_multiskill_tasks','n_multiskill_robots'
)

def atomic_csv(p,rows,names):
    bridge.csv_atomic(p,rows,names)

def atomic_json(p,obj):
    bridge.atomic_json(p,obj)

def load_case_generator(source,allow_mirror):
    if source:
        src=Path(source).resolve()
        f=src/'data_generation'/'problem_generator.py'
        h=bridge.sha256(f)
        if h!=EXPECTED_PROBE_GENERATOR_SHA:
            raise RuntimeError('SADCHER GENERATOR HASH DOES NOT MATCH PROBE. Expected '
                               +EXPECTED_PROBE_GENERATOR_SHA+' but found '+h)
        return bridge.load_sadcher_generator(src),'sadcher_official_at_probe_commit',h
    if not allow_mirror:
        raise RuntimeError('FULL REQUIRES --sadcher-root original repository. '
                           'Mirrored generator is allowed only for SMOKE.')
    return None,'mirrored_generator_smoke_NOT_PUBLISHABLE',None

def check_models():
    st=ROOT/'reference'/'stage_a_best_model.pt'
    at=ROOT/'weights'/'AttentionNet_published.pt'
    for name,p in [('stage_a',st),('attention_original',at)]:
        if not p.exists(): raise FileNotFoundError(p)
        actual=bridge.sha256(p)
        if actual!=EXPECTED_WEIGHTS[name]:
            raise RuntimeError(f'{name} weight hash mismatch: {actual}')
    stage=sd.load_stage_a(st).eval().cpu()
    attention=sd.load_attention().eval().cpu()
    return {'stage_a':stage,'attention_original':attention}

def make_protocol(a,source_kind,source_sha):
    return {
      'protocol_version':VERSION,
      'assessment':'cross-DATASET ONLY; original HeteroMRTA shared simulator',
      'native_SADCHER_simulator_used':False,
      'generator':source_kind,
      'sadcher_generator_sha256':source_sha,
      'case_count':a.cases,
      'seed_base':a.seed_base,
      'seed_rule':'seed_base + index * 1009',
      'case_shape':{'tasks':20,'robots':15,'skills':5,'precedence':0},
      'depot':'same start/end location, matching SADCHER published option',
      'input_transformation':{'location_divide_by':100,'duration_divide_by':20},
      'HeteroMRTA_simulator':'original env/task_env.py unchanged',
      'HeteroMRTA_simulator_sha256':bridge.sha256(ROOT/'env'/'task_env.py'),
      'bridge_sha256':bridge.sha256(ROOT/'paper2_sadcher_crossdataset.py'),
      'evaluator_sha256':bridge.sha256(Path(__file__)),
      'model_weights_sha256':EXPECTED_WEIGHTS,
      'models':list(MODELS),
      'search_protocol':'K1 is first of same four rollouts used by K4',
      'candidate_types':list(K4_CANDIDATES),
      'search_seed_offset':bridge.SEARCH_SEED_OFFSET,
      'guidance':{name:{'alpha':bridge.ALPHA[name], 'beta':bridge.BETA[name]} for name in MODELS},
      'failure_penalty':200.0,
      'primary_metric':'mean penalized makespan at guided_scarcity_K4',
      'primary_comparison':'stage_a minus attention_original (negative favors Stage A)',
      'bootstrap':'paired case bootstrap 20000 draws, 95% percentile CI',
      'checkpoint_selection':'none; weights locked before test',
      'cross_simulator_equivalence':'NOT established in prior PROBE',
      'python':platform.python_version(),
      'torch':torch.__version__,
      'numpy':np.__version__,
      'cpu_threads':a.cpu_threads,
    }

def verify_resume(rows,a,source_kind):
    by_case=defaultdict(list)
    for r in rows:
        try: ix=int(r['case_index'])
        except (ValueError,KeyError) as exc: raise RuntimeError('Broken resume CSV') from exc
        by_case[ix].append(r)
    expected={(m,k) for m in MODELS for k,_ in METHODS}
    for i, rr in by_case.items():
        if i<0 or i>=a.cases or len(rr)!=4: raise RuntimeError(f'Invalid resume case {i}')
        if {(r['model'],r['method']) for r in rr}!=expected:
            raise RuntimeError(f'Missing/duplicate method in resume case {i}')
        if any(int(r['case_seed'])!=a.seed_base+i*1009 for r in rr):
            raise RuntimeError(f'Seed conflict in resume case {i}')
        if any(r['source']!=source_kind or int(r['replay_verified'])!=1 for r in rr):
            raise RuntimeError(f'Source or replay verification mismatch in case {i}')
        if len(set(r['case_sha256'] for r in rr))!=1:
            raise RuntimeError(f'Case input hash mismatch in case {i}')
    return set(by_case)

def bootstrap_and_summarize(rows,out,draws):
    # Paired resampling is by case; no false independence from K1/K4 or models.
    groups=defaultdict(list)
    for r in rows: groups[r['model'],r['method']].append(r)
    summary=[]
    for model in MODELS:
        for method,k in METHODS:
            rs=groups[model,method]
            if not rs:continue
            vals=np.array([float(r['penalized_makespan']) for r in rs])
            summary.append({
               'model':model,'method':method,'K':k,'n':len(rs),
               'success_count':sum(int(r['complete']) for r in rs),
               'success_rate':float(np.mean([int(r['complete']) for r in rs])),
               'mean_penalized_makespan':float(vals.mean()),
               'std_penalized_makespan':float(vals.std(ddof=1)) if len(vals)>1 else 0.,
               'median_penalized_makespan':float(np.median(vals)),
               'mean_search_elapsed_s':float(np.mean([float(r['search_elapsed_s']) for r in rs])),
               'mean_waiting_time':float(np.mean([float(r['waiting_time']) for r in rs])),
            })
    if summary:atomic_csv(out/'evaluation_summary.csv',summary,list(summary[0]))
    pairs=[]
    for method,k in METHODS:
        ours={int(r['case_index']):r for r in groups['stage_a',method]}
        theirs={int(r['case_index']):r for r in groups['attention_original',method]}
        if set(ours)!=set(theirs):raise RuntimeError('Paired case IDs do not match')
        if not ours:continue
        ix=sorted(ours)
        if any(ours[j]['case_sha256']!=theirs[j]['case_sha256'] for j in ix):
            raise RuntimeError('Input mismatch across models')
        d=np.array([float(ours[j]['penalized_makespan'])-
                    float(theirs[j]['penalized_makespan']) for j in ix])
        if len(d)>1:
            rng=np.random.default_rng(20261230+int(k))
            boot=np.empty(draws,dtype=float)
            for offset in range(0,draws,500):
                stop=min(draws,offset+500)
                sel=rng.integers(0,len(d),size=(stop-offset,len(d)))
                boot[offset:stop]=d[sel].mean(axis=1)
            lo,hi=map(float,np.quantile(boot,[.025,.975]))
        else:lo=hi=float('nan')
        pairs.append({
          'comparison':'stage_a minus attention_original','method':method,'K':k,
          'n_paired_cases':len(d),'mean_delta':float(d.mean()),
          'CI95_lower':lo,'CI95_upper':hi,
          'stage_a_wins':int((d<-1e-9).sum()),
          'attention_wins':int((d>1e-9).sum()),
          'ties':int((abs(d)<=1e-9).sum()),
          'conclusion':'CI excludes zero' if (hi<0 or lo>0) else 'CI includes zero',
          'scope':'Cross-dataset in unchanged HeteroMRTA; NOT cross-simulator'
        })
    if pairs:atomic_csv(out/'paired_statistics.csv',pairs,list(pairs[0]))
    return summary,pairs

def evaluate(a):
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.cpu_threads)
    gen,kind,gen_sha=load_case_generator(a.sadcher_root,a.mirror_smoke)
    models=check_models()
    if gen is not None:
        # Guard against differences in RNG/bridge behavior: reproduce one PROBE input.
        probe0=bridge.generate_case(gen,2026111200,20,15,5)
        actual=bridge.case_hash(probe0)
        if actual!=EXPECTED_PROBE_CASE0_SHA:
            raise RuntimeError('PROBE case 0 is not reproducible with this generator+bridge: '+actual)
    proto=make_protocol(a,kind,gen_sha)
    protocol_path=out/'protocol_locked.json'
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text(encoding='utf8'))
        if old!=proto:raise RuntimeError('RESUME BLOCKED: locked protocol/hash changed. Use new output folder.')
    else:atomic_json(protocol_path,proto)
    data_path=out/'evaluation_per_case.csv'
    rows=[]
    if data_path.exists():
        with data_path.open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    done=verify_resume(rows,a,kind)
    cases_dir=out/'cases';plans_dir=out/'plans'
    cases_dir.mkdir(exist_ok=True);plans_dir.mkdir(exist_ok=True)
    print(f'[START] {a.cases} cases; resumed={len(done)}; source={kind}; native SADCHER simulator=NO',flush=True)
    seen_case_hashes=set()
    for i in range(a.cases):
        seed=int(a.seed_base+i*1009)
        case=bridge.generate_case(gen,seed,20,15,5)
        digest=bridge.case_hash(case)
        if digest in seen_case_hashes:raise RuntimeError(f'Duplicate case content at index {i}')
        seen_case_hashes.add(digest)
        case_path=cases_dir/f'case_{i:04d}.json'
        if case_path.exists():
            old=json.loads(case_path.read_text(encoding='utf8'))
            if old.get('sha256')!=digest or old.get('seed')!=seed:
                raise RuntimeError(f'Resumed case {i} has changed')
        else:atomic_json(case_path,{'sha256':digest,'seed':seed,'data':case})
        if i in done:
            oldrows=[r for r in rows if int(r['case_index'])==i]
            if any(r['case_sha256']!=digest for r in oldrows):
                raise RuntimeError(f'Resumed result/data mismatch at case {i}')
            for m in MODELS:
                for method,_ in METHODS:
                    if not (plans_dir/f'case_{i:04d}_{m}_{method}.json').is_file():
                        raise RuntimeError(f'Missing archived winner trace case={i} model={m} method={method}')
            continue
        env,_mapping=bridge.build_hmrta_env(case)
        new=[]
        for model_name in MODELS:
            model=models[model_name]
            search_seed=seed+bridge.SEARCH_SEED_OFFSET
            t0=time.perf_counter()
            winner,first,candidates=sd.choose_candidates(
                env,model,search_seed,bridge.ALPHA[model_name],bridge.BETA[model_name])
            full_elapsed=time.perf_counter()-t0
            if len(candidates)!=4:raise RuntimeError('Expected exactly four rollouts')
            for method,k in METHODS:
                picked=first if k==1 else winner
                candidate=picked[1];res=candidate['result']
                replay_elapsed=float(sd.g.replay_and_verify(env,candidate))
                if not np.isfinite(float(res['makespan'])):raise RuntimeError('Nonfinite makespan')
                complete=int(bool(res['done']))
                idx=next((j for j,c in enumerate(candidates) if c[1] is candidate),None)
                if idx is None:raise RuntimeError('Winner outside common search budget')
                if k==1 and idx!=0:raise RuntimeError('K1 must be first of K4 rollouts')
                if k==4 and not 0<=idx<4:raise RuntimeError('Illegal candidate index')
                trace_hash=bridge.trace_hash(candidate['trace'])
                atomic_json(plans_dir/f'case_{i:04d}_{model_name}_{method}.json',{
                   'case_index':i,'case_sha256':digest,'model':model_name,'method':method,'K':k,
                   'winner_candidate':candidates[idx][0],
                   'result':res,'trace':candidate['trace'],'routes':candidate['routes'],
                   'trace_sha256':trace_hash,'replay_verified':True,
                   'evaluation_simulator':'HeteroMRTA only',
                })
                new.append({
                    'case_index':i,'case_seed':seed,'case_sha256':digest,
                    'source':kind,'model':model_name,'method':method,'K':k,
                    'complete':complete,'makespan':float(res['makespan']),
                    'penalized_makespan':float(res['makespan']) if complete else 200.0,
                    'makespan_source_units':float(res['makespan'])*20,
                    'waiting_time':float(res['waiting_time']),
                    # K1 uses just first rollout timing; K4 measures all 4 rollouts + selection.
                    'search_elapsed_s':float(res['runtime_s']) if k==1 else full_elapsed,
                    'replay_elapsed_s':replay_elapsed,'replay_verified':1,
                    'search_seed':search_seed,'n_candidate_rollouts':k,
                    'completed_candidates':sum(bool(c[1]['result']['done']) for c in candidates[:k]),
                    'winner_candidate':candidates[idx][0],
                    'n_decisions':len(candidate['trace']),'trace_sha256':trace_hash,
                    'n_multiskill_tasks':int((np.asarray(case['R'])[1:-1].sum(axis=1)>1).sum()),
                    'n_multiskill_robots':int((np.asarray(case['Q']).sum(axis=1)>1).sum()),
                })
        if len(new)!=4:raise RuntimeError('Expected 4 rows per case')
        rows.extend(new)
        atomic_csv(data_path,rows,COLUMNS)
        if a.log_every and ((i+1)%a.log_every==0 or i==a.cases-1):
            print(f'[CASE] {i+1}/{a.cases} StageA_K4={new[1]["penalized_makespan"]:.4f} '
                  f'Attention_K4={new[3]["penalized_makespan"]:.4f}; replay=4/4',flush=True)
    if len(rows)!=a.cases*4:raise RuntimeError(f'Missing rows: got {len(rows)}, expected {a.cases*4}')
    summary,pairs=bootstrap_and_summarize(rows,out,a.bootstrap_draws)
    atomic_json(out/'scientific_scope.json',{
       'dataset':'SADCHER original zero-precedence generator, fixed 20 tasks / 15 robots / 5 skills',
       'evaluation_engine':'original HeteroMRTA TaskEnv (both models)',
       'cross_simulator_validation':'NOT ESTABLISHED; do NOT claim SADCHER simulator results',
       'training_or_adaptation':'NONE',
       'pre_registered_primary_comparison':'stage_a minus attention_original at K4',
       'all_replay_verified':bool(all(int(r['replay_verified'])==1 for r in rows)),
       'complete_case_count':a.cases,
       'total_evaluation_rows':len(rows),
       'published_attention_reference':'original AttentionNet weights with OUR shared K4 search, not original authors full Flashforward method',
       'bootstrap_draws':a.bootstrap_draws,
       'K4_pair':next((p for p in pairs if p['K']==4),None),
    })
    print('[SUMMARY]',summary,flush=True)
    print('[PAIRED]',pairs,flush=True)
    print('[DONE]',str(out),flush=True)

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='/kaggle/working/PAPER2_SADCHER_CROSSDATASET_300')
    p.add_argument('--sadcher-root',default='')
    p.add_argument('--mirror-smoke',action='store_true',help='NO SCIENTIFIC CLAIM (synthetic source mimic)')
    p.add_argument('--cases',type=int,default=300)
    p.add_argument('--seed-base',type=int,default=FULL_SEED_BASE)
    p.add_argument('--cpu-threads',type=int,default=1)
    p.add_argument('--bootstrap-draws',type=int,default=20000)
    p.add_argument('--log-every',type=int,default=10)
    args=p.parse_args()
    if args.cases<1 or args.cpu_threads<1 or args.bootstrap_draws<1000:
        raise ValueError('Invalid cases, cpu-threads or bootstrap draws')
    if not (0<=args.seed_base<=2**32-1) or args.seed_base+(args.cases-1)*1009>=2**32:
        raise ValueError('SADCHER seeds overflow uint32')
    if args.mirror_smoke and args.cases>3:
        raise ValueError('SMOKE mirror capped at 3; official generator required for scientific run')
    if not args.mirror_smoke and (args.cases!=300 or args.seed_base!=FULL_SEED_BASE):
        raise RuntimeError('Published FULL protocol is LOCKED at 300 cases and seed base 2026121200')
    return args

if __name__=='__main__':evaluate(parse_args())
