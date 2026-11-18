#!/usr/bin/env python3
"""Paper 2 -- frozen-weight NATIVE AUTHORS' INFERENCE vs our complete K4 system.

Scientific scope:
    - Authors' native GREEDY inference is original Worker.run_episode(training=False,
      sample=False,max_waiting=False). CFM is a training-time mechanism and is NOT
      a K4 inference routine. Optional --native-policy sampling10 matches the
      published test.py's sampled 10-run option.
    - Our complete Stage A + Guided/Scarcity K4 is re-run on THE SAME inputs
      and same machine, and checked against the 801 previously saved results.
    - No training. No weight updates. Both use the byte-identical authors' TaskEnv.
    - Search budget is inherently DIFFERENT (native greedy 1 rollout vs K4 4);
      this is a full deployed-system comparison, NOT a neural-network ablation.
    - SADCHER cases test cross-DATASET only, not cross-simulator equivalence.
"""
from __future__ import annotations
import argparse
import csv
import copy
import hashlib
import json
import lzma
import math
import os
import platform
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / 'official'
sys.path.insert(0, str(OFFICIAL))
from attention import AttentionNet
from worker import Worker
from env.task_env import TaskEnv
from train_temporal_spatial_experiments import mixed_env, MODES
from train_selective_distillation import load_stage_a, choose_candidates

# Git blob SHAs verified against public marmotlab/HeteroMRTA at preparation time.
OFFICIAL_GIT_SHA = {
    'attention.py': 'bc5213ee2e0de1cb473dc6a745632c83df88bbe6',
    'worker.py': '7958d14470c0fc2e5e651d00d67443a848cd34ac',
    'parameters.py': '7f7a25af02c8561dad8326c76792bdfa87c8645b',
    'env/task_env.py': '154832bf1313d7bb87dceac3271fa38f8902f947',
}
OFFSET_NATIVE_SEED = 680013
OFFSET_K4 = {'MIXED_501':99331, 'SADCHER_300':847237}
FAILURE_COST = 200.0
PROTOCOL_VERSION = 'paper2_official_native_system_level_v1'
CSV_COLUMNS = [
    'cohort','case_id','mode','case_index','case_seed','input_sha256','source_case_sha256',
    'native_policy','native_seed','original_complete','original_success_rate',
    'original_makespan','original_penalized_makespan','original_waiting_time',
    'original_travel_dist','original_elapsed_s','original_rollouts','original_replay_verified',
    'stage_a_complete','stage_a_makespan','stage_a_penalized_makespan',
    'stage_a_waiting_time','stage_a_elapsed_s','stage_a_rollouts',
    'stage_a_reference_makespan','stage_a_reference_verified',
    'stage_a_winner_candidate','paired_delta_stage_minus_authors',
    'notes',
]


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''): h.update(part)
    return h.hexdigest()


def git_blob_sha(path: Path) -> str:
    data=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def verify_official_source():
    for rel, expected in OFFICIAL_GIT_SHA.items():
        p=OFFICIAL/rel
        got=git_blob_sha(p)
        if got!=expected:
            raise RuntimeError(f'OFFICIAL SOURCE ALTERED: {rel} SHA {got} expected {expected}')


def seed_everything(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32-1))
    torch.manual_seed(int(seed))


def normalized_json_bytes(x):
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')


def sadcher_case_sha(x):
    return hashlib.sha256(normalized_json_bytes(x)).hexdigest()


def build_sadcher_env(x):
    """Exact structural bridge of prior paper2_sadcher_crossdataset.build_hmrta_env.

    The original HeteroMRTA task_env.py transition/reward implementation is
    completely untouched. SADCHER's identical abilities are grouped by species.
    """
    raw_q=np.asarray(x['Q'],dtype=int)
    requirements=np.asarray(x['R'][1:-1],dtype=int)
    coords=np.asarray(x['task_locations'],dtype=float)
    durations=np.asarray(x['T_e'][1:-1],dtype=float)
    if x['precedence_constraints']:
        raise ValueError('SADCHER precedence exists. This benchmark requires zero precedence.')
    uniq,inv=np.unique(raw_q,axis=0,return_inverse=True)
    task_dict={};agent_dict={};depot_dict={}
    species={'abilities':uniq,'number':[]}
    for k,(r,t,loc) in enumerate(zip(requirements,durations,coords[1:-1])):
        task_dict[k]={'ID':int(k),'requirements':r.copy(),'members':[],'cost':[],
          'location':loc/100.0,'feasible_assignment':False,'finished':False,
          'time_start':0,'time_finish':0,'status':r.copy(),'time':float(t/20.0),
          'sum_waiting_time':0,'efficiency':0,'abandoned_agent':[],
          'optimized_ability':None,'optimized_species':[]}
    mapping=[];aid=0
    for sid,abil in enumerate(uniq):
        original_indices=np.flatnonzero(inv==sid)
        species['number'].append(len(original_indices));species[sid]=[]
        depot_dict[sid]={'location':coords[0]/100.0,'members':[], 'ID':-sid-1}
        for original_index in original_indices:
            agent_dict[aid]={'ID':aid,'species':sid,'abilities':abil.copy(),
               'location':coords[0]/100.0,'route':[-sid-1],'current_task':-sid-1,
               'contributed':False,'arrival_time':[0.0],'cost':[np.array([0.0])],
               'travel_time':0.0,'velocity':0.2,'next_decision':0.0,
               'depot':coords[0]/100.0,'travel_dist':0.0,'sum_waiting_time':0.0,
               'current_action_index':0,'decision_step':0,'task_waiting_ratio':1.0,
               'trajectory':[],'angle':0.0,'returned':False,'assigned':False,
               'pre_set_route':None,'no_choice':False}
            depot_dict[sid]['members'].append(aid);species[sid].append(aid)
            mapping.append(int(original_index));aid+=1
    species['number']=list(map(int,species['number']))
    env=TaskEnv(traits_dim=5)
    env.reset(test_env=(task_dict,agent_dict,depot_dict,species));env.init_state()
    assert len(env.task_dic)==len(requirements) and len(env.agent_dic)==len(raw_q)
    for idx,raw in enumerate(mapping):
        assert np.array_equal(env.agent_dic[idx]['abilities'],raw_q[raw])
        assert np.allclose(env.agent_dic[idx]['depot'],coords[0]/100.0)
    for tid in task_dict:
        assert np.array_equal(env.task_dic[tid]['requirements'],requirements[tid])
        assert np.allclose(env.task_dic[tid]['location'],coords[tid+1]/100.0)
        assert np.isclose(env.task_dic[tid]['time'],durations[tid]/20.0)
    return env


def input_fingerprint(env):
    """Hash immutable initial task and robot data, independent of dictionary internals."""
    data={
        'tasks':[
            {'id':int(t['ID']),'abilities':np.asarray(t['requirements']).tolist(),
             'location':np.asarray(t['location']).tolist(),'duration':float(t['time'])}
            for t in env.task_dic.values()],
        'agents':[
            {'id':int(a['ID']),'species':int(a['species']),
             'abilities':np.asarray(a['abilities']).tolist(),
             'depot':np.asarray(a['depot']).tolist(), 'speed':float(a['velocity'])}
            for a in env.agent_dic.values()],
    }
    return hashlib.sha256(normalized_json_bytes(data)).hexdigest()


def load_reference():
    file=ROOT/'reference'/'stage_a_K4_801.csv'
    with file.open(encoding='utf8',newline='') as f: rr=list(csv.DictReader(f))
    expected={('MIXED_501',f'{mode}:{i}') for mode in MODES for i in range(167)}
    expected.update(('SADCHER_300',str(i)) for i in range(300))
    by={(r['cohort'],r['case_id']):r for r in rr}
    if len(rr)!=801 or set(by)!=expected:
        raise RuntimeError('Stage A reference lacks unique and complete 801 cases')
    if any(r['model']!='stage_a' or r['method']!='guided_scarcity_K4'
           or float(r['alpha'])!=1.1 or float(r['beta'])!=0.8
           or int(r['replay_verified'])!=1 for r in rr):
        raise RuntimeError('Mismatching Stage A frozen reference/coefficients')
    return by


def load_sadcher_cases():
    p=ROOT/'cases'/'sadcher300_official_original_cases.json.xz'
    cases=json.loads(lzma.decompress(p.read_bytes()).decode('utf8'))
    if len(cases)!=300: raise RuntimeError('SADCHER case count is not 300')
    out={}
    for v in cases:
        digest=sadcher_case_sha(v['data'])
        if digest!=v['sha256']:
            raise RuntimeError(f'SADCHER ORIGINAL CASE SHA MISMATCH index={v["index"]} {digest} != {v["sha256"]}')
        out[int(v['index'])]=v
    if set(out)!=set(range(300)):
        raise RuntimeError('SADCHER cases are incomplete/duplicated')
    return out


def case_env(ref, sadcher_cases):
    if ref['cohort']=='MIXED_501':
        e=mixed_env(int(ref['case_seed']),ref['mode'])
        e.init_state()
        return e
    raw=sadcher_cases[int(ref['case_index'])]
    if int(ref['case_seed'])!=int(raw['seed']) or ref['case_sha256']!=raw['sha256']:
        raise RuntimeError('SADCHER stored case does not match Stage A case')
    return build_sadcher_env(raw['data'])


def run_native_once(env_template, net, seed, sample=False, time_limit=120):
    # Native Worker + native original decision process (source hashes verified).
    seed_everything(seed)
    w=Worker(0,net,net,0,device='cpu',seed=int(seed))
    w.env=copy.deepcopy(env_template)
    w.env.init_state()
    start=time.perf_counter()
    if time_limit and hasattr(signal,'SIGALRM'):
        def _alarm(_signum, _frame): raise TimeoutError('Authors native episode exceeded wall-clock limit')
        previous=signal.signal(signal.SIGALRM,_alarm)
        signal.setitimer(signal.ITIMER_REAL,time_limit)
    try:
        with torch.no_grad(): _,_,m=w.run_episode(False,bool(sample),False)
    finally:
        elapsed=time.perf_counter()-start
        if time_limit and hasattr(signal,'SIGALRM'):
            signal.setitimer(signal.ITIMER_REAL,0)
            signal.signal(signal.SIGALRM,previous)
    success=float(m['success_rate'][0])
    result={
        'complete':int(success>=1.0-1e-10 and bool(w.env.finished)),
        'success_rate':success,'makespan':float(m['makespan'][0]),
        'waiting_time':float(m['waiting_time'][0]),
        'travel_dist':float(m['travel_dist'][0]),
        'elapsed_s':float(elapsed),
        'route_sha':hashlib.sha256(normalized_json_bytes([
              {'route':[int(t) for t in a['route']],
               'arrival_time':[float(t) for t in a['arrival_time']]}
              for a in w.env.agent_dic.values()])).hexdigest(),
    }
    result['penalized_makespan']=result['makespan'] if result['complete'] else FAILURE_COST
    return result


def run_native(env, net, seed, policy, timeout):
    n=1 if policy=='greedy' else 10
    options=[]
    total_time=0.
    for i in range(n):
        # Identical original policy. With sample=True, choose best of ten
        # complete episodes, as in the authors' test.py sampling=True mode.
        result=run_native_once(env,net,seed+i*7919,sample=(policy!='greedy'),time_limit=timeout)
        total_time+=result['elapsed_s']
        options.append(result)
    winner=min(options,key=lambda r:(r['penalized_makespan'],r['makespan']))
    return {**winner,'elapsed_s':total_time,'rollouts':n}


def run_stage_a_k4(env, stage, ref):
    seed=int(ref['case_seed'])+OFFSET_K4[ref['cohort']]
    env=copy.deepcopy(env);env.init_state()
    start=time.perf_counter()
    winner,baseline,candidates=choose_candidates(env,stage,seed,alpha=1.1,beta=0.8)
    elapsed=time.perf_counter()-start
    if len(candidates)!=4: raise RuntimeError('Stage A did not use exactly 4 rollouts')
    name,entry=winner
    m=entry['result']
    complete=int(bool(m['done']))
    return dict(complete=complete,makespan=float(m['makespan']),
         penalized_makespan=float(m['makespan']) if complete else FAILURE_COST,
         waiting_time=float(m['waiting_time']),elapsed_s=float(elapsed),
         rollouts=4, winner_candidate=name)


def atomic_csv(rows,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp')
    with temp.open('w',encoding='utf8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=CSV_COLUMNS,extrasaction='ignore')
        wr.writeheader();wr.writerows(rows)
    os.replace(temp,p)


def atomic_json(obj,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    os.replace(temp,p)


def paired_bootstrap(rows,cohort,draws=10000):
    by=[r for r in rows if r['cohort']==cohort]
    if len(by)<2:return None
    groups=defaultdict(list)
    for r in by:groups[r['mode']].append(float(r['paired_delta_stage_minus_authors']))
    rng=np.random.default_rng(20261114 + (1 if cohort=='MIXED_501' else 2))
    vals=np.empty(draws)
    pools=[np.array(v,dtype=float) for v in groups.values()]
    for offset in range(0,draws,256):
        count=min(256,draws-offset)
        blocks=[]
        for v in pools:
            ix=rng.integers(0,len(v),size=(count,len(v)))
            blocks.append(v[ix])
        vals[offset:offset+count]=np.concatenate(blocks,axis=1).mean(axis=1)
    lo,hi=np.quantile(vals,[.025,.975])
    return float(lo),float(hi)


def summarize(rows,out,draws=10000):
    summary=[]
    for cohort in ('MIXED_501','SADCHER_300'):
        rr=[r for r in rows if r['cohort']==cohort]
        if not rr:continue
        d=np.array([float(r['paired_delta_stage_minus_authors']) for r in rr])
        ci=paired_bootstrap(rr,cohort,draws=draws)
        summary.append({
            'cohort':cohort,'n':len(rr),'native_policy':rr[0]['native_policy'],
            'native_success_rate':np.mean([int(r['original_complete']) for r in rr]),
            'stage_a_success_rate':np.mean([int(r['stage_a_complete']) for r in rr]),
            'native_mean_penalized_makespan':np.mean([float(r['original_penalized_makespan']) for r in rr]),
            'stage_a_mean_penalized_makespan':np.mean([float(r['stage_a_penalized_makespan']) for r in rr]),
            'delta_stage_minus_native':float(d.mean()),
            'paired_CI95_low':ci[0] if ci else '',
            'paired_CI95_high':ci[1] if ci else '',
            'stage_a_wins':int((d<-1e-9).sum()),
            'native_wins':int((d>1e-9).sum()),
            'ties':int((abs(d)<=1e-9).sum()),
            'native_mean_sameday_cpu_time_s':np.mean([float(r['original_elapsed_s']) for r in rr]),
            'stage_a_mean_sameday_cpu_time_s':np.mean([float(r['stage_a_elapsed_s']) for r in rr]),
            'reference_matches':sum(int(r['stage_a_reference_verified']) for r in rr),
            'limitation':('native inference = 1 greedy rollout' if rr[0]['native_policy']=='greedy' else 'native inference = 10 sampled rollouts')+'; ours K4 4 rollouts; training-time CFM not used at test'
        })
    if summary:
        p=out/'summaries.csv'
        with p.open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    return summary


def build_protocol(args):
    return {
       'version':PROTOCOL_VERSION,'date_prepared':'2026-11-14',
       'official_repository':'https://github.com/marmotlab/HeteroMRTA',
       'native_inference':'UNMODIFIED official Worker.run_episode(training=False,sample=False,max_waiting=False)',
       'native_policy':args.native_policy,
       'CFM_role':'training-time; not an extra inference-time search operation',
       'our_system':'frozen Stage A + Guided-Scarcity K4; alpha=1.1,beta=0.8',
       'comparison_type':'whole-system; UNEQUAL candidate budget (1 vs 4 for default greedy)',
       'same_environment':'byte-identical original TaskEnv; same per-case state',
       'SADCHER_bridge':'original case JSON; zero precedence; coordinates/100; task durations/20; skills grouped',
       'failure_penalty':FAILURE_COST,'checkpoint_selection':'previously frozen; no training',
       'test_reuse':'POST-HOC comparison on previously seen test cases',
       'cohort':args.cohort,'mixed_per_mode':args.mixed_per_mode,
       'sadcher_count':args.sadcher_count,'cpu_threads':args.cpu_threads,
       'native_seed_rule':f'case_seed+{OFFSET_NATIVE_SEED}',
       'stage_a_search_seed_offsets':OFFSET_K4,
       'native_time_limit_s':args.timeout,'native_replay_checks_per_cohort':args.replay_checks,
       'official_git_sha':OFFICIAL_GIT_SHA,
       'file_hashes':{
           'published_attention_weights':sha(ROOT/'weights'/'AttentionNet_published.pt'),
           'stage_a_weights':sha(ROOT/'weights'/'stage_a_best_model.pt'),
           'reference_801':sha(ROOT/'reference'/'stage_a_K4_801.csv'),
           'sadcher_300_cases':sha(ROOT/'cases'/'sadcher300_official_original_cases.json.xz'),
       },
       'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
    }


def main():
    parser=argparse.ArgumentParser(description='Frozen Stage A K4 versus original authors native policy (no training).')
    parser.add_argument('--cohort',choices=['both','mixed','sadcher'],default='both')
    parser.add_argument('--mixed-per-mode',type=int,default=167)
    parser.add_argument('--sadcher-count',type=int,default=300)
    parser.add_argument('--native-policy',choices=['greedy','sampling10'],default='greedy')
    parser.add_argument('--cpu-threads',type=int,default=1)
    parser.add_argument('--timeout',type=int,default=120)
    parser.add_argument('--replay-checks',type=int,default=2,
                        help='Number of native deterministic re-runs per cohort (outside timing).')
    parser.add_argument('--out',default=str(ROOT/'outputs'/'full'))
    parser.add_argument('--bootstrap-draws',type=int,default=10000)
    args=parser.parse_args()
    if not (0<=args.mixed_per_mode<=167 and 0<=args.sadcher_count<=300):
        raise ValueError('Cannot request more cases than the locked reference 167x3+300')
    if args.cpu_threads<1:raise ValueError('cpu_threads must be >=1')
    torch.set_num_threads(args.cpu_threads)
    verify_official_source()
    ref=load_reference();sadcher=load_sadcher_cases()
    att_ck=torch.load(ROOT/'weights'/'AttentionNet_published.pt',map_location='cpu',weights_only=True)
    if att_ck.get('model_name')!='AttentionNet_published':raise RuntimeError('Not the published AttentionNet weights')
    attention=AttentionNet(11,15,128).cpu().eval()
    attention.load_state_dict(att_ck['weights'],strict=True)
    stage=load_stage_a(ROOT/'weights'/'stage_a_best_model.pt').cpu().eval()
    params=sum(p.numel() for p in attention.parameters()),sum(p.numel() for p in stage.parameters())
    print('[WEIGHTS] original AttentionNet=%d Stage A=%d' % params,flush=True)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    proto=build_protocol(args);proto_path=out/'protocol_locked.json'
    if proto_path.exists():
        old=json.loads(proto_path.read_text(encoding='utf8'))
        if old!=proto: raise RuntimeError('Resume BLOCKED due to changed protocol, hashes, versions or options. Use different --out.')
    else:atomic_json(proto,proto_path)
    result_file=out/'per_case.csv'
    if result_file.exists():
        with result_file.open(encoding='utf8',newline='') as f:results=list(csv.DictReader(f))
    else:results=[]
    done=set()
    for r in results:
        key=(r['cohort'],r['case_id'])
        if key in done:raise RuntimeError('Duplicate resume row: '+str(key))
        if int(r['stage_a_reference_verified'])!=1:raise RuntimeError('Saved row missing Stage A reference check: '+str(key))
        done.add(key)
    planned=[]
    if args.cohort in ('both','mixed'):
        planned += [ref['MIXED_501',f'{mode}:{i}'] for mode in MODES for i in range(args.mixed_per_mode)]
    if args.cohort in ('both','sadcher'):
        planned += [ref['SADCHER_300',str(i)] for i in range(args.sadcher_count)]
    print(f'[START] planned={len(planned)} completed={len(done)} mode={args.native_policy}; NO TRAINING',flush=True)
    replay_count=Counter()
    for j,ref_row in enumerate(planned):
        key=(ref_row['cohort'],ref_row['case_id'])
        if key in done:continue
        e=case_env(ref_row,sadcher)
        digest=input_fingerprint(e)
        # Confirm new and old code get the exact same immutable inputs.
        e_second=case_env(ref_row,sadcher)
        if input_fingerprint(e_second)!=digest:raise RuntimeError('Inputs changed across independent env constructors')
        authors_seed=int(ref_row['case_seed'])+OFFSET_NATIVE_SEED
        # Alternate evaluation order to reduce warm-cache advantage for one system.
        if (j%2)==0:
            orig=run_native(e,attention,authors_seed,args.native_policy,args.timeout)
            ours=run_stage_a_k4(e_second,stage,ref_row)
        else:
            ours=run_stage_a_k4(e_second,stage,ref_row)
            orig=run_native(e,attention,authors_seed,args.native_policy,args.timeout)
        expected=float(ref_row['makespan'])
        if int(ours['complete'])!=int(ref_row['complete']) or not math.isclose(ours['makespan'],expected,rel_tol=0,abs_tol=1e-6):
            atomic_json({'case':list(key),'expected_reference':dict(ref_row),
                'our_reproduction':ours,'input_sha256':digest},out/'MISMATCH_STOP.json')
            raise RuntimeError(f'Stage A failed exact reference re-run case={key}, expected={expected}, got={ours["makespan"]}; STOPPED')
        verified=''
        if replay_count[ref_row['cohort']]<args.replay_checks:
            repeat=run_native(e,attention,authors_seed,args.native_policy,args.timeout)
            for k in ['complete','makespan','waiting_time','travel_dist','route_sha']:
                x,y=orig[k],repeat[k]
                if isinstance(x,float):identical=math.isclose(x,y,rel_tol=0,abs_tol=1e-9)
                else:identical=x==y
                if not identical:raise RuntimeError(f'Native replay mismatch {key} field {k}: {x} vs {y}')
            replay_count[ref_row['cohort']]+=1
            verified=1
        native_pen=float(orig['penalized_makespan']);stage_pen=float(ours['penalized_makespan'])
        row={
            'cohort':ref_row['cohort'],'case_id':ref_row['case_id'],
            'mode':ref_row['mode'],'case_index':ref_row['case_index'],'case_seed':ref_row['case_seed'],
            'input_sha256':digest,'source_case_sha256':ref_row['case_sha256'],
            'native_policy':args.native_policy,'native_seed':authors_seed,
            'original_complete':orig['complete'],'original_success_rate':orig['success_rate'],
            'original_makespan':orig['makespan'],'original_penalized_makespan':native_pen,
            'original_waiting_time':orig['waiting_time'],'original_travel_dist':orig['travel_dist'],
            'original_elapsed_s':orig['elapsed_s'],'original_rollouts':orig['rollouts'],
            'original_replay_verified':verified,
            'stage_a_complete':ours['complete'],'stage_a_makespan':ours['makespan'],
            'stage_a_penalized_makespan':stage_pen,'stage_a_waiting_time':ours['waiting_time'],
            'stage_a_elapsed_s':ours['elapsed_s'],'stage_a_rollouts':ours['rollouts'],
            'stage_a_reference_makespan':expected,'stage_a_reference_verified':1,
            'stage_a_winner_candidate':ours['winner_candidate'],
            'paired_delta_stage_minus_authors':stage_pen-native_pen,
            'notes':'POST-HOC matched inputs; native greedy 1 vs K4 4 rollouts' if args.native_policy=='greedy' else 'POST-HOC matched inputs; native sample10 vs K4 4 rollouts',
        }
        results.append(row);done.add(key)
        atomic_csv(results,result_file)
        if len(done)%10==0 or len(done)==len(planned):
            say=f'{len(done)}/{len(planned)} {key[0]} {key[1]} | native={orig["makespan"]:.3f} K4={ours["makespan"]:.3f}'
            print('[EVAL] '+say,flush=True)
    summary=summarize(results,out,draws=args.bootstrap_draws)
    atomic_json({'n_completed':len(results),'n_planned':len(planned),'result':'Complete' if len(results)==len(planned) else 'Partial',
       'note':'Post-hoc. Original CFM is training-time, native inference is original Worker. Full systems have unequal candidate budget.',
       'summary':summary},out/'summary.json')
    print('[DONE]',json.dumps(summary,ensure_ascii=False,indent=2,default=float),flush=True)


if __name__=='__main__':main()
