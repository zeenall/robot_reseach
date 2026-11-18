#!/usr/bin/env python3
"""Paper 2 independent SADCHER zero-precedence compatibility / evaluation probe.

NO TRAINING / NO architecture edits. Keeps original HeteroMRTA TaskEnv unchanged.
A) Use SADCHER's original published generator if the repo is supplied (--sadcher-root).
B) Bridge to original HeteroMRTA under documented (100,20) coordinate/time scales.
C) Test fixed Stage A and published AttentionNet policies on identical instances using
   exactly one or four simulator-assisted rollouts per model.
D) Save decisions and plans, exact replay on fresh TaskEnv, and optional SADCHER
   native discrete-time simulator diagnostic. Cross-simulator consistency is NOT
   assumed or reported without verification. See README for validity caveats.

Published generators: https://github.com/jakbichler/Sadcher
HeteroMRTA original: https://github.com/marmotlab/HeteroMRTA
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import guided_coalition_search as g
import train_selective_distillation as sd
import scarcity_bottleneck_search as sc
from env.task_env import TaskEnv

CONFIG_VERSION='sadcher_bridge_gate_v1'
GENERATOR_SEED=2026111200
SEARCH_SEED_OFFSET=847237
# Pre-fixed Stage A / AttentionNet guidance coefficients from previous experiments.
MODELS=('stage_a','attention_original')
ALPHA={'stage_a':1.1,'attention_original':0.7}
BETA={'stage_a':0.8,'attention_original':0.4}
SCALE_POS=100.0
SCALE_TIME=20.0


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def atomic_json(p, data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=clean_json),encoding='utf8');os.replace(temp,p)


def clean_json(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,)):return float(x)
    raise TypeError(type(x).__name__)


def csv_atomic(p, rows, names):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(str(p)+'.tmp')
    with temp.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=names,extrasaction='ignore');w.writeheader();w.writerows(rows)
    os.replace(temp,p)


def load_sadcher_generator(repo):
    if not repo:return None
    repo=Path(repo).resolve()
    f=repo/'data_generation'/'problem_generator.py'
    if not f.is_file():raise FileNotFoundError(f'Original SADCHER generator not found: {f}')
    spec=importlib.util.spec_from_file_location('_sadcher_generator_original',str(f))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if not hasattr(m,'generate_random_data_with_precedence'):
        raise RuntimeError('SADCHER generator missing expected function')
    return m.generate_random_data_with_precedence


def generate_mirror(seed,n_tasks,n_robots,n_skills):
    """Source-matched independently coded fallback for *smoke/debug only*.
    Not treated as a published native-generator run in the scientific report.
    """
    rng=np.random.RandomState(seed)
    q=rng.randint(0,2,(n_robots,n_skills))
    for i in range(n_robots):
        if not q[i].any():q[i,rng.randint(0,n_skills)]=1
    for k in range(n_skills):
        if not q[:,k].any():q[rng.randint(0,n_robots),k]=1
    r=rng.randint(0,2,(n_tasks,n_skills))
    for j in range(n_tasks):
        if not r[j].any():r[j,rng.randint(0,n_skills)]=1
    te=rng.randint(50,100,n_tasks)
    points=rng.randint(0,100,(n_tasks+2,2))
    points[-1]=points[0]
    r=np.vstack((np.zeros((1,n_skills),dtype=int),r,np.zeros((1,n_skills),dtype=int)))
    te=np.hstack(([0],te,[0]))
    tt=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=-1)
    return {'Q':q,'R':r,'T_e':te,'T_t':tt,'task_locations':points,'precedence_constraints':[]}


def generate_case(original_generator,seed,n_tasks=20,n_robots=15,n_skills=5):
    if original_generator is not None:
        state=np.random.get_state()
        try:
            np.random.seed(int(seed));p=original_generator(n_tasks,n_robots,n_skills,0)
        finally:
            np.random.set_state(state)
        x={k:p[k] for k in ('Q','R','T_e','T_t','task_locations','precedence_constraints')}
        x={k:(np.array(v,copy=True) if k!='precedence_constraints' and v is not None else v) for k,v in x.items()}
        if x['precedence_constraints'] is not None and len(x['precedence_constraints'])!=0:
            raise RuntimeError('Original SADCHER generator produced precedence constraints with n_precedence=0')
    else:x=generate_mirror(seed,n_tasks,n_robots,n_skills)
    # Explicitly zero-precedence and common start/end depot, as described in SADCHER CLI.
    x['precedence_constraints']=[]
    x['task_locations'][-1]=x['task_locations'][0].copy()
    x['T_t']=np.linalg.norm(x['task_locations'][:,None]-x['task_locations'][None,:],axis=-1)
    check_case(x,n_tasks,n_robots,n_skills)
    return x


def check_case(x,nt,nr,nsk):
    Q=np.asarray(x['Q']);R=np.asarray(x['R']);T=np.asarray(x['T_e']);L=np.asarray(x['task_locations'])
    if Q.shape!=(nr,nsk) or R.shape!=(nt+2,nsk) or T.shape!=(nt+2,) or L.shape!=(nt+2,2):
        raise AssertionError('SADCHER shape mismatch')
    if not np.array_equal(L[0],L[-1]):raise AssertionError('Different start/end depot')
    if not np.all(np.isin(Q,[0,1])) or not np.all(np.isin(R,[0,1])):raise AssertionError('Non-binary skills')
    if not np.all(Q.sum(1)>0) or not np.all(R[1:-1].sum(1)>0):raise AssertionError('Empty skill vector')
    if not np.all(Q.sum(0)>0):raise AssertionError('Missing team skill')
    if np.any(T[1:-1]<=0):raise AssertionError('Zero task duration')
    if any(np.asarray(x['precedence_constraints']).flatten()):raise AssertionError('Nonzero precedence')


def build_hmrta_env(x):
    """Faithful interface adapter (not a changed transition/reward function).
    SADCHER robot order is mapped explicitly because HMRTA groups by species.
    """
    raw_q=np.asarray(x['Q'],dtype=int)
    requirements=np.asarray(x['R'][1:-1],dtype=int)
    coords=np.asarray(x['task_locations'],dtype=float)
    durations=np.asarray(x['T_e'][1:-1],dtype=float)
    uniq,inv=np.unique(raw_q,axis=0,return_inverse=True)
    task_dict={};agent_dict={};depot_dict={}
    species={'abilities':uniq,'number':[]}
    for k,(r,t,loc) in enumerate(zip(requirements,durations,coords[1:-1])):
        task_dict[k]={'ID':int(k),'requirements':r.copy(),'members':[],'cost':[],
          'location':loc/SCALE_POS,'feasible_assignment':False,'finished':False,
          'time_start':0,'time_finish':0,'status':r.copy(),'time':float(t/SCALE_TIME),
          'sum_waiting_time':0,'efficiency':0,'abandoned_agent':[],
          'optimized_ability':None,'optimized_species':[]}
    mapping=[]
    aid=0
    for sid, abil in enumerate(uniq):
        original_indices=np.flatnonzero(inv==sid)
        species['number'].append(len(original_indices));species[sid]=[]
        depot_dict[sid]={'location':coords[0]/SCALE_POS,'members':[], 'ID':-sid-1}
        for original_index in original_indices:
            agent_dict[aid]={'ID':aid,'species':sid,'abilities':abil.copy(),
               'location':coords[0]/SCALE_POS,'route':[-sid-1],'current_task':-sid-1,
               'contributed':False,'arrival_time':[0.0],'cost':[np.array([0.0])],
               'travel_time':0.0,'velocity':0.2,'next_decision':0.0,
               'depot':coords[0]/SCALE_POS,'travel_dist':0.0,'sum_waiting_time':0.0,
               'current_action_index':0,'decision_step':0,'task_waiting_ratio':1.0,
               'trajectory':[],'angle':0.0,'returned':False,'assigned':False,
               'pre_set_route':None,'no_choice':False}
            depot_dict[sid]['members'].append(aid);species[sid].append(aid)
            mapping.append(int(original_index));aid+=1
    species['number']=list(map(int,species['number']))
    env=TaskEnv(traits_dim=5)
    env.reset(test_env=(task_dict,agent_dict,depot_dict,species));env.init_state()
    # Strict structural audit of the input after bridge (sums are preserved exactly).
    assert len(env.task_dic)==len(requirements) and len(env.agent_dic)==len(raw_q)
    for aid, raw in enumerate(mapping):
        assert np.array_equal(env.agent_dic[aid]['abilities'],raw_q[raw])
        assert np.allclose(env.agent_dic[aid]['depot'],coords[0]/100)
    for tid in task_dict:
        assert np.array_equal(env.task_dic[tid]['requirements'],requirements[tid])
        assert np.allclose(env.task_dic[tid]['location'],coords[tid+1]/100)
        assert np.isclose(env.task_dic[tid]['time'],durations[tid]/20)
    return env,mapping


def case_hash(case):
    d={k:(np.asarray(v).tolist() if k!='precedence_constraints' else []) for k,v in case.items()}
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def trace_hash(trace):
    return hashlib.sha256(json.dumps(trace,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def find_candidate(model, env, seed, label, K):
    """Pre-fixed four-candidate schedule, first candidate is guided greedy K=1.
    Same search family as Paper2 confirmatory experiment; no search over test choices.
    """
    alpha=ALPHA[label];beta=BETA[label]
    base=sd.choose_candidates(env,model,seed,alpha,beta)
    winner,baseline,candidates=base
    if len(candidates)!=4:raise AssertionError('K budget exceeded')
    if K==1:
        chosen=baseline[1];search_method='guided_K1'
        t=chosen['result']['runtime_s'];completed=int(bool(chosen['result']['done']))
    elif K==4:
        chosen=winner[1];search_method='guided_scarcity_K4'
        t=sum(float(c[1]['result']['runtime_s']) for c in candidates)
        completed=sum(bool(c[1]['result']['done']) for c in candidates)
    else:raise ValueError('K must be 1 or 4')
    replay_s=g.replay_and_verify(env,chosen)
    return chosen,search_method,t,float(replay_s),completed,candidates


def verify_native_sadcher(root,case,trace,mapping,continuous_cmax,max_ticks=30000):
    """Diagnostic replay in actual SADCHER discrete timestep Simulator.

    We inject robot assignments at the ceil-scaled event time recorded in TaskEnv.
    This is a transfer test, not an assertion of mathematical equivalence: a
    scheduled decision may happen when an agent is unavailable in SADCHER.
    Discrete tick motion and SADCHER's boolean coalition semantics can also
    change execution order. These cases are flagged, NEVER silently patched.
    """
    if not root:return {'status':'not_run','reason':'native SADCHER repository not supplied'}
    root=Path(root).resolve()
    if not (root/'simulation_environment'/'simulator_2D.py').is_file():
        return {'status':'not_run','reason':'original SADCHER simulator file missing'}
    # Untrusted imported third-party code: user opted into executing this repository.
    sys.path.insert(0,str(root))
    from simulation_environment.simulator_2D import Simulation
    prob={k:copy.deepcopy(v) for k,v in case.items()}
    sim=Simulation(prob,scheduler_name=None,use_idle=False)
    events=[]
    for e in trace:
        aid=int(e['agent_id']);target=int(e['action']);tid=(len(sim.tasks)-1 if target==0 else target)
        events.append((int(math.ceil(float(e['time'])*SCALE_TIME-1e-9)),aid,tid))
    events.sort(key=lambda z:z[0])
    cursor=0
    for tick in range(max_ticks):
        if sim.sim_done:break
        while cursor<len(events) and events[cursor][0] <=sim.timestep:
            planned,aid,tid=events[cursor]; raw=int(mapping[aid]);robot=sim.robots[raw]
            # SAG: we do not silently overwrite still-occupied robots.
            if not robot.available:
                return {'status':'semantic_mismatch','reason':'robot_not_available_at_recorded_decision',
                        'tick':int(sim.timestep),'decision_tick':planned,'robot_id':raw,
                        'unconsumed_actions':len(events)-cursor}
            task=sim.tasks[tid]
            if task.status=='DONE' and tid!=sim.last_task_id:
                return {'status':'semantic_mismatch','reason':'task_already_done_before_assignment',
                        'tick':int(sim.timestep),'robot_id':raw,'task_id':tid}
            robot.current_task=task;robot.available=False;task.assigned=True
            cursor+=1
        sim.step()
    if not sim.sim_done:
        return {'status':'timeout','ticks':max_ticks,'actions_consumed':cursor}
    native_ms=float(sim.makespan)
    all_tasks_done=all(task.status=='DONE' for task in sim.tasks[1:-1])
    robots_returned=bool(sim.all_robots_at_exit_location(threshold=0.01))
    delta=native_ms-float(continuous_cmax)*SCALE_TIME
    # Strict tolerance: 1 tick difference (NOT guaranteed to pass given models).
    aligned=int(cursor==len(events) and all_tasks_done and robots_returned and abs(delta)<=1.0)
    return {'status':'aligned' if aligned else 'semantic_mismatch',
            'native_makespan':native_ms,'scaled_hmrta_makespan':float(continuous_cmax)*SCALE_TIME,
            'difference_ticks':delta,'actions_consumed':cursor,'all_actions_consumed':cursor==len(events),
            'n_events':len(events),'all_tasks_done':all_tasks_done,'robots_returned':robots_returned,
            'equivalence_within_1_tick':bool(aligned)}


FIELDS=('case_index','case_seed','case_sha256','generator','model','method','K','success',
        'makespan_hmrta','makespan_sadcher_units','waiting_hmrta','native_reward',
        'rollouts','completed_candidates','search_elapsed_s','replay_elapsed_s','replay_verified',
        'n_decisions','trace_sha256','native_sadcher_status','native_sadcher_ms',
        'native_difference_ticks','native_reason')


def score(rows,output,bootstrap_draws=5000):
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['method'])].append(r)
    summary=[]
    for (model,method),sub in sorted(groups.items()):
        vals=[float(x['makespan_hmrta']) if int(x['success']) else 200 for x in sub]
        summary.append({'model':model,'method':method,'n':len(sub),
           'success':sum(int(x['success']) for x in sub),'mean_penalized_makespan_hmrta':float(np.mean(vals)),
           'mean_sadcher_units':float(np.mean(vals))*20,
           'mean_search_elapsed_s':float(np.mean([float(x['search_elapsed_s']) for x in sub]))})
    csv_atomic(output/'summary.csv',summary,list(summary[0]) if summary else [])
    pairs=[]
    for method in ('guided_K1','guided_scarcity_K4'):
        own={int(r['case_index']):r for r in rows if r['model']=='stage_a' and r['method']==method}
        ref={int(r['case_index']):r for r in rows if r['model']=='attention_original' and r['method']==method}
        if set(own)!=set(ref):raise RuntimeError('Mismatched cases in comparison')
        vals=np.array([(float(own[j]['makespan_hmrta']) if int(own[j]['success']) else 200)-
          (float(ref[j]['makespan_hmrta']) if int(ref[j]['success']) else 200) for j in sorted(own)])
        if len(vals)>=2:
            rng=np.random.default_rng(202611121)
            means=np.empty(bootstrap_draws)
            for i in range(bootstrap_draws):means[i]=vals[rng.integers(0,len(vals),len(vals))].mean()
            lo,hi=map(float,np.quantile(means,[0.025,0.975]))
        else:lo=hi=float('nan')
        pairs.append({'comparison':'stage_a - attention_original','method':method,
                      'n':len(vals),'mean_delta_hmrta':float(vals.mean()),
                      'CI95_lower':lo,'CI95_upper':hi,
                      'note':'exploratory: native simulator agreement required for cross-simulator claim'})
    csv_atomic(output/'pairs.csv',pairs,list(pairs[0]) if pairs else [])
    return summary,pairs


def run(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    orig=load_sadcher_generator(args.sadcher_root)
    if orig is None and not args.mirror_smoke:
        raise RuntimeError('Original SADCHER repo required for scientific evaluation. '
                           'Set --sadcher-root or use --mirror-smoke for plumbing test only.')
    if args.native and not args.sadcher_root:raise RuntimeError('--native requires --sadcher-root')
    stage_path=ROOT/'reference'/'stage_a_best_model.pt'
    att_path=ROOT/'weights'/'AttentionNet_published.pt'
    models={'stage_a':sd.load_stage_a(stage_path).eval().cpu(),
            'attention_original':sd.load_attention().eval().cpu()}
    model_hashes={'stage_a':sha256(stage_path),'attention_original':sha256(att_path)}
    spec={'version':CONFIG_VERSION,'generator':'sadcher_original' if orig else 'mirrored_smoke_not_published',
          'seed_base':args.seed_base,'n_cases':args.cases,'n_tasks':args.tasks,
          'n_robots':args.robots,'n_skills':args.skills,'precedence':0,
          'same_start_end':True,'scale_position':100,'scale_time':20,
          'native_requested':bool(args.native),'model_hashes':model_hashes,
          'sadcher_generator_sha256':sha256(Path(args.sadcher_root)/'data_generation'/'problem_generator.py') if args.sadcher_root else None,
          'sadcher_simulator_sha256':sha256(Path(args.sadcher_root)/'simulation_environment'/'simulator_2D.py') if args.sadcher_root and (Path(args.sadcher_root)/'simulation_environment'/'simulator_2D.py').exists() else None,
          'methods':['guided_K1','guided_scarcity_K4'],'K4_candidates':['guided_greedy','guided_sample','scarcity_greedy','scarcity_sample'],
          'gating':'native SADCHER semantic checks must pass for cross-simulator claims',
          'evaluation':'offline simulator-assisted planning, not online policy'}
    spec_file=root/'protocol_locked.json'
    if spec_file.exists():
        old=json.loads(spec_file.read_text(encoding='utf8'))
        if old!=spec:raise RuntimeError('Protocol/checkpoint mismatch. Use NEW --out folder; do not overwrite tests.')
    else:atomic_json(spec_file,spec)
    path=root/'evaluation_per_case.csv';rows=[]
    if path.exists():
        with path.open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
        n_per_case=4
        if len(rows)%n_per_case!=0:raise RuntimeError('Partial/incomplete CSV: do not auto-skip')
    completed={int(r['case_index']) for r in rows}
    if any(sum(int(q['case_index'])==case for q in rows)!=4 for case in completed):
        raise RuntimeError('Corrupted resume case; inspect CSV')
    cases_dir=root/'cases';cases_dir.mkdir(exist_ok=True)
    plans_dir=root/'plans';plans_dir.mkdir(exist_ok=True)
    for i in range(args.cases):
        seed=int(args.seed_base+i*1009)
        problem=generate_case(orig,seed,args.tasks,args.robots,args.skills)
        sha=case_hash(problem)
        case_file=cases_dir/f'case_{i:04d}.json'
        if case_file.exists():
            saved=json.loads(case_file.read_text(encoding='utf8'))
            if saved['sha256']!=sha:raise RuntimeError(f'Case data changed since prior run: {i}')
        else:atomic_json(case_file,{'sha256':sha,'seed':seed,'data':problem})
        if i in completed:continue
        env,mapping=build_hmrta_env(problem)
        new=[]
        for model_name in MODELS:
            model=models[model_name]
            now=time.perf_counter()
            winner,base,candidates=sd.choose_candidates(env,model,
                seed+SEARCH_SEED_OFFSET,ALPHA[model_name],BETA[model_name])
            search_s=time.perf_counter()-now
            if len(candidates)!=4:raise AssertionError('not exactly 4 rollouts')
            for method,winner_tuple,K in (('guided_K1',base,1),('guided_scarcity_K4',winner,4)):
                selected=winner_tuple[1]
                rr=selected['result']
                replay_s=g.replay_and_verify(env,selected)
                native={'status':'not_run','reason':'Not requested'}
                if args.native and rr['done']:
                    try:
                        native=verify_native_sadcher(args.sadcher_root,problem,selected['trace'],mapping,rr['makespan'],args.max_native_ticks)
                    except Exception as ex:
                        native={'status':'runtime_error','reason':type(ex).__name__+': '+str(ex)[:300]}
                trace_path=plans_dir/f'case_{i:04d}_{model_name}_{method}.json'
                atomic_json(trace_path,{'case_sha256':sha,'model':model_name,
                    'K':K,'method':method,'episode_result':rr,
                    'agent_id_to_sadcher_id':mapping,'trace':selected['trace'],'routes':selected['routes'],
                    'native_cross_sim':native})
                new.append({'case_index':i,'case_seed':seed,'case_sha256':sha,
                    'generator':spec['generator'],'model':model_name,'method':method,'K':K,
                    'success':int(bool(rr['done'])),'makespan_hmrta':float(rr['makespan']),
                    'makespan_sadcher_units':float(rr['makespan'])*20,
                    'waiting_hmrta':float(rr['waiting_time']),'native_reward':float(rr['native_reward']),
                    'rollouts':K,'completed_candidates':sum(bool(x[1]['result']['done']) for x in candidates[:K]),
                    'search_elapsed_s':search_s if K==4 else float(base[1]['result']['runtime_s']),
                    'replay_elapsed_s':float(replay_s),'replay_verified':1,'n_decisions':len(selected['trace']),
                    'trace_sha256':trace_hash(selected['trace']),
                    'native_sadcher_status':native.get('status'),
                    'native_sadcher_ms':native.get('native_makespan',''),
                    'native_difference_ticks':native.get('difference_ticks',''),
                    'native_reason':native.get('reason','')})
        if len(new)!=4:raise AssertionError('Expected exactly 2 models x 2 methods')
        rows.extend(new);csv_atomic(path,rows,FIELDS)
        print(f'[CASE] {i+1}/{args.cases} seed={seed} StageA_K4={new[1]["makespan_hmrta"]:.3f} '
              f'Attention_K4={new[3]["makespan_hmrta"]:.3f} '
              f'native={new[1]["native_sadcher_status"]}/{new[3]["native_sadcher_status"]}',flush=True)
    summary,pairs=score(rows,root,bootstrap_draws=1000 if args.mirror_smoke else 5000)
    native_rows=[r for r in rows if r['method']=='guided_scarcity_K4']
    validated=sum(r['native_sadcher_status']=='aligned' for r in native_rows)
    attempted=sum(r['native_sadcher_status']!='not_run' for r in native_rows)
    gate={'original_generator_used':bool(orig),'native_check_requested':bool(args.native),
          'native_evaluations_attempted':attempted,'native_aligned':validated,
          'native_alignment_fraction':validated/attempted if attempted else None,
          'cross_simulator_equivalence_established':bool(args.native and attempted==len(native_rows) and validated==attempted),
          'conclusion':'Only cross-dataset SAME HeteroMRTA simulator evaluation is supported unless native equivalence gate passes.'}
    atomic_json(root/'compatibility_gate.json',gate)
    print('[GATE]',gate,flush=True)
    print('[SUMMARY]',summary,flush=True)
    print('[PAIRS]',pairs,flush=True)
    print('[DONE] outputs=',root,flush=True)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='/kaggle/working/SADCHER_CROSSDATASET_PROBE')
    p.add_argument('--sadcher-root',default='',help='Path to checkout of original published SADCHER repository')
    p.add_argument('--mirror-smoke',action='store_true',help='Debug only. Not eligible for cross-benchmark claims.')
    p.add_argument('--native',action='store_true',help='Optional native SADCHER timestep diagnostic, needs SADCHER source')
    p.add_argument('--cases',type=int,default=10)
    p.add_argument('--seed-base',type=int,default=GENERATOR_SEED)
    p.add_argument('--tasks',type=int,default=20)
    p.add_argument('--robots',type=int,default=15)
    p.add_argument('--skills',type=int,default=5)
    p.add_argument('--cpu-threads',type=int,default=1)
    p.add_argument('--max-native-ticks',type=int,default=30000)
    a=p.parse_args()
    if a.cases<1 or not (0<=a.seed_base<=2**32-1) or a.tasks<1 or a.robots<1 or a.skills!=5:
        raise ValueError('Bad dimensions: pretrained policies require exactly 5 skills')
    return a


if __name__=='__main__':run(parse_args())
