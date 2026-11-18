"""Strict, paired, inference-only best-of-K test-time rollout comparison.

The rollouts simulate COMPLETE TaskEnv schedules. This is an OFFLINE simulator-
access method; it is not an online one-step action-selection comparison. Every
search-enabled model gets the identical rollout *count* K and common instance
and RNG seed plan. We report total CPU runtime; equal K != equal wall time.
The reused RALTestSet is DIAGNOSTIC and not used for training/model selection.
"""
from __future__ import annotations
import copy,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from attention import AttentionNet
from warm_relational_kan import WarmRelationalKAN,count_parameters
from coalition_arrival_kan import CoalitionArrivalKAN
from train_temporal_spatial_experiments import mixed_env,MODES,Limits,official_env
from train_warm_hybrid import write_csv
from benchmark_core import run_episode

HERE=Path(__file__).resolve().parent
TEST_SEED_START=987001000
SEARCH_MODELS={'stage_a','control','arrival','waiting','attention'}


def load_models(out):
    from train_coordination_ablations import load_stage_a, ARM_NAMES
    models={'original_hybrid':WarmRelationalKAN(),'stage_a':load_stage_a()}
    orig=torch.load(HERE/'reference'/'best_hybrid.pt',map_location='cpu',weights_only=True)
    if orig.get('model_name')!='WarmRelationalKAN':raise RuntimeError('Missing original hybrid source')
    models['original_hybrid'].load_state_dict(orig['weights'],strict=True)
    for arm in ARM_NAMES:
        path=out/arm/'best_model.pt'
        if not path.exists():raise FileNotFoundError(f'All three arms must finish before final eval: {path}')
        ck=torch.load(path,map_location='cpu',weights_only=True)
        if ck.get('arm')!=arm or ck.get('best_job') is None:
            raise RuntimeError(f'Unexpected checkpoint identity {path}')
        net=CoalitionArrivalKAN() if arm=='arrival' else WarmRelationalKAN()
        net.load_state_dict(ck['weights'],strict=True)
        models[arm]=net
    att=AttentionNet(11,15,128)
    pub=torch.load(HERE/'weights'/'AttentionNet_published.pt',map_location='cpu',weights_only=True)
    if pub.get('model_name')!='AttentionNet_published':raise RuntimeError('Incorrect published checkpoint')
    att.load_state_dict(pub['weights'],strict=True)
    models['attention']=att
    for m in models.values():m.cpu().eval()
    return models


def simulated_search(template,model,case_seed,k):
    """Return best completed simulated schedule (greedy + k-1 samples).

    Candidate objective is penalized makespan (200 if not completed), then wait;
    no validation/test labels or checkpoints are consulted to choose a rollout.
    Runs models on the same TaskEnv instance and no gradients. Each run uses
    a deterministic RNG fork for torch sample choices and independent shuffled
    robot release order; the (greedy) first rollout is always included.
    """
    best=None;best_key=None;attempts=[]
    start=time.perf_counter()
    for j in range(k):
        # Corresponding candidates across models get the same numeric seeds.
        # Sampling remains policy-dependent by design.
        seed=int(case_seed)+j*7919
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            r,_=run_episode(copy.deepcopy(template),model,Limits,torch.device('cpu'),
                            seed=seed,sample=(j>0),record=False)
        candidate=(0 if r['done'] else 1,
                   float(r['makespan']) if r['done'] else 200.,
                   float(r['waiting_time']))
        attempts.append({'try':j,'done':int(r['done']),'makespan':float(r['makespan'])})
        if best_key is None or candidate<best_key:
            best_key=candidate;best=dict(r);best['selected_rollout']=j
    best['runtime_total_s']=float(time.perf_counter()-start)
    best['rollouts']=k
    best['penalized_makespan']=float(best['makespan']) if best['done'] else 200.
    return best


def paired_bootstrap(diff,seed=1732,reps=3000):
    arr=np.asarray(diff,dtype=np.float64)
    if len(arr)<2:return [None,None]
    rng=np.random.default_rng(seed)
    draws=[]
    for start in range(0,reps,100):
        idx=rng.integers(0,len(arr),size=(min(100,reps-start),len(arr)))
        draws.extend(arr[idx].mean(1).tolist())
    return [float(x) for x in np.quantile(draws,[.025,.975])]


def evaluate_final(out,args):
    torch.set_num_threads(1)
    models=load_models(out)
    names=list(models)
    specs={mode:[TEST_SEED_START+1000003*j+1009*i for i in range(args.test_each)]
           for j,mode in enumerate(MODES)}
    rows=[]
    print(f'[TEST SETUP] models={names} heldout_per_mode={args.test_each} '
          f'official_DIAGNOSTIC={args.official_cases} search_K={args.search_k}',flush=True)
    # Skip an already completed evaluation ONLY if all fingerprint-relevant fields match.
    params={'search_k':args.search_k,'test_each':args.test_each,
            'official_cases':args.official_cases,'test_seed_start':TEST_SEED_START,
            'model_checkpoints':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(out.glob('*/best_model.pt'))},
            'stage_a':hashlib.sha256((HERE/'reference'/'stage_a_best_model.pt').read_bytes()).hexdigest()}
    state_path=out/'evaluation_provenance.json'
    if state_path.exists() and (out/'evaluation_summary.csv').exists():
        try:
            old=json.loads(state_path.read_text())
            if old==params:
                print('[TEST] Matching finished evaluation already on disk, no recompute.',flush=True)
                return
        except Exception:pass
    with torch.inference_mode():
        allsets=[(f'new_heldout_{mode}',[(i,int(s),mixed_env(s,mode))
                     for i,s in enumerate(seeds)]) for mode,seeds in specs.items()]
        # Explicitly load reused official only for diagnostic performance reporting.
        if args.official_cases:
            allsets.append(('reused_official_DIAGNOSTIC',
                [(i,f'env_{i}',official_env(i)) for i in range(args.official_cases)]))
        for dataset,cases in allsets:
            print(f'[TEST] {dataset} cases={len(cases)}',flush=True)
            for i,seed,env in cases:
                env.init_state()
                for name,model in models.items():
                    # Only certain models use K rollouts; all models get greedy reference.
                    for k in ([1,args.search_k] if name in SEARCH_MODELS and args.search_k>1 else [1]):
                        r=simulated_search(env,model,case_seed=30361000+i+1000003*len(dataset),k=k)
                        rows.append({'dataset':dataset,'case':i,'seed':seed,
                            'model':name,'method':'greedy' if k==1 else f'best_of_{k}',
                            'parameters':count_parameters(model),**r})
                if (i+1)%10==0:
                    print(f'[TEST] {dataset} {i+1}/{len(cases)}',flush=True)
    write_csv(out/'evaluation_per_case.csv',rows)
    keys=sorted(set((r['dataset'],r['model'],r['method']) for r in rows))
    summary=[]
    for ds,name,method in keys:
        sub=[r for r in rows if r['dataset']==ds and r['model']==name and r['method']==method]
        done=[r for r in sub if r['done']]
        summary.append({'dataset':ds,'model':name,'method':method,'cases':len(sub),
            'completed':len(done),'success_rate':len(done)/len(sub),
            'penalized_makespan':float(np.mean([x['penalized_makespan'] for x in sub])),
            'makespan_completed':float(np.mean([x['makespan'] for x in done])) if done else None,
            'mean_waiting':float(np.mean([x['waiting_time'] for x in sub])),
            'mean_native_reward':float(np.mean([x['native_reward'] for x in sub])),
            'mean_reward_minus_0_2_wait':float(np.mean([x['native_reward']-0.2*x['waiting_time'] for x in sub])),
            'mean_runtime_total_s':float(np.mean([x['runtime_total_s'] for x in sub])),
            'parameters':sub[0]['parameters']})
    write_csv(out/'evaluation_summary.csv',summary)
    by_key={(r['dataset'],r['case'],r['model'],r['method']):r for r in rows}
    comparisons=[];deltas=[]
    for ds in sorted(set(r['dataset'] for r in rows)):
        ids=sorted(set(r['case'] for r in rows if r['dataset']==ds))
        available={(r['model'],r['method']) for r in rows if r['dataset']==ds}
        pair_specs=[(('control','greedy'),('stage_a','greedy')),
          (('arrival','greedy'),('control','greedy')),
          (('waiting','greedy'),('control','greedy')),
          (('arrival','greedy'),('stage_a','greedy')),
          (('waiting','greedy'),('stage_a','greedy')),
          (('arrival','greedy'),('attention','greedy')),
          (('waiting','greedy'),('attention','greedy')),
          (('stage_a','greedy'),('attention','greedy')),
          (('stage_a','greedy'),('original_hybrid','greedy'))]
        if args.search_k>1:
            m=f'best_of_{args.search_k}'
            pair_specs += [((name,m),(name,'greedy')) for name in SEARCH_MODELS]
            pair_specs += [((name,m),('attention',m)) for name in ('arrival','waiting','stage_a','control')]
        for (a,ma),(b,mb) in pair_specs:
            if (a,ma) not in available or (b,mb) not in available:continue
            values=[];a_success=0;b_success=0
            for i in ids:
                x=by_key[ds,i,a,ma];y=by_key[ds,i,b,mb]
                d=float(x['penalized_makespan']-y['penalized_makespan'])
                values.append(d)
                a_success+=int(x['done']);b_success+=int(y['done'])
                deltas.append({'dataset':ds,'case':i,'model_a':a,'method_a':ma,
                    'model_b':b,'method_b':mb,'a_penalized':x['penalized_makespan'],
                    'b_penalized':y['penalized_makespan'],'a_done':x['done'],
                    'b_done':y['done'],'delta_a_minus_b':d})
            comparisons.append({'dataset':ds,'model_a':a,'method_a':ma,
                'model_b':b,'method_b':mb,'cases':len(ids),
                'a_completed':a_success,'b_completed':b_success,
                'delta_penalized_makespan_a_minus_b':float(np.mean(values)),
                'ci95_paired_bootstrap_exploratory':str(paired_bootstrap(values))})
    write_csv(out/'paired_per_case.csv',deltas)
    write_csv(out/'paired_statistics.csv',comparisons)
    state_path.write_text(json.dumps(params,indent=2))
    print('[EVALUATION SUMMARY]\n'+json.dumps(summary,indent=2),flush=True)
    print('[DONE] saved paired comparisons (CI exploratory; repeated tuning/multiple comparisons)',flush=True)
