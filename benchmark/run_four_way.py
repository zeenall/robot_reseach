#!/usr/bin/env python3
"""Frozen Stage A: K1, K2, adaptive K2->K4, and original K4.

All four policies select from a shared set of *exact* original candidates:
 guided_greedy, guided_sample, scarcity_greedy, scarcity_sample.
 K1: [guided_greedy]
 K2: [guided_greedy, scarcity_greedy]
 K4: all four
 Adaptive: K2, add guided_sample + scarcity_sample only if a validation-frozen rule fires.

No model training. Held-out Mixed-501 and SADCHER-300 have been evaluated in
prior work; this is a *post-hoc* deployment/sensitivity experiment. Runtime
for all methods is reconstructed as the sum of individually TIMED candidate
executions (counterfactual for skipped candidates), not independently measured
end-to-end time. Live execution verification re-runs all methods for a small
subset, including a genuinely early-stopped Adaptive policy.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,platform,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'official'))
from train_selective_distillation import load_stage_a
from train_temporal_spatial_experiments import MODES,mixed_env
import scarcity_bottleneck_search as sc
import guided_coalition_search as g
from run_official_fullsystem import (load_reference,load_sadcher_cases,case_env,input_fingerprint,
                                    OFFSET_K4,sha,atomic_json)

VERSION='paper2_stage_a_four_methods_2026_11_15_v1'
VAL_BASE_SEED=2711150000
SAMPLE_SEED_STRIDE=7919
ALPHA=1.1
BETA=0.8
FAILURE_COST=200.0
METHODS=('K1','K2','Adaptive','K4')
K4_ORDER=('guided_greedy','guided_sample','scarcity_greedy','scarcity_sample')
VALIDATION_COLUMN=['mode','index','case_seed','input_sha256','k1_cost','k2_cost','k4_cost',
                   'greedy_relative_gap','k4_gain_over_k2','k1_time_s','k2_time_s','k4_time_s',
                   'k1_success','k2_success','k4_success']
TEST_COLUMNS=['cohort','case_id','mode','case_seed','input_sha256','method','complete',
  'makespan','penalized_makespan','waiting_time','rollouts_actual','time_estimated_s',
  'selected_candidate','k1_makespan','k2_makespan','k4_makespan',
  'greedy_relative_gap','adaptive_expanded','reference_k4_makespan','k4_reference_verified',
  'component_times_s','warning_time_estimate']


def csv_atomic(path, rows, cols):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
    os.replace(tmp,path)


def select_best(entries):
    return min(entries,key=lambda it:g._rank(it[1]))


def details(entry):
    r=entry[1]['result']
    return dict(name=entry[0],complete=int(bool(r['done'])),makespan=float(r['makespan']),
        cost=float(r['makespan']) if bool(r['done']) else FAILURE_COST,
        waiting=float(r['waiting_time']))


def candidate_specs(seed):
    return {
      'guided_greedy':(int(seed),False,False),
      'scarcity_greedy':(int(seed)+2*SAMPLE_SEED_STRIDE,True,False),
      'guided_sample':(int(seed)+SAMPLE_SEED_STRIDE,False,True),
      'scarcity_sample':(int(seed)+3*SAMPLE_SEED_STRIDE,True,True),
    }


def calculate_candidates(env,model,seed,names=None):
    """Run each original candidate once, separately timing its complete episode."""
    names=names or ('guided_greedy','scarcity_greedy','guided_sample','scarcity_sample')
    choice=sc.Choice(alpha=ALPHA,beta=BETA)
    spec=candidate_specs(seed)
    entries={};elapsed={}
    with torch.inference_mode():
      for name in names:
        seed_i,scarcity,sample=spec[name]
        tic=time.perf_counter()
        result=sc.candidate(env,model,seed_i,choice,scarcity=scarcity,sample=sample)
        elapsed[name]=time.perf_counter()-tic
        entries[name]=(name,result)
    return entries,elapsed


def summaries_candidates(entries,elapsed):
    k1=select_best([entries['guided_greedy']])
    k2=select_best([entries['guided_greedy'],entries['scarcity_greedy']])
    k4=select_best([entries[n] for n in K4_ORDER])
    one,two,four=details(k1),details(k2),details(k4)
    denom=max(1e-9,abs(details(entries['guided_greedy'])['cost']),
                       abs(details(entries['scarcity_greedy'])['cost']))
    gap=abs(details(entries['guided_greedy'])['cost']-details(entries['scarcity_greedy'])['cost'])/denom
    times={
        'K1':elapsed['guided_greedy'],
        'K2':elapsed['guided_greedy']+elapsed['scarcity_greedy'],
        'K4':sum(elapsed.values()),
    }
    return {'K1':one,'K2':two,'K4':four,'gap':float(gap),'times':times}


def should_expand(summary,rule):
    if not summary['K2']['complete']: return True
    if rule['direction']=='always_k2':return False
    if rule['direction']=='always_k4':return True
    if rule['direction']=='small_gap':return summary['gap']<=float(rule['threshold'])+1e-14
    if rule['direction']=='large_gap':return summary['gap']>=float(rule['threshold'])-1e-14
    raise ValueError('Unknown rule '+str(rule))


def val_seed(mode,index):
    return int(VAL_BASE_SEED + MODES.index(mode)*1000003 + index*1009)


def build_validation(model,n,out):
    p=out/'validation_candidates.csv'
    rows=[]
    if p.exists():
        with p.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    used={(r['mode'],int(r['index'])) for r in rows}
    target={(mode,i) for mode in MODES for i in range(n)}
    if used-target:raise RuntimeError('Existing validation file belongs to different n; use new --out')
    for mode in MODES:
      for i in range(n):
        if (mode,i) in used:continue
        seed=val_seed(mode,i)
        env=mixed_env(seed,mode);env.init_state()
        entries,times=calculate_candidates(env,model,seed+99331)
        sm=summaries_candidates(entries,times)
        row=dict(mode=mode,index=i,case_seed=seed,input_sha256=input_fingerprint(env),
          k1_cost=sm['K1']['cost'],k2_cost=sm['K2']['cost'],k4_cost=sm['K4']['cost'],
          greedy_relative_gap=sm['gap'],k4_gain_over_k2=sm['K2']['cost']-sm['K4']['cost'],
          k1_time_s=sm['times']['K1'],k2_time_s=sm['times']['K2'],k4_time_s=sm['times']['K4'],
          k1_success=sm['K1']['complete'],k2_success=sm['K2']['complete'],k4_success=sm['K4']['complete'])
        rows.append(row);used.add((mode,i));csv_atomic(p,rows,VALIDATION_COLUMN)
        if len(rows)%10==0:print('[VAL]',len(rows),'/',len(target),flush=True)
    return rows


def rules_for_validation(rows):
    gaps=np.array([float(r['greedy_relative_gap']) for r in rows])
    thresholds=sorted(set([0.,0.002,0.005,0.01,0.02,0.04,0.07,0.1,0.15,0.25,0.4,0.7,1.,
                           *[float(np.quantile(gaps,q)) for q in np.linspace(0,1,21)]]))
    rules=[dict(direction='always_k2',threshold=None),dict(direction='always_k4',threshold=None)]
    rules += [dict(direction=direction,threshold=t) for direction in ('small_gap','large_gap') for t in thresholds]
    return rules


def calibrate(rows,allowed_pct=0.5):
    k4=np.array([float(r['k4_cost']) for r in rows]);k2=np.array([float(r['k2_cost']) for r in rows])
    tm2=np.array([float(r['k2_time_s']) for r in rows]);tm4=np.array([float(r['k4_time_s']) for r in rows])
    avgk4=float(k4.mean()); allrules=[];eligible=[]
    for rule in rules_for_validation(rows):
      expand=np.array([should_expand({'K2':{'complete':int(r['k2_success'])},
                      'gap':float(r['greedy_relative_gap'])},rule) for r in rows],dtype=bool)
      costs=np.where(expand,k4,k2)
      runtime=np.where(expand,tm4,tm2)
      out={**rule,'n_val':len(rows),'mean_cost':float(costs.mean()),
           'loss_vs_K4_pct':float((costs.mean()/avgk4-1)*100),
           'mean_time_estimated_s':float(runtime.mean()),
           'expansion_rate':float(expand.mean()),
           'mean_rollouts':float(2+2*expand.mean()),
           'success_rate':float(np.mean(np.where(expand,
                 [int(r['k4_success']) for r in rows],
                 [int(r['k2_success']) for r in rows])))}
      allrules.append(out)
      if out['loss_vs_K4_pct']<=allowed_pct+1e-10 and out['success_rate']>=np.mean([int(r['k4_success']) for r in rows])-1e-10:
          eligible.append(out)
    if not eligible:raise RuntimeError('Even always K4 was not eligible: bug')
    # Frozen rule selected ONLY from independent validation, not 801 prior TEST cases.
    # Minimize measured time first, then cost and tie-break deterministic.
    choice=min(eligible,key=lambda r:(r['mean_time_estimated_s'],r['mean_cost'],
                                      {'always_k2':0,'small_gap':1,'large_gap':2,'always_k4':3}[r['direction']],
                                      r['threshold'] or 0))
    return choice,allrules



def select_method_on_validation(rows, rule, allowed_pct):
    """Choose the 4-way deployment choice using independent validation only.

    For all methods, require success >= K4 success and mean cost excess <=
    allowed_pct compared with K4; minimize candidate CPU time among qualifiers.
    This is hyperparameter selection, NOT weight training.
    """
    k4_cost=np.array([float(r['k4_cost']) for r in rows])
    k4_success=float(np.mean([int(r['k4_success']) for r in rows]))
    base=float(k4_cost.mean())
    candidates=[]
    for method in METHODS:
        costs=[];times=[];succ=[];rollouts=[]
        for r in rows:
            expand=should_expand({'K2':{'complete':int(r['k2_success'])},
                'gap':float(r['greedy_relative_gap'])},rule)
            m={'K1':'k1','K2':'k2','K4':'k4','Adaptive':'k4' if expand else 'k2'}[method]
            costs.append(float(r[m+'_cost']))
            times.append(float(r[m+'_time_s']))
            succ.append(int(r[m+'_success']))
            rollouts.append({'K1':1,'K2':2,'K4':4,'Adaptive':4 if expand else 2}[method])
        d=dict(method=method,n_val=len(rows),mean_cost=float(np.mean(costs)),
            mean_time_estimated_s=float(np.mean(times)),
            mean_rollouts=float(np.mean(rollouts)),success_rate=float(np.mean(succ)),
            mean_cost_excess_vs_K4_pct=100*(float(np.mean(costs))/base-1))
        d['eligible']=bool(d['mean_cost_excess_vs_K4_pct']<=allowed_pct+1e-10 and
                           d['success_rate']>=k4_success-1e-10)
        candidates.append(d)
    eligible=[c for c in candidates if c['eligible']]
    if not eligible:raise RuntimeError('K4 must be an eligible deployment option')
    chosen=min(eligible,key=lambda x:(x['mean_time_estimated_s'],x['mean_cost'],x['method']))
    return chosen,candidates


def bootstrap95(paired,cohort,n_draws=10000):
    rows=[r for r in paired if r['cohort']==cohort]
    groups=defaultdict(list)
    for r in rows: groups[r['mode']].append(float(r['delta_vs_K4']))
    rng=np.random.default_rng(20261115+(0 if cohort=='MIXED_501' else 1))
    samples=np.empty(n_draws)
    pools=[np.array(x) for x in groups.values()]
    for st in range(0,n_draws,250):
        num=min(250,n_draws-st)
        draws=np.concatenate([p[rng.integers(len(p),size=(num,len(p)))] for p in pools],axis=1)
        samples[st:st+num]=draws.mean(axis=1)
    return [float(x) for x in np.quantile(samples,[.025,.975])]


def make_summaries(rows,out,draws=10000):
    summary=[];paired=[]
    bycase=defaultdict(dict)
    for r in rows: bycase[(r['cohort'],r['case_id'])][r['method']]=r
    for key,rm in bycase.items():
      if set(rm)!=set(METHODS): continue
      for m in METHODS:
        if m=='K4':continue
        paired.append(dict(cohort=key[0],case_id=key[1],mode=rm[m]['mode'],method=m,
           delta_vs_K4=float(rm[m]['penalized_makespan'])-float(rm['K4']['penalized_makespan']),
           delta_time_vs_K4=float(rm[m]['time_estimated_s'])-float(rm['K4']['time_estimated_s'])))
    for cohort in ('MIXED_501','SADCHER_300'):
      for m in METHODS:
        rr=[r for r in rows if r['cohort']==cohort and r['method']==m]
        if not rr:continue
        ds=[float(r['penalized_makespan']) for r in rr]
        times=[float(r['time_estimated_s']) for r in rr]
        p=[r for r in paired if r['cohort']==cohort and r['method']==m]
        ci=bootstrap95(p,cohort,draws) if len(p)>3 else [None,None]
        summary.append(dict(cohort=cohort,method=m,n=len(rr),mean_penalized_makespan=float(np.mean(ds)),
           success_rate=float(np.mean([int(r['complete']) for r in rr])),
           mean_rollouts=float(np.mean([int(r['rollouts_actual']) for r in rr])),
           mean_time_estimated_s=float(np.mean(times)),
           mean_delta_vs_K4=float(np.mean([float(x['delta_vs_K4']) for x in p])) if p else 0.,
           delta_vs_K4_CI95_low=ci[0],delta_vs_K4_CI95_high=ci[1],
           k4_reference_matches=sum(int(r['k4_reference_verified']) for r in rr),
           runtime_label='sum of separately timed candidate calls, not standalone end-to-end'))
    if summary:csv_atomic(out/'summary.csv',summary,list(summary[0]))
    if paired:csv_atomic(out/'paired_vs_K4.csv',paired,list(paired[0]))
    return summary


def independent_live_verify(env,model,seed,rule,expected,case_label,out_rows):
    """Actually execute each policy independently for selected cases, ensuring early stop."""
    subset={'K1':('guided_greedy',),'K2':('guided_greedy','scarcity_greedy'),
            'K4':('guided_greedy','scarcity_greedy','guided_sample','scarcity_sample')}
    for m,names in subset.items():
      tic=time.perf_counter();entries,_=calculate_candidates(env,model,seed,names)
      actual_wall=time.perf_counter()-tic
      best=details(select_best(list(entries.values())))
      exp=expected[m]
      if best['complete']!=exp['complete'] or not math.isclose(best['makespan'],exp['makespan'],rel_tol=0,abs_tol=1e-6):
        raise RuntimeError(f'LIVE VERIFY FAILED {case_label} {m} {best} != {exp}')
      out_rows.append(dict(case=case_label,method=m,actual_rollouts=len(names),actual_wall_s=actual_wall,
            derived_candidate_sum_s=exp['time_estimated_s'],result_makespan=best['makespan'],verified=1))
    tic=time.perf_counter()
    entries,tm=calculate_candidates(env,model,seed,subset['K2'])
    k2=select_best(list(entries.values()));pair=details(k2)
    a=details(entries['guided_greedy']);b=details(entries['scarcity_greedy'])
    gap=abs(a['cost']-b['cost'])/max(abs(a['cost']),abs(b['cost']),1e-9)
    expand=should_expand({'K2':pair,'gap':gap},rule)
    if expand:
        extra,_=calculate_candidates(env,model,seed,('guided_sample','scarcity_sample'))
        entries.update(extra)
    actual_wall=time.perf_counter()-tic
    best=details(select_best(list(entries.values())))
    exp=expected['Adaptive']
    if best['complete']!=exp['complete'] or not math.isclose(best['makespan'],exp['makespan'],rel_tol=0,abs_tol=1e-6):
      raise RuntimeError(f'LIVE ADAPTIVE FAILED {case_label}: {best} != {exp}')
    if len(entries)!=int(exp['rollouts_actual']):raise RuntimeError('Adaptive did not early-stop correctly')
    out_rows.append(dict(case=case_label,method='Adaptive',actual_rollouts=len(entries),
       actual_wall_s=actual_wall,derived_candidate_sum_s=exp['time_estimated_s'],
       result_makespan=best['makespan'],verified=1))


def run(args):
    if args.threads<1:raise ValueError('threads >= 1')
    if not (0<=args.mixed_per_mode<=167 and 0<=args.sadcher_count<=300):raise ValueError('Requested cases exceed frozen cohorts')
    if args.val_per_mode<1:raise ValueError('val-per-mode >= 1')
    if not (0<=args.adaptive_max_loss_pct<=5):raise ValueError('max loss must be between 0 and 5 percent')
    torch.set_num_threads(args.threads)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    weights=ROOT/'weights'/'stage_a_best_model.pt'
    stage=load_stage_a(weights).eval().cpu()
    for p in stage.parameters():p.requires_grad_(False)
    ref=load_reference();sadcher=load_sadcher_cases()
    proto=dict(version=VERSION,weights_sha256=sha(weights),ref_sha256=sha(ROOT/'reference'/'stage_a_K4_801.csv'),
       cases_sha256=sha(ROOT/'cases'/'sadcher300_official_original_cases.json.xz'),
       val_seed_base=VAL_BASE_SEED,val_per_mode=args.val_per_mode,
       adaptive_max_loss_pct=args.adaptive_max_loss_pct,
       mixed_per_mode=args.mixed_per_mode,sadcher_count=args.sadcher_count,
       alpha=ALPHA,beta=BETA,threads=args.threads,python=platform.python_version(),torch=torch.__version__,
       no_training=True,time_measurement='sum separately measured candidate runtimes')
    lock=out/'protocol.json'
    if lock.exists():
        old=json.loads(lock.read_text(encoding='utf8'))
        if old!=proto:raise RuntimeError('Resume options differ from protocol lock; use NEW output directory')
    else:atomic_json(proto,lock)
    print('[MODEL] Stage A frozen',sum(p.numel() for p in stage.parameters()),'weights_sha256',proto['weights_sha256'],flush=True)
    val=build_validation(stage,args.val_per_mode,out)
    rule,allrules=calibrate(val,args.adaptive_max_loss_pct)
    csv_atomic(out/'adaptive_candidate_rules_validation.csv',allrules,list(allrules[0]))
    frozen_path=out/'adaptive_frozen_rule.json'
    if frozen_path.exists():
      prev=json.loads(frozen_path.read_text(encoding='utf8'))
      if prev!=rule:raise RuntimeError('Frozen adaptive rule changed: corrupt validation data')
    else:atomic_json(rule,frozen_path)
    print('[ADAPTIVE RULE FROM VALIDATION ONLY]',json.dumps(rule,ensure_ascii=False),flush=True)
    selected,method_val=select_method_on_validation(val,rule,args.adaptive_max_loss_pct)
    csv_atomic(out/'validation_four_method_selection.csv',method_val,list(method_val[0]))
    select_path=out/'validation_selected_method.json'
    if select_path.exists():
        if json.loads(select_path.read_text(encoding='utf8'))!=selected:
            raise RuntimeError('Frozen method selection changed. Do not resume.')
    else:atomic_json(selected,select_path)
    print('[VALIDATION CHOSEN METHOD]',json.dumps(selected,ensure_ascii=False),flush=True)
    planned=[]
    if args.mixed_per_mode:
      planned += [ref['MIXED_501',f'{mode}:{i}'] for mode in MODES for i in range(args.mixed_per_mode)]
    if args.sadcher_count:
      planned += [ref['SADCHER_300',str(i)] for i in range(args.sadcher_count)]
    p=out/'per_case_all_methods.csv'
    if p.exists():
        with p.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    else:rows=[]
    counted=set()
    for r in rows:
        key=(r['cohort'],r['case_id'],r['method'])
        if key in counted:raise RuntimeError('Duplicate persisted result '+str(key))
        counted.add(key)
    expected_keys={(r['cohort'],r['case_id'],m) for r in planned for m in METHODS}
    if not counted.issubset(expected_keys):raise RuntimeError('Persisted results do not belong to current configuration')
    if any(set(m for co,ci,m in counted if (co,ci)==(r['cohort'],r['case_id'])) not in (set(),set(METHODS)) for r in planned):
       raise RuntimeError('Partially saved case detected; refusing to silently resume')
    live_rows=[]
    for i,rr in enumerate(planned):
      co,ci=rr['cohort'],rr['case_id']
      if (co,ci,'K4') in counted:continue
      env=case_env(rr,sadcher)
      digest=input_fingerprint(env)
      seed=int(rr['case_seed'])+OFFSET_K4[co]
      entries,times=calculate_candidates(env,stage,seed)
      sm=summaries_candidates(entries,times)
      # Mandatory reproducibility guard: if even one K4 result diverges, stop.
      ref_m=float(rr['makespan']);our_m=sm['K4']['makespan']
      if not math.isclose(ref_m,our_m,rel_tol=0,abs_tol=1e-6) or int(rr['complete'])!=sm['K4']['complete']:
         atomic_json({'case':[co,ci],'reference':rr,'ourK4':sm['K4'],'sha':digest},out/'STOP_K4_MISMATCH.json')
         raise RuntimeError(f'K4 MISMATCH for {co}/{ci}: actual={our_m} old={ref_m}')
      expand=should_expand(sm,rule)
      for method in METHODS:
         if method=='Adaptive':
           result=sm['K4'] if expand else sm['K2']
           used=4 if expand else 2
           est=sm['times']['K4'] if expand else sm['times']['K2']
         else:
           result=sm[method]
           used={'K1':1,'K2':2,'K4':4}[method]
           est=sm['times'][method]
         rows.append(dict(cohort=co,case_id=ci,mode=rr['mode'],case_seed=rr['case_seed'],
           input_sha256=digest,method=method,complete=result['complete'],makespan=result['makespan'],
           penalized_makespan=result['cost'],waiting_time=result['waiting'],
           rollouts_actual=used,time_estimated_s=est,selected_candidate=result['name'],
           k1_makespan=sm['K1']['makespan'],k2_makespan=sm['K2']['makespan'],k4_makespan=sm['K4']['makespan'],
           greedy_relative_gap=sm['gap'],adaptive_expanded=int(expand),
           reference_k4_makespan=ref_m,k4_reference_verified=1,
           component_times_s=json.dumps(times,sort_keys=True),
           warning_time_estimate='Derived from measured candidate calls; do NOT label independent wall time'))
      csv_atomic(p,rows,TEST_COLUMNS)
      if args.live_verify_n>0:
        cohort_index=sum(1 for x in planned[:i+1] if x['cohort']==co)-1
        if cohort_index<args.live_verify_n:
          expected={r['method']:r for r in rows[-4:]}
          independent_live_verify(env,stage,seed,rule,expected,co+':'+ci,live_rows)
          csv_atomic(out/'live_runtime_verification.csv',live_rows,list(live_rows[0]))
      if (i+1)%10==0 or (i+1)==len(planned):
         print('[EVAL]',i+1,'/',len(planned),co,ci,'K1',round(sm['K1']['makespan'],4),
               'K2',round(sm['K2']['makespan'],4),'K4',round(sm['K4']['makespan'],4),
               'adaptive_rollouts',4 if expand else 2,flush=True)
    summary=make_summaries(rows,out,args.bootstrap_draws)
    atomic_json(dict(protocol=proto,adaptive_rule=rule,method_preselected_on_validation=selected,calibration_cases=len(val),n_test_cases=len(planned),
         n_rows=len(rows),all_k4_matches_reference=True,summary=summary,
         caution='This is post-hoc research. Time is reconstructed additive candidate time; use live_runtime_verification.csv for actual independent timing.'),
         out/'final_report.json')
    print('[DONE] summary:',json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--val-per-mode',type=int,default=24)
    ap.add_argument('--mixed-per-mode',type=int,default=167)
    ap.add_argument('--sadcher-count',type=int,default=300)
    ap.add_argument('--adaptive-max-loss-pct',type=float,default=0.5)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--live-verify-n',type=int,default=2)
    ap.add_argument('--bootstrap-draws',type=int,default=10000)
    ap.add_argument('--out',default=str(ROOT/'outputs'/'four_methods_full'))
    args=ap.parse_args();run(args)

if __name__=='__main__':main()
