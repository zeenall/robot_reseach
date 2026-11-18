#!/usr/bin/env python3
"""Direct, independent end-to-end CPU runtime comparison for Paper 2.

Methods: frozen Stage A K1, K2, Adaptive K2->K4, K4 and official native
AttentionNet greedy and sampled-best-of-10. Identical HeteroMRTA instances;
prior validation-frozen Adaptive rule; NO TRAINING. This is post-hoc.

Native authors' 'sampling=True' runs 10 *sampled* episodes, not greedy+9.
All methods are measured separately, same machine/session and single thread.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,platform,random,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'official'))
from train_selective_distillation import load_stage_a
from train_temporal_spatial_experiments import MODES
from attention import AttentionNet
import run_four_way as fw
from run_official_fullsystem import (load_reference,load_sadcher_cases,case_env,
  input_fingerprint, OFFSET_K4,OFFSET_NATIVE_SEED,run_native,sha,
  verify_official_source,atomic_json,run_native_once)

METHODS=('StageA_K1','StageA_K2','StageA_Adaptive','StageA_K4',
         'AttentionNet_Greedy1','AttentionNet_Sampling10')
CSV_COLUMNS=('cohort','case_id','mode','case_seed','input_sha256','method',
  'complete','makespan','penalized_makespan','wall_s','rollouts',
  'matched_results36','matched_K4_reference','adaptive_expanded','choice',
  'timed_out_rollouts','measurement','case_order')
VERSION='paper2_direct_wall_6methods_2026_11_15_v1'


def append_atomic_csv(path,rows):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=CSV_COLUMNS)
        w.writeheader(); w.writerows(rows)
    os.replace(tmp,path)


def load_previous_36():
    p=ROOT/'reference'/'four_way_results36_compact.csv'
    with p.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    d={(r['cohort'],r['case_id'],r['method']):r for r in rows}
    if len(rows)!=3204 or len(d)!=3204:
        raise RuntimeError('Results 36 reference incomplete or duplicate')
    return d


def load_adaptive_rule():
    p=ROOT/'reference'/'adaptive_frozen_rule_results36.json'
    rule=json.loads(p.read_text(encoding='utf-8'))
    if rule['direction']!='small_gap' or not math.isclose(float(rule['threshold']),0.07538408036670655,abs_tol=1e-14):
        raise RuntimeError('Adaptive was modified after validation; stop.')
    if int(rule['n_val'])!=72:
        raise RuntimeError('Adaptive validation count differs; stop.')
    return rule


def measure_stage(method,env,model,seed,rule):
    if method=='StageA_K1':names=('guided_greedy',)
    elif method=='StageA_K2':names=('guided_greedy','scarcity_greedy')
    elif method=='StageA_K4':names=('guided_greedy','scarcity_greedy','guided_sample','scarcity_sample')
    elif method=='StageA_Adaptive':names=('guided_greedy','scarcity_greedy')
    else:raise ValueError(method)
    # Timer includes complete policy execution (candidate env copies, policy, search,
    # action selection) and *excludes* model load + initial case generation, for all.
    start=time.perf_counter()
    entries,_=fw.calculate_candidates(env,model,seed,names)
    expand=False
    if method=='StageA_Adaptive':
        first=fw.select_best([entries['guided_greedy'],entries['scarcity_greedy']])
        a=fw.details(entries['guided_greedy'])['cost']
        b=fw.details(entries['scarcity_greedy'])['cost']
        gap=abs(a-b)/max(1e-9,abs(a),abs(b))
        # The rule sees only initial greedy outcomes, never unexecuted sample outcomes.
        expand=fw.should_expand({'K2':fw.details(first),'gap':gap},rule)
        if expand:
            extra,_=fw.calculate_candidates(env,model,seed,('guided_sample','scarcity_sample'))
            entries.update(extra)
    best=fw.details(fw.select_best(list(entries.values())))
    wall=time.perf_counter()-start
    return dict(complete=best['complete'],makespan=best['makespan'],
        penalized_makespan=best['cost'],wall_s=wall,rollouts=len(entries),
        adaptive_expanded=int(expand),choice=best['name'],timed_out_rollouts=0)


def measure_native(method,env,model,seed,timeout):
    # Ten independent Categorical-sampled complete episodes, or one greedy
    # episode. A rollout hitting the predeclared limit counts as attempted but
    # unsuccessful; never silently discard or substitute a greedy outcome.
    start=time.perf_counter()
    sample=(method=='AttentionNet_Sampling10')
    n=10 if sample else 1
    outcomes=[];timeouts=0
    for j in range(n):
        try:
            res=run_native_once(env,model,seed+j*7919,sample=sample,time_limit=timeout)
            outcomes.append(res)
        except TimeoutError:
            timeouts+=1
    wall=time.perf_counter()-start
    if outcomes:
        best=min(outcomes,key=lambda r:(r['penalized_makespan'],r['makespan']))
        complete=best['complete'];mk=best['makespan'];cost=best['penalized_makespan']
    else:
        complete=0;mk=200.;cost=200.
    return dict(complete=complete,makespan=mk,penalized_makespan=cost,
        wall_s=wall,rollouts=n,adaptive_expanded='',
        choice=('sampling10' if sample else 'greedy'),timed_out_rollouts=timeouts)

def summarize(rows,out):
    grouped=defaultdict(list)
    for r in rows: grouped[(r['cohort'],r['method'])].append(r)
    summary=[]
    for (co,method),rr in sorted(grouped.items()):
        mm=np.array([float(r['makespan']) for r in rr]);cost=np.array([float(r['penalized_makespan']) for r in rr]);tt=np.array([float(r['wall_s']) for r in rr])
        summary.append(dict(cohort=co,method=method,n=len(rr),mean_makespan=float(np.mean(mm)),
            mean_penalized_makespan=float(np.mean(cost)),
            success_rate=float(np.mean([int(r['complete']) for r in rr])),
            mean_wall_s=float(tt.mean()),median_wall_s=float(np.median(tt)),
            mean_rollouts=float(np.mean([int(r['rollouts']) for r in rr])),
            timeout_rollouts_total=sum(int(r['timed_out_rollouts']) for r in rr),
            min_wall_s=float(tt.min()),max_wall_s=float(tt.max()),
            timing='direct independent policy wall; excludes model loading and case generation'))
    with (out/'summary_direct_wall.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    return summary


def bootstrap_pairs(rows,out,draws=10000):
    bycase=defaultdict(dict)
    for r in rows:bycase[(r['cohort'],r['case_id'])][r['method']]=r
    output=[]
    for cohort in ('MIXED_501','SADCHER_300'):
        completed=[(ci,d) for (co,ci),d in bycase.items() if co==cohort and set(d)==set(METHODS)]
        if len(completed)<3:continue
        for method in METHODS:
            if method=='StageA_K4':continue
            deltas_by_mode=defaultdict(list);times=[]
            for ci,d in completed:
                mode=d['StageA_K4']['mode']
                deltas_by_mode[mode].append(float(d[method]['penalized_makespan'])-float(d['StageA_K4']['penalized_makespan']))
                times.append(float(d[method]['wall_s'])-float(d['StageA_K4']['wall_s']))
            rng=np.random.default_rng(20261115+len(method)+(0 if cohort=='MIXED_501' else 100))
            pools=[np.asarray(x) for x in deltas_by_mode.values()]
            boots=np.zeros(draws)
            for st in range(0,draws,200):
                k=min(200,draws-st)
                boot=np.concatenate([p[rng.integers(0,len(p),size=(k,len(p)))] for p in pools],axis=1)
                boots[st:st+k]=boot.mean(axis=1)
            lo,hi=np.quantile(boots,[.025,.975]);combined=np.concatenate(pools)
            output.append(dict(cohort=cohort,method=method,n=len(completed),
              paired_delta_method_minus_K4=float(combined.mean()),
              paired_CI95_low=float(lo),paired_CI95_high=float(hi),
              mean_wall_delta_method_minus_K4_s=float(np.mean(times))))
    if output:
        with (out/'paired_vs_K4.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)


def run(args):
    if args.threads<1:raise ValueError('CPU_THREADS must be >=1')
    torch.set_num_threads(args.threads)
    verify_official_source()
    weights=ROOT/'weights'/'stage_a_best_model.pt'
    official=ROOT/'weights'/'AttentionNet_published.pt'
    if not official.is_file():raise FileNotFoundError('AttentionNet weights absent; first run GitHub download/verify cell')
    stage=load_stage_a(weights).cpu().eval()
    ckpt=torch.load(official,map_location='cpu',weights_only=True)
    if ckpt.get('model_name')!='AttentionNet_published':raise RuntimeError('Unrecognized official checkpoint')
    att=AttentionNet(11,15,128).cpu().eval()
    att.load_state_dict(ckpt['weights'],strict=True)
    for model in (stage,att):
        for p in model.parameters():p.requires_grad_(False)
    refs=load_reference();sadcher=load_sadcher_cases();previous=load_previous_36();rule=load_adaptive_rule()
    planned=[]
    if args.mixed_per_mode:
        planned.extend(refs['MIXED_501',f'{mode}:{i}'] for mode in MODES for i in range(args.mixed_per_mode))
    if args.sadcher_count:
        planned.extend(refs['SADCHER_300',str(i)] for i in range(args.sadcher_count))
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    proto=dict(version=VERSION+'b',methods=list(METHODS),mixed_per_mode=args.mixed_per_mode,
        sadcher_count=args.sadcher_count,threads=args.threads,timeout=args.native_timeout,
        no_training=True,adaptive_frozen=rule,
        stage_weights=sha(weights),official_weights=sha(official),
        ref36_sha=sha(ROOT/'reference'/'four_way_results36_compact.csv'),
        case_sha=sha(ROOT/'cases'/'sadcher300_official_original_cases.json.xz'),
        mixed_reference_sha=sha(ROOT/'reference'/'stage_a_K4_801.csv'),
        cpu=platform.processor(),python=platform.python_version(),torch=torch.__version__,
        wall='perf_counter around each entire inference; pretrained model loads excluded',
        native_timeout_handling='any timed-out native rollout is counted attempted but unsuccessful',
        official_sampling='10 independent categorical sampled episodes; best selected',
        test_status='post-hoc reused 801 cases')
    pproto=out/'protocol_locked.json'
    if pproto.exists() and json.loads(pproto.read_text(encoding='utf-8'))!=proto:
        raise RuntimeError('Protocol changed on resume. Use new output directory.')
    atomic_json(proto,pproto)
    p=out/'per_case_direct_wall.csv'
    if p.exists():
        with p.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    else:rows=[]
    seen=set()
    for r in rows:
        k=(r['cohort'],r['case_id'],r['method'])
        if k in seen:raise RuntimeError('Duplicate stored row '+str(k))
        seen.add(k)
    for idx,reference in enumerate(planned):
        key=(reference['cohort'],reference['case_id'])
        existing={x[2] for x in seen if x[:2]==key}
        if existing==set(METHODS):continue
        if existing:raise RuntimeError('Incomplete case on disk. Use a new output directory: '+str(key))
        e=case_env(reference,sadcher)
        fingerprint=input_fingerprint(e)
        searchseed=int(reference['case_seed'])+OFFSET_K4[reference['cohort']]
        nativeseed=int(reference['case_seed'])+OFFSET_NATIVE_SEED
        # Rotate evaluation order to reduce systematic warm-cache/thermal bias.
        shifted=METHODS[idx%len(METHODS):]+METHODS[:idx%len(METHODS)]
        local=[]
        print('[CASE]',idx+1,'/',len(planned),key,flush=True)
        with torch.inference_mode():
            for method in shifted:
                print('  [METHOD]',method,flush=True)
                if method.startswith('StageA_'):
                    got=measure_stage(method,e,stage,searchseed,rule)
                    earlier=previous[reference['cohort'],reference['case_id'],method.removeprefix('StageA_')]
                    expected=float(earlier['makespan'])
                    matched=int(int(got['complete'])==int(earlier['complete']) and
                      math.isclose(got['makespan'],expected,abs_tol=1e-6,rel_tol=0))
                    if not matched:
                        atomic_json({'case':key,'method':method,'current':got,'previous36':earlier},out/'STOP_MISMATCH.json')
                        raise RuntimeError(f'Stage A {method} result changed on case {key}: {got["makespan"]} vs {expected}')
                    k4match=1 if method!='StageA_K4' else int(math.isclose(got['makespan'],float(reference['makespan']),abs_tol=1e-6,rel_tol=0))
                    if not k4match:raise RuntimeError('Stage A K4 original reference mismatch')
                else:
                    got=measure_native(method,e,att,nativeseed,args.native_timeout)
                    matched='';k4match=''
                r=dict(cohort=reference['cohort'],case_id=reference['case_id'],mode=reference['mode'],
                  case_seed=reference['case_seed'],input_sha256=fingerprint,method=method,
                  complete=int(got['complete']),makespan=got['makespan'],
                  penalized_makespan=got['penalized_makespan'],wall_s=got['wall_s'],
                  rollouts=got['rollouts'],matched_results36=matched,
                  matched_K4_reference=k4match,adaptive_expanded=got['adaptive_expanded'],
                  choice=got['choice'],timed_out_rollouts=got['timed_out_rollouts'],measurement='direct independent end-to-end, no load',
                  case_order='>'.join(shifted))
                local.append(r)
        rows.extend(local);seen.update((r['cohort'],r['case_id'],r['method']) for r in local)
        append_atomic_csv(p,rows)
        if (idx+1)%10==0 or idx+1==len(planned):
            vals={r['method']:round(float(r['makespan']),3) for r in local}
            print('[DONE]',idx+1,'/',len(planned),key,vals,flush=True)
    summary=summarize(rows,out)
    bootstrap_pairs(rows,out,args.bootstrap_draws)
    atomic_json(dict(n_cases=len(planned),n_rows=len(rows),all_stage_matches_results36=all(str(r['matched_results36'])=='1' for r in rows if r['method'].startswith('StageA_')),
       adaptive_rule=rule,post_hoc=True,summary=summary),out/'final_report.json')
    print('[SUCCESS] All pretrained weights frozen; summary:',json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--mixed-per-mode',type=int,default=167)
    ap.add_argument('--sadcher-count',type=int,default=300)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--native-timeout',type=int,default=25)
    ap.add_argument('--bootstrap-draws',type=int,default=10000)
    args=ap.parse_args()
    if not (0<=args.mixed_per_mode<=167 and 0<=args.sadcher_count<=300):raise ValueError('Invalid case counts')
    run(args)
if __name__=='__main__':main()
