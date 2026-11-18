#!/usr/bin/env python3
"""Paper 2: scarcity-aware coalition choice and critical-bottleneck prefix repair.

Inference only: unchanged Stage A and published AttentionNet weights; unchanged
original TaskEnv action legality/transitions. FOUR DISTINCT search methods, each
budgeted to exactly K complete simulated rollouts per instance:
  guided            = Guided greedy + K-1 stochastic guided rollouts
  scarcity          = Scarcity-guided greedy + K-1 stochastic scarcity rollouts
  critical_repair   = Guided greedy + K-1 critical-path branch repairs/fallbacks
  combined          = Scarcity greedy + K-1 critical-path branch repairs/fallbacks

Every selected schedule is EXACTLY replayed on a fresh TaskEnv copy; proposed
repair is a legal alternative decision followed by suffix re-planning. It never
mutates an existing executed route or bypasses the authors' action masks.

CRITICAL METHODOLOGICAL NOTE: This is OFFLINE simulator-assisted optimization;
looking at a *completed simulated schedule* to find its bottleneck would NOT
be available to an online robot controller without such a simulator.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import guided_coalition_search as g
from guided_coalition_search import HeuristicSettings, _rank
from train_temporal_spatial_experiments import mixed_env, official_env, MODES, Limits
from warm_relational_kan import count_parameters
from benchmark_core import metrics

# New, disjoint synthetic instance families (not used in experiments 23--27).
VALID_SEED=1949000000
TEST_SEED=2049000000
# Fixed Stage A/AttentionNet alphas chosen previously on result27 validation.
FROZEN_ALPHA={'stage_a':1.1, 'attention':0.7}
# Scarcity beta is selected using ONLY NEW synthetic validation, per model.
BETA_GRID=(0.0, 0.40, 0.80, 1.20)
MAX_SCORE=200.0

@dataclass(frozen=True)
class Choice:
    beta: float = 0.0
    alpha: float = 1.1


def _norm_costs(costs):
    arr=np.asarray(costs,dtype=np.float64)
    if len(arr)==0:return arr
    return (arr-np.median(arr))/max(1.,float(np.std(arr)))


def _availability_weight(env,robot):
    """Estimate ability supply without knowing future random outcomes."""
    ct=int(robot.get('current_task',-1))
    if ct<0:
        return 1.0
    task=env.task_dic.get(ct)
    if task is None or not task['feasible_assignment']:
        return 0.05  # already committed to an incomplete coalition
    delay=max(0.,float(task['time_finish'])-float(env.current_time))
    return 1.0/(1.0+delay/8.)


def scarcity_costs(env,agent_id,legal_actions):
    """A legal-action cost: deficit relief vs opportunity cost of reserving skills.

    All statistics derive from CURRENT observed TaskEnv state; no future
    solutions, no objective evaluation labels, no official/test tuning.
    """
    agent=env.agent_dic[agent_id]
    own=np.maximum(np.asarray(agent['abilities'],dtype=float),0.)
    dims=len(own)
    tasks=[t for t in env.task_dic.values() if not t['feasible_assignment']]
    demand=np.zeros(dims,dtype=float)
    for task in tasks:
        demand+=np.maximum(np.asarray(task['status'],dtype=float),0.)
    supply=np.zeros(dims,dtype=float)
    for rob in env.agent_dic.values():
        supply+=np.maximum(np.asarray(rob['abilities'],dtype=float),0.)*_availability_weight(env,rob)
    # Normalize to avoid uncontrolled magnitudes on specialist-heavy instances.
    priority=np.clip(demand/np.maximum(1.,supply),0.,15.)
    if priority.max()>1e-12:
        priority=priority/priority.max()
    scores=[]
    now=float(env.current_time)
    for action in legal_actions:
        if action==0:
            scores.append(0.0)
            continue
        t=env.task_dic[int(action)-1]
        deficit=np.maximum(np.asarray(t['status'],dtype=float),0.)
        useful=np.minimum(own,deficit)
        if useful.sum()<1e-9:
            scores.append(10.)
            continue
        est=g.estimate_task(env,agent_id,int(action)-1)
        est_ready=max(0.,float(est['estimated_ready'])-now)
        duration=float(t['time'])
        # Prioritize applying scarce skills to tasks needing them, especially
        # long tasks which risk dominating the final completion time.
        relief=float(np.dot(useful,1. + priority)) / max(1.,float(deficit.sum()))
        criticality=1.+min(duration,30.)/40.
        # Reserving the same scarce robot for a distant/long coalition reduces
        # supply for other outstanding tasks; penalize this opportunity cost.
        other_need=np.maximum(demand-deficit,0.)
        pressure=float(np.dot(useful,priority*other_need/np.maximum(1.,demand)))
        commitment=(est_ready+duration)/max(1.,float(env.duration_scale))
        waiting=float(est['coalition_wait'])/max(1.,float(env.duration_scale))
        cost= -2.5*relief*criticality + 0.42*commitment*pressure + 0.06*waiting
        scores.append(float(cost))
    return np.asarray(scores,dtype=float)


def policy_probabilities(model,env,agent_id,choice,scarcity,task,agents,mask):
    settings=HeuristicSettings(alpha=float(choice.alpha))
    base=g._probabilities(model,env,agent_id,settings,True,task,agents,mask)
    if not scarcity or choice.beta==0.:
        return base
    m=np.asarray(mask,dtype=bool).reshape(-1)
    legal=[int(i) for i in np.flatnonzero(~m) if i>0]
    if len(legal)<2:
        return base
    costs=scarcity_costs(env,agent_id,legal)
    adjustment=np.zeros(len(base),dtype=float)
    for act,value in zip(legal,_norm_costs(costs)):
        adjustment[act]=float(np.clip(-choice.beta*value,-2.5,2.5))
    logits=np.log(np.maximum(base,1e-30))+adjustment
    logits[m]=-np.inf
    good=~m
    x=np.exp(logits-np.max(logits[good]))
    x[m]=0.
    return x/x.sum()


def _trace_item(n,rounds,now,agent_id,action):
    # Exact schema expected by guided_coalition_search.replay_and_verify.
    return {'decision':int(n),'round':int(rounds),'time':float(now),
            'agent_id':int(agent_id),'action':int(action),
            'target_task':int(action-1)}


def episode(env,model,seed,choice,*,scarcity=False,sample=False,
            prefix=None,branch_step=None,branch_action=None):
    """Exact original event protocol; optional identical prefix + ONE legal edit.

    A branch repeats base plan actions up to branch_step, forces a legal
    alternative at branch_step, then re-plans the suffix with the model.
    """
    rng=random.Random(int(seed))
    trace=[];legal_log=[]
    count=rounds=0;status='completed'
    branch_applied=False
    t0=time.perf_counter()
    while not env.finished and float(env.current_time)<Limits.max_time:
        if rounds>=Limits.max_decision_rounds or count>=Limits.max_actions:
            status='safety_iteration_limit';break
        release,now=env.next_decision()
        if not np.isfinite(now): status='nonfinite_event_time';break
        env.current_time=float(now)
        ready,blocked=list(release[0]),list(release[1])
        rng.shuffle(ready)
        ready+=blocked
        if not ready:
            env.finished=env.check_finished();rounds+=1
            if not env.finished:status='no_agents_to_schedule';break
            continue
        for aid in ready:
            if count>=Limits.max_actions:status='safety_iteration_limit';break
            agent=env.agent_dic[aid]
            task,agents,mask=env.agent_observe(aid,max_waiting=False)
            all_assigned=bool(np.all(env.get_matrix(env.task_dic,'feasible_assignment')))
            no_choice=bool(np.asarray(mask[0,1:],dtype=bool).all())
            if no_choice and not all_assigned:
                agent['no_choice']=True
                continue
            if no_choice and all_assigned and int(agent['current_task'])<0:
                continue
            legal=[int(i) for i in np.flatnonzero(~np.asarray(mask,dtype=bool).reshape(-1))]
            if prefix is not None and branch_step is not None and count<branch_step:
                if count>=len(prefix):raise AssertionError('Broken repair prefix length')
                old=prefix[count]
                if (old['agent_id']!=int(aid) or old['round']!=int(rounds)
                    or abs(old['time']-float(now))>1e-8):
                    raise AssertionError('Repair prefix diverged BEFORE edited decision')
                action=int(old['action'])
            elif prefix is not None and branch_step is not None and count==branch_step:
                if count>=len(prefix):raise AssertionError('Bad branch index')
                old=prefix[count]
                if old['agent_id']!=int(aid) or old['round']!=int(rounds) or abs(old['time']-float(now))>1e-8:
                    raise AssertionError('Edited branch event not reproducible')
                action=int(branch_action)
                if action==int(old['action']):raise AssertionError('No-op edit')
                branch_applied=True
            else:
                probabilities=policy_probabilities(model,env,aid,choice,scarcity,task,agents,mask)
                if sample:
                    action=int(torch.multinomial(torch.as_tensor(probabilities,dtype=torch.float64),1).item())
                else:
                    action=int(np.argmax(probabilities))
            if action not in legal:
                raise AssertionError(f'Illegal action {action} at decision {count}; legal={legal}')
            trace.append(_trace_item(count,rounds,now,aid,action))
            legal_log.append(legal)
            _,workable,_=env.agent_step(aid,action,rounds)
            if not workable:status='invalid_transition';break
            count+=1
        if status!='completed':break
        env.finished=env.check_finished()
        rounds+=1
    if not env.finished and status=='completed':status='time_horizon'
    result=metrics(env,Limits.max_time,time.perf_counter()-t0,count,rounds,status)
    routes={str(i):[int(x) for x in a['route']] for i,a in env.agent_dic.items()}
    times={int(i):float(t['time_finish']) for i,t in env.task_dic.items() if t['feasible_assignment']}
    members={int(i):[int(x) for x in t['members']] for i,t in env.task_dic.items()}
    arrivals={int(i):{int(x):float(env.get_arrival_time(x,int(i))) for x in m}
              for i,m in members.items() if m}
    # In original TaskEnv, Cmax includes robots' FINAL RETURN TO DEPOTS.
    # Last task completion alone is not necessarily the bottleneck.
    terminal_robot=max(env.agent_dic, key=lambda rid:float(env.agent_dic[rid]['arrival_time'][-1]))
    terminal_route=[int(x) for x in env.agent_dic[terminal_robot]['route'] if int(x)>=0]
    terminal_task=terminal_route[-1] if terminal_route else -1
    return {'result':result,'trace':trace,'routes':routes,'legal':legal_log,
            'task_finishes':times,'task_members':members,'arrivals':arrivals,
            'terminal_robot':int(terminal_robot),'terminal_task':int(terminal_task),
            'terminal_arrival':float(env.agent_dic[terminal_robot]['arrival_time'][-1]),
            'branch_applied':branch_applied,'seed':int(seed)}


def candidate(template,model,seed,choice,scarcity=False,sample=False,
              prefix=None,branch_step=None,branch_action=None,shuffle_seed=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        data=episode(copy.deepcopy(template),model,
                     int(seed if shuffle_seed is None else shuffle_seed),choice,
                     scarcity=scarcity,sample=sample,prefix=prefix,
                     branch_step=branch_step,branch_action=branch_action)
    data['sampling_seed']=int(seed)
    return data


def branch_options(base,max_branches):
    """Legal-prefix edits selected from task-finish AND final-depot-return chains.

    Makespan in the original TaskEnv includes last robot's return to its depot.
    We therefore prioritize the last task on that robot's route, followed by
    the task that finishes latest. Every proposed alternative was LEGAL when
    its original decision was made. No simulator lookahead is used here beyond
    the already-budgeted initial rollout.
    """
    finishes=base['task_finishes']
    if not finishes:return []
    finish_task=max(finishes,key=lambda tid:finishes[tid])
    terminal_task=base['terminal_task']
    targets=[]
    if terminal_task in finishes:targets.append((terminal_task,'return_chain'))
    if finish_task not in {t[0] for t in targets}:targets.append((finish_task,'last_finish'))
    possibilities=[]
    for target_rank,(target,reason) in enumerate(targets):
        arrival=base['arrivals'].get(target,{})
        latest=sorted(arrival,key=lambda aid:(-arrival[aid],aid))
        terminal_robot=base['terminal_robot']
        for idx,(ev,legal) in enumerate(zip(base['trace'],base['legal'])):
            if target+1 not in legal or ev['action']==target+1:continue
            aid=ev['agent_id']
            if aid==terminal_robot and reason=='return_chain':
                priority=0
            elif aid in latest:
                priority=1+latest.index(aid)
            else:
                priority=20
            rank=(priority,target_rank,ev['time'],idx)
            possibilities.append((rank,idx,target+1,reason,target))
    possibilities.sort(key=lambda r:r[0])
    result=[];used_agents=set()
    for _,idx,action,reason,target in possibilities:
        aid=base['trace'][idx]['agent_id']
        if aid in used_agents:continue
        result.append((idx,action,reason,target))
        used_agents.add(aid)
        if len(result)>=max_branches:break
    if len(result)<max_branches:
        for _,idx,action,reason,target in possibilities:
            if (idx,action) not in {(i,a) for i,a,_,_ in result}:
                result.append((idx,action,reason,target))
                if len(result)>=max_branches:break
    return result


def search_case(template,model,seed_base,k,choice):
    """Four methods; each K full rollouts, no uncharged full simulations."""
    records={}
    for scarcity,mode_name in ((False,'guided'),(True,'scarcity')):
        if scarcity and choice.beta==0.:
            # Still record separate identical baseline for beta=0.
            pass
        t0=time.perf_counter()
        initial=candidate(template,model,seed_base,choice,scarcity=scarcity,sample=False)
        initial.update({'attempt':0,'kind':'greedy','branch_step':-1,
                        'critical_task':-1,'branch_action':-1})
        baseline_candidates=[initial]
        for j in range(1,k):
            item=candidate(template,model,seed_base+j*7919,choice,
                           scarcity=scarcity,sample=True)
            item.update({'attempt':j,'kind':'sample','branch_step':-1,
                         'critical_task':-1,'branch_action':-1})
            baseline_candidates.append(item)
        elapsed=time.perf_counter()-t0
        records[mode_name]={'candidates':baseline_candidates,
                            'winner':min(baseline_candidates,key=lambda x:_rank(x)),
                            'search_runtime_s':elapsed,'rollouts':k}
        # Critical-path neighborhood: reuse initial greedy as 1st of K.
        t0=time.perf_counter()
        repairs=[initial]
        options=branch_options(initial,k-1)
        used_fallback=0
        for j in range(1,k):
            if j<=len(options):
                at,alt,kind,critical=options[j-1]
                # Reconstruct the identical prefix with the SAME event shuffle
                # RNG as initial; sample suffix with a new torch seed.
                item=candidate(template,model,seed_base+j*7919,choice,
                               scarcity=scarcity,sample=(j%2==1),
                               prefix=initial['trace'],branch_step=at,
                               branch_action=alt,shuffle_seed=seed_base)
                item.update({'kind':kind+'_early','branch_step':at,
                             'critical_task':critical,'branch_action':alt})
                if not item['branch_applied']:
                    raise AssertionError('Critical repair branch never applied')
            else:
                used_fallback+=1
                item=candidate(template,model,seed_base+j*7919,choice,
                               scarcity=scarcity,sample=True)
                item.update({'kind':'fallback_sample','branch_step':-1,
                             'critical_task':-1,'branch_action':-1})
            item['attempt']=j
            repairs.append(item)
        # If no repair option is available, this is a documented K-rollout
        # fallback, not an unreported extra search attempt.
        extra=time.perf_counter()-t0
        repair_name='combined' if scarcity else 'critical_repair'
        records[repair_name]={'candidates':repairs,
                              'winner':min(repairs,key=lambda x:_rank(x)),
                              'search_runtime_s':float(initial['result']['runtime_s'])+extra,
                              'rollouts':k,'fallbacks':used_fallback,
                              'branches':len(options)}
    for record in records.values():
        winner=record['winner']
        # Same standalone replay checker as paper2 Guided Coalition work.
        record['replay_runtime_s']=g.replay_and_verify(template,winner)
        if len(record['candidates'])!=k:
            raise AssertionError('Incorrect search budget')
    return records


def seed_for(mode,i,base):
    return int(base+MODES.index(mode)*1000003+i*1009)


def calibrate(models,n):
    rows=[];beta_choice={}
    for name,model in models.items():
        alpha=FROZEN_ALPHA[name]
        best=None
        for beta in BETA_GRID:
            scores=[];completed=0
            for mode in MODES:
                for i in range(n):
                    seed=seed_for(mode,i,VALID_SEED)
                    env=mixed_env(seed,mode)
                    env.init_state()
                    item=candidate(env,model,VALID_SEED+113*(i+1)+MODES.index(mode),
                                   Choice(beta=float(beta),alpha=alpha),scarcity=True)
                    r=item['result'];completed+=int(r['done'])
                    scores.append(float(r['makespan']) if r['done'] else MAX_SCORE)
            r={'model':name,'beta':beta,'alpha':alpha,'cases':len(scores),
               'completed':completed,'penalized_makespan':float(np.mean(scores))}
            rows.append(r)
            key=(-completed,r['penalized_makespan'],beta)
            if best is None or key<best[0]:best=(key,beta)
        beta_choice[name]=float(best[1])
        print('[CALIBRATION]',name,'frozen_alpha',alpha,
              'selected_beta',best[1],flush=True)
    return beta_choice,rows


def atomic_csv(path,rows,columns=None):
    if not rows:
        return
    path=Path(path)
    columns=columns or list(rows[0].keys())
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def summary_and_pairs(rows,k):
    test_rows=[dict(r,dataset='new_heldout_POOLED',
                    case=f"{r['dataset']}::{r['case']}") for r in rows
               if r['dataset'].startswith('new_heldout_')]
    combined=rows+test_rows
    summary=[];paired=[]
    for dataset in sorted({r['dataset'] for r in combined}):
        data=[r for r in combined if r['dataset']==dataset]
        keys=sorted(set((r['model'],r['method']) for r in data))
        for model,method in keys:
            group=[r for r in data if (r['model'],r['method'])==(model,method)]
            summary.append({'dataset':dataset,'model':model,'method':method,
                            'cases':len(group),'completed':sum(int(r['done']) for r in group),
                            'penalized_makespan':float(np.mean([r['penalized_makespan'] for r in group])),
                            'waiting_mean':float(np.mean([r['waiting_time'] for r in group])),
                            'mean_search_cpu_s':float(np.mean([r['search_runtime_s'] for r in group])),
                            'parameters':group[0]['parameters']})
        look={(r['case'],r['model'],r['method']):r for r in data}
        ids=sorted({r['case'] for r in data})
        comps=[(('stage_a','scarcity'),('stage_a','guided')),
               (('stage_a','critical_repair'),('stage_a','guided')),
               (('stage_a','combined'),('stage_a','guided')),
               (('stage_a','combined'),('stage_a','scarcity')),
               (('stage_a','combined'),('stage_a','critical_repair')),
               (('stage_a','combined'),('attention','combined')),
               (('stage_a','combined'),('attention','guided')),
               (('stage_a','guided'),('attention','guided')),
               (('attention','combined'),('attention','guided'))]
        for (ma,xa),(mb,xb) in comps:
            diffs=[look[cid,ma,xa]['penalized_makespan']-look[cid,mb,xb]['penalized_makespan']
                   for cid in ids]
            strata=[str(i).split('::')[0] for i in ids] if dataset=='new_heldout_POOLED' else None
            ci=g.bootstrap(diffs,n=3000,seed=20261110,strata=strata) if len(diffs)>1 else (None,None)
            paired.append({'dataset':dataset,'model_a':ma,'method_a':xa,
                           'model_b':mb,'method_b':xb,'cases':len(diffs),
                           'delta_a_minus_b':float(np.mean(diffs)),
                           'ci95_lower':ci[0],'ci95_upper':ci[1]})
    return summary,paired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='/kaggle/working/SCARCITY_BOTTLENECK_RESULTS')
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--val-each',type=int,default=12)
    p.add_argument('--test-each',type=int,default=50)
    p.add_argument('--official',type=int,default=0,
                   help='Reused official cases, DIAGNOSTIC ONLY; default OFF')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not (2<=args.k<=16):raise ValueError('K must be 2..16')
    if args.val_each<1 or args.test_each<1 or not 0<=args.official<=50:raise ValueError('Invalid cases')
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    models,hashes=g._load_models()
    beta,cal_rows=calibrate(models,1 if args.smoke else args.val_each)
    atomic_csv(out/'calibration.csv',cal_rows)
    datasets=[]
    test_each=1 if args.smoke else args.test_each
    for mode in MODES:
        datasets.append((f'new_heldout_{mode}',[(i,seed_for(mode,i,TEST_SEED),mixed_env(seed_for(mode,i,TEST_SEED),mode))
                                                 for i in range(test_each)]))
    if not args.smoke and args.official:
        datasets.append(('reused_official_DIAGNOSTIC',[(i,f'env_{i}',official_env(i)) for i in range(args.official)]))
    rows=[];cand_rows=[];action_rows=[];route_rows=[]
    for dataset,cases in datasets:
        print('[TEST]',dataset,'cases',len(cases),'K',args.k,flush=True)
        for cid,seed,env in cases:
            env.init_state()
            base_seed=410000000+cid*1009+sum(map(ord,dataset))*701
            for name,model in models.items():
                choice=Choice(alpha=FROZEN_ALPHA[name],beta=beta[name])
                methods=search_case(env,model,base_seed,args.k,choice)
                for method,report in methods.items():
                    win=report['winner'];r=win['result']
                    plan_id=f'{dataset}_{cid}_{name}_{method}'
                    rows.append({'dataset':dataset,'case':cid,'seed':seed,'model':name,
                                 'method':method,'parameters':count_parameters(model),
                                 'alpha':choice.alpha,'beta':choice.beta,
                                 'K':args.k,'rollouts_actual':report['rollouts'],
                                 'winner_attempt':win['attempt'],
                                 'selected_seed':win['seed'],
                                 'winner_kind':win['kind'],
                                 'done':int(r['done']),'makespan':float(r['makespan']),
                                 'penalized_makespan':float(r['makespan']) if r['done'] else MAX_SCORE,
                                 'waiting_time':float(r['waiting_time']),
                                 'native_reward':float(r['native_reward']),
                                 'actions':int(r['actions']),'status':r['status'],
                                 'search_runtime_s':float(report['search_runtime_s']),
                                 'replay_runtime_s':float(report['replay_runtime_s']),
                                 'replay_verified':1,'plan_id':plan_id})
                    for c in report['candidates']:
                        cr=c['result']
                        cand_rows.append({'plan_id':plan_id,'attempt':c['attempt'],
                                          'kind':c['kind'],'branch_step':c['branch_step'],
                                          'branch_action':c['branch_action'],
                                          'critical_task':c['critical_task'],
                                          'terminal_robot':c['terminal_robot'],
                                          'terminal_task':c['terminal_task'],
                                          'terminal_arrival':c['terminal_arrival'],
                                          'max_task_finish':max(c['task_finishes'].values(),default=0.),
                                          'done':int(cr['done']),
                                          'makespan':float(cr['makespan']),
                                          'status':cr['status'],
                                          'selected':int(c is win)})
                    for e in win['trace']:
                        action_rows.append({'plan_id':plan_id,**e})
                    for robot,route in win['routes'].items():
                        route_rows.append({'plan_id':plan_id,'agent_id':robot,
                                           'route_json':json.dumps(route)})
            print('[REPLAY PASS]',dataset,cid+1,'/',len(cases),flush=True)
            # Atomic progress per case for partial recovery if Kaggle interrupts.
            for filename,values in (('evaluation_per_case.csv',rows),
                                    ('candidate_rollouts.csv',cand_rows),
                                    ('selected_plan_actions.csv',action_rows),
                                    ('selected_plan_routes.csv',route_rows)):
                atomic_csv(out/filename,values)
    summ,pairs=summary_and_pairs(rows,args.k)
    atomic_csv(out/'evaluation_summary.csv',summ)
    atomic_csv(out/'paired_statistics.csv',pairs)
    provenance={'model_sha256':hashes,'old_guided_alpha_fixed':FROZEN_ALPHA,
                'scarcity_beta_grid':BETA_GRID,'selected_beta_validation_only':beta,
                'validation_seed':VALID_SEED,'test_seed':TEST_SEED,
                'K_full_simulated_rollouts_per_method':args.k,
                'baseline_method':'guided=greedy+K-1 stochastic',
                'repair_method':'greedy+K-1 legal prefix branch repairs, fallback stochastic',
                'search_protocol':'OFFLINE simulator-assisted; completed simulated bottlenecks guide proposed repairs',
                'equal_K_not_equal_model_calls_or_CPU_time':True,
                'plan_replay':'original TaskEnv exact event/time/metric/route check',
                'published_attention_comparison':'original checkpoint in common protocol; not published full pipeline',
                'official_cases':'reused diagnostic, never calibration or confirmatory heldout',
                'no_retraining':True,'smoke':bool(args.smoke)}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    for row in summ:
        if row['dataset'] in ('new_heldout_POOLED','reused_official_DIAGNOSTIC'):
            print('[SUMMARY]',row['dataset'],row['model'],row['method'],
                  'complete',row['completed'],'/',row['cases'],
                  'makespan',round(row['penalized_makespan'],4),flush=True)
    print('[DONE] scarcity and bottleneck repair evaluation; exact replay passed',flush=True)

if __name__=='__main__':main()
