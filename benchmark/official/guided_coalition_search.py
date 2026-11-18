#!/usr/bin/env python3
"""Replayable, capability- and arrival-guided coalition planning (inference only).

Uses the unchanged original authors' TaskEnv transitions and action masks.
NO training, test-label access, or future outcome lookups in the action heuristic.
K full simulated episodes form an OFFLINE simulator-assisted planner, NOT an
online policy and NOT a plan guaranteed robust to unknown real-world events.
The chosen full plan is recorded and exactly replayed on a fresh environment.
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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from attention import AttentionNet
from benchmark_core import metrics
from train_temporal_spatial_experiments import mixed_env, official_env, MODES, Limits
from warm_relational_kan import WarmRelationalKAN, count_parameters

# Distinct from previous papers' 2026-11 evaluation/test seeds. Freeze after use.
VAL_SEED = 1543000000
TEST_SEED = 1743000000
ALPHA_GRID = (0.0, 0.35, 0.7, 1.1)


@dataclass(frozen=True)
class HeuristicSettings:
    alpha: float = 0.7
    # Fixed a priori; tune ALPHA ONLY on synthetic validation data.
    travel: float = 0.80
    coalition_wait: float = 0.60
    ready_delay: float = 0.30
    missing_skill: float = 1.40
    partner_count: float = 0.12
    depot_penalty: float = 0.3


def _float(x, default=0.0):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except (TypeError, ValueError):
        return default


def _eta(agent, task, now, *, env, is_current=False):
    """Earliest estimated arrival from CURRENT visible state, not future rollout."""
    velocity = max(_float(agent.get('velocity', 0.2), 0.2), 1e-6)
    ready = now
    if not is_current:
        ct = int(agent.get('current_task', -1))
        if ct >= 0 and ct != task['ID']:
            # A complete assigned coalition has a known finish time.
            # An incomplete coalition has unknown release, so is excluded.
            current = env.task_dic[ct] if hasattr(env,'task_dic') and ct in env.task_dic else None
            if not current or not current['feasible_assignment']:
                return float('inf')
            ready = max(now,float(current['time_finish']))
    dist = float(np.linalg.norm(np.asarray(agent['location'], dtype=float)
                                - np.asarray(task['location'], dtype=float)))
    return ready + dist / velocity


def estimate_task(env, agent_id, task_id):
    """Shortage-aware greedy coalition completion estimate, no env mutation.

    Actual missing trait counts come from task['status'] and the current agent's
    abilities. Candidate complement agents are not assumed movable when tied
    to any other task. Treats potential join as a *heuristic*, not a feasibility
    certificate; the original TaskEnv still decides legality and completion.
    """
    now = float(env.current_time)
    task = env.task_dic[task_id]
    agent = env.agent_dic[agent_id]
    deficit = np.maximum(np.asarray(task['status'], dtype=float), 0.0)
    own = np.maximum(np.asarray(agent['abilities'], dtype=float), 0.0)
    remaining = np.maximum(deficit - own, 0.0)
    own_eta = _eta(agent, task, now, env=env, is_current=True)
    member_eta = [float(env.get_arrival_time(a, task_id)) for a in task['members']]
    used = set(task['members']) | {agent_id}
    partners = 0
    complement_etas = []
    while float(remaining.sum()) > 1e-9:
        options = []
        for other_id, other in env.agent_dic.items():
            if other_id in used: continue
            arrival = _eta(other, task, now, env=env)
            if not math.isfinite(arrival): continue
            abilities = np.maximum(np.asarray(other['abilities'], dtype=float), 0.0)
            gain = float(np.minimum(remaining, abilities).sum())
            if gain <= 0: continue
            # Earliest useful partners, penalizing delays and rewarding coverage.
            options.append((arrival + 0.35 / gain, arrival, -gain, other_id))
        if not options:
            break
        _, arrival, _, chosen = min(options)
        used.add(chosen)
        partners += 1
        complement_etas.append(arrival)
        remaining = np.maximum(remaining - np.maximum(
            np.asarray(env.agent_dic[chosen]['abilities'], dtype=float), 0.), 0.)
    all_arrivals = member_eta + [own_eta] + complement_etas
    latest = max(all_arrivals)
    wait = float(sum(max(0., latest - eta) for eta in all_arrivals))
    travel = max(0., own_eta - now)
    delay = max(0., latest - now)
    missing = float(remaining.sum())
    return {'travel':travel, 'coalition_wait':wait,
            'ready_delay':delay, 'missing_skill':missing,
            'partners':partners,'estimated_ready':latest,
            'complete_estimate':missing<1e-9}


def guided_bias(env, agent_id, legal_mask, settings:HeuristicSettings):
    """Legal-action logit residual. Never edits or bypasses original action mask."""
    mask = np.asarray(legal_mask, dtype=bool).reshape(-1)
    bias = np.zeros(len(mask), dtype=np.float64)
    task_actions = [j for j in range(1, len(mask)) if not mask[j]]
    if not task_actions or settings.alpha == 0:
        return bias
    raw=[]
    for action in task_actions:
        c = estimate_task(env, agent_id, action-1)
        # Costs in native environment's time scale (positions ~[0,1], speed 0.2).
        cost=(settings.travel*c['travel'] + settings.coalition_wait*c['coalition_wait']
              +settings.ready_delay*c['ready_delay']
              +settings.missing_skill*c['missing_skill']
              +settings.partner_count*c['partners'])
        raw.append(cost)
    arr=np.asarray(raw,dtype=np.float64)
    med=float(np.median(arr))
    spread=max(1.,float(np.std(arr)))
    for action,cost in zip(task_actions,arr):
        bias[action] = float(np.clip(-settings.alpha*(cost-med)/spread,-3.,3.))
    if not mask[0]: bias[0] = -settings.alpha * settings.depot_penalty
    return bias


def _probabilities(model, env, agent_id, settings, guided, task, agents, mask):
    t=torch.as_tensor(task,dtype=torch.float32)
    a=torch.as_tensor(agents,dtype=torch.float32)
    m=torch.as_tensor(mask,dtype=torch.bool)
    ix=torch.tensor([[[agent_id]]],dtype=torch.long)
    with torch.inference_mode():
        p, _=model(t,a,m,ix)
    p=np.asarray(p.detach().cpu().numpy(),dtype=float).reshape(-1)
    m=np.asarray(mask,dtype=bool).reshape(-1)
    if len(p)!=len(m) or not np.isfinite(p).all() or (p[~m]<0).any():
        raise RuntimeError('Invalid probability vector')
    if not guided or settings.alpha==0:
        return p
    # Force illegal actions to exactly zero; alpha=0 is exactly the base policy.
    logits=np.log(np.maximum(p,1e-30)) + guided_bias(env,agent_id,m,settings)
    logits[m]=-np.inf
    if not np.isfinite(logits[~m]).any(): raise RuntimeError('No legal action')
    z=np.exp(logits-np.max(logits[~m]))
    z[m]=0
    return z/z.sum()


def run_logged_episode(env, model, seed, *, guided=False, sample=False,
                       settings=HeuristicSettings(), scripted=None):
    """Exact original event/step protocol, plus action/time/route event log.

    scripted is an immutable action list from a previous rollout. On replay,
    it is NOT allowed to recompute/select an alternative model action.
    """
    rng=random.Random(int(seed))
    trace=[]
    count=rounds=0
    status='completed'
    started=time.perf_counter()
    if scripted is not None:
        scripted=list(scripted)
    while not env.finished and float(env.current_time)<Limits.max_time:
        if rounds >= Limits.max_decision_rounds or count >= Limits.max_actions:
            status='safety_iteration_limit';break
        release,current_time=env.next_decision()
        if not np.isfinite(current_time):status='nonfinite_event_time';break
        env.current_time=float(current_time)
        ready,blocked=list(release[0]),list(release[1])
        rng.shuffle(ready)
        ready += blocked
        if not ready:
            env.finished=env.check_finished()
            rounds+=1
            if not env.finished:status='no_agents_to_schedule';break
            continue
        for agent_id in ready:
            if count>=Limits.max_actions:status='safety_iteration_limit';break
            agent=env.agent_dic[agent_id]
            task,agents,mask=env.agent_observe(agent_id,max_waiting=False)
            all_assigned=bool(np.all(env.get_matrix(env.task_dic,'feasible_assignment')))
            no_choice=bool(np.asarray(mask[0,1:],dtype=bool).all())
            if no_choice and not all_assigned:
                agent['no_choice']=True
                continue
            if no_choice and all_assigned and int(agent['current_task'])<0:
                continue
            if scripted is None:
                probs=_probabilities(model,env,agent_id,settings,guided,task,agents,mask)
                if sample:
                    # Draw with torch to mirror shared benchmark stochastic protocol.
                    action=int(torch.multinomial(torch.tensor(probs,dtype=torch.float64),1).item())
                else:
                    action=int(np.argmax(probs))
            else:
                if count>=len(scripted):
                    raise AssertionError(f'Plan ended early at decision {count}')
                old=scripted[count]
                if (int(old['agent_id'])!=int(agent_id)
                    or int(old['round'])!=int(rounds)
                    or abs(float(old['time'])-float(env.current_time))>1e-8):
                    raise AssertionError(f'Plan diverged at {count}: {old} != agent {agent_id}, round {rounds}, time {env.current_time}')
                action=int(old['action'])
            if action<0 or action>=mask.shape[1] or bool(mask[0,action]):
                raise AssertionError(f'Illegal replay/selection action={action} agent={agent_id}')
            trace.append({'decision':count, 'round':rounds,'time':float(env.current_time),
                          'agent_id':int(agent_id),'action':action,
                          'target_task':int(action-1)})
            _,workable,_=env.agent_step(agent_id,action,rounds)
            if not workable:status='invalid_transition';break
            count+=1
        if status!='completed':break
        env.finished=env.check_finished()
        rounds+=1
    if not env.finished and status=='completed':status='time_horizon'
    if scripted is not None and count!=len(scripted):
        raise AssertionError(f'Unconsumed plan actions: {count} of {len(scripted)}')
    result=metrics(env,Limits.max_time,time.perf_counter()-started,count,rounds,status)
    routes={str(i):[int(x) for x in a['route']] for i,a in env.agent_dic.items()}
    return result,trace,routes


def replay_and_verify(template, winner):
    replay=copy.deepcopy(template)
    actual,trace,routes=run_logged_episode(replay,None,winner['seed'],scripted=winner['trace'])
    for field in ('done','status','actions','rounds'):
        if actual[field]!=winner['result'][field]:
            raise AssertionError(f'Replay mismatch {field}: {actual[field]} vs {winner["result"][field]}')
    for field in ('makespan','waiting_time','travel_dist','native_reward','efficiency','success_rate'):
        if not math.isclose(float(actual[field]),float(winner['result'][field]),rel_tol=1e-8,abs_tol=1e-7):
            raise AssertionError(f'Replay mismatch {field}: {actual[field]} vs {winner["result"][field]}')
    if routes!=winner['routes'] or trace!=winner['trace']:
        raise AssertionError('Replay routes/trace differ')
    return float(actual['runtime_s'])


def _candidate(template, model, seed, guided, sample, settings):
    env=copy.deepcopy(template)
    # Controlled RNG sampling; matching numeric seeds across both models.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        result,trace,routes=run_logged_episode(env,model,seed,guided=guided,
                                              sample=sample,settings=settings)
    return {'result':result,'trace':trace,'routes':routes,'seed':int(seed),
            'guided':bool(guided),'sample':bool(sample)}


def _rank(c):
    r=c['result']
    return (0 if r['done'] else 1, float(r['makespan']) if r['done'] else 200.,
            float(r['waiting_time']))


def search_case(template, model, seed_base, k, alpha, *, replay=True):
    """Compare 1-rollout Greedy, guided Greedy, and two K-rollout searches.

    Both K-rollout searches start with their own greedy episode, then K-1
    samples. They use identical numeric seeds and equal model evaluation counts.
    """
    settings=HeuristicSettings(alpha=float(alpha))
    methods={}
    for use_guidance, label in ((False,'plain'),(True,'guided')):
        candidates=[]
        t0=time.perf_counter()
        for attempt in range(k):
            seed=int(seed_base)+attempt*7919
            x=_candidate(template,model,seed,use_guidance,attempt>0,settings)
            x['attempt']=attempt
            candidates.append(x)
        runtime=time.perf_counter()-t0
        chosen=min(candidates,key=_rank)
        method=f'{label}_best_of_{k}'
        methods[method]={'winner':chosen,'search_runtime_s':runtime,
                         'all_attempts':[{'attempt':c['attempt'],**c['result']} for c in candidates]}
        methods[f'{label}_greedy']={'winner':candidates[0],
                                   'search_runtime_s':float(candidates[0]['result']['runtime_s']),
                                   'all_attempts':[{'attempt':0,**candidates[0]['result']}]}
    if replay:
        for m,v in methods.items():
            v['replay_runtime_s']=replay_and_verify(template,v['winner'])
    return methods


def _load_models():
    st=HERE/'reference'/'stage_a_best_model.pt'
    att=HERE/'weights'/'AttentionNet_published.pt'
    if not st.exists() or not att.exists():
        raise FileNotFoundError(f'Missing Stage A or AttentionNet weights in {HERE}')
    sw=torch.load(st,map_location='cpu',weights_only=True)
    if sw.get('model_name')!='WarmRelationalKAN' or sw.get('stage')!='A':
        raise RuntimeError('Wrong Stage A checkpoint')
    stage=WarmRelationalKAN()
    stage.load_state_dict(sw['weights'],strict=True)
    aw=torch.load(att,map_location='cpu',weights_only=True)
    if aw.get('model_name')!='AttentionNet_published':
        raise RuntimeError('Wrong AttentionNet published checkpoint')
    attention=AttentionNet(11,15,128)
    attention.load_state_dict(aw['weights'],strict=True)
    stage.eval().cpu();attention.eval().cpu()
    print('[MODELS] Stage A',count_parameters(stage),'AttentionNet',count_parameters(attention),flush=True)
    return {'stage_a':stage,'attention':attention}, {'stage_a':hashlib.sha256(st.read_bytes()).hexdigest(),
                                                    'attention':hashlib.sha256(att.read_bytes()).hexdigest()}


def _seed(mode,i,base):
    return int(base + MODES.index(mode)*1000003 + i*1009)


def calibration(models,n):
    """Alpha selection only on NEW synthetic validation cases; never official/test."""
    scores=[];best={}
    for name,model in models.items():
        for alpha in ALPHA_GRID:
            completed=0;values=[]
            for mode in MODES:
                for i in range(n):
                    env=mixed_env(_seed(mode,i,VAL_SEED),mode)
                    env.init_state()
                    sample=_candidate(env,model,seed=VAL_SEED+901+i+3*MODES.index(mode),
                                      guided=True,sample=False,settings=HeuristicSettings(alpha))
                    r=sample['result'];completed+=int(r['done'])
                    values.append(float(r['makespan']) if r['done'] else 200.)
            scores.append({'model':name,'alpha':float(alpha),'cases':len(values),
                           'completed':completed,'penalized_makespan':float(np.mean(values))})
        options=[r for r in scores if r['model']==name]
        chosen=min(options,key=lambda r:(-r['completed'],r['penalized_makespan'],r['alpha']))
        best[name]=float(chosen['alpha'])
        print('[CALIBRATE]',name,'alpha=',chosen['alpha'],'val=',chosen['penalized_makespan'],flush=True)
    return best,scores


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def bootstrap(diff,n=3000,seed=12766,strata=None):
    v=np.asarray(diff,dtype=float)
    rng=np.random.default_rng(seed)
    bins=[np.arange(len(v))] if strata is None else [
        np.flatnonzero(np.asarray(strata)==group)
        for group in sorted(set(strata))]
    means=[]
    for i in range(0,n,100):
        take=min(100,n-i)
        draw=[]
        for ids in bins:
            choose=ids[rng.integers(len(ids),size=(take,len(ids)))]
            draw.append(v[choose].sum(1))
        means.extend(sum(draw)/len(v))
    return [float(x) for x in np.quantile(means,[.025,.975])]


def analysis(rows):
    # Additional, balanced synthetic TEST aggregate. No extra evaluated cases.
    synthetic=[dict(r,dataset='new_heldout_POOLED',
                    case=str(r['dataset'])+'::'+str(r['case']))
               for r in rows if r['dataset'].startswith('new_heldout_')]
    combined=rows+synthetic
    summary=[];pairs=[]
    for ds in sorted({r['dataset'] for r in combined}):
        sub=[r for r in combined if r['dataset']==ds]
        for name,method in sorted({(r['model'],r['method']) for r in sub}):
            parts=[r for r in sub if (r['model'],r['method'])==(name,method)]
            summary.append({'dataset':ds,'model':name,'method':method,'cases':len(parts),
                'completed':sum(int(x['done']) for x in parts),
                'penalized_makespan':float(np.mean([x['penalized_makespan'] for x in parts])),
                'mean_waiting':float(np.mean([x['waiting_time'] for x in parts])),
                'mean_search_cpu_s':float(np.mean([x['search_runtime_s'] for x in parts])),
                'parameters':parts[0]['parameters']})
        found={(r['case'],r['model'],r['method']):r for r in sub}
        case_ids=sorted({r['case'] for r in sub})
        comparisons=[(('stage_a','guided_greedy'),('stage_a','plain_greedy')),
                     (('stage_a','guided_best_of_K'),('stage_a','plain_best_of_K')),
                     (('stage_a','guided_best_of_K'),('attention','plain_best_of_K')),
                     (('stage_a','guided_best_of_K'),('attention','guided_best_of_K')),
                     (('stage_a','plain_best_of_K'),('attention','plain_best_of_K')),
                     (('attention','guided_best_of_K'),('attention','plain_best_of_K'))]
        k_match=next((r['method'] for r in sub if r['method'].startswith('plain_best_of_')),None)
        if k_match is None:continue
        k=k_match.split('_')[-1]
        for (ma,sa),(mb,sb) in comparisons:
            sa=sa.replace('K',k);sb=sb.replace('K',k)
            values=[]
            for cid in case_ids:
                a=found[cid,ma,sa];b=found[cid,mb,sb]
                values.append(a['penalized_makespan']-b['penalized_makespan'])
            strata=[str(cid).split('::')[0] for cid in case_ids] if ds=='new_heldout_POOLED' else None
            ci=bootstrap(values,strata=strata) if len(values)>1 else [None,None]
            pairs.append({'dataset':ds,'model_a':ma,'method_a':sa,'model_b':mb,'method_b':sb,
                          'cases':len(values),'delta_a_minus_b':float(np.mean(values)),
                          'ci95_lower':ci[0],'ci95_upper':ci[1]})
    return summary,pairs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='/kaggle/working/GUIDED_COALITION_RESULTS')
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--val-each',type=int,default=12)
    p.add_argument('--test-each',type=int,default=50)
    p.add_argument('--official',type=int,default=50)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.k<2 or a.k>16:raise ValueError('K must be 2..16')
    if a.val_each<1 or a.test_each<1 or not (0<=a.official<=50):raise ValueError('Invalid dataset size')
    torch.set_num_threads(1)
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    models,hashes=_load_models()
    print('[CALIBRATION] disjoint synthetic validation; official NEVER used in alpha choice',flush=True)
    alpha,calib_rows=calibration(models,1 if a.smoke else a.val_each)
    write_csv(dest/'calibration.csv',calib_rows)
    datasets=[]
    n=1 if a.smoke else a.test_each
    for mode in MODES:
        datasets.append((f'new_heldout_{mode}',[(i,_seed(mode,i,TEST_SEED),mixed_env(_seed(mode,i,TEST_SEED),mode)) for i in range(n)]))
    if not a.smoke and a.official:
        datasets.append(('reused_official_DIAGNOSTIC',[(i,f'env_{i}',official_env(i)) for i in range(a.official)]))
    rows=[];action_rows=[];route_rows=[]
    journal=dest/'per_case_progress.csv'
    for ds,cases in datasets:
        print(f'[TEST] {ds} {len(cases)} cases; equal K={a.k}',flush=True)
        for cid,seed,env in cases:
            env.init_state()
            case_seed=342000000+cid*1009+sum(ord(ch) for ch in ds)*701
            for model_name,model in models.items():
                methods=search_case(env,model,case_seed,a.k,alpha[model_name],replay=True)
                for method,result in methods.items():
                    win=result['winner'];r=win['result']
                    key=f'{ds}_{cid}_{model_name}_{method}'
                    rows.append({'dataset':ds,'case':cid,'seed':seed,
                        'model':model_name,'method':method,'alpha':alpha[model_name],
                        'parameters':count_parameters(model),
                        'selected_rollout':win['attempt'],
                        'selected_seed':win['seed'], 'done':int(r['done']),
                        'makespan':float(r['makespan']),
                        'penalized_makespan':float(r['makespan']) if r['done'] else 200.,
                        'waiting_time':float(r['waiting_time']),
                        'native_reward':float(r['native_reward']),
                        'actions':int(r['actions']), 'status':r['status'],
                        'search_runtime_s':float(result['search_runtime_s']),
                        'replay_runtime_s':float(result['replay_runtime_s']),
                        'replay_verified':1, 'plan_id':key})
                    for evt in win['trace']:
                        action_rows.append({'plan_id':key,**evt})
                    for robot,route in win['routes'].items():
                        route_rows.append({'plan_id':key,'agent_id':robot,
                                           'route_json':json.dumps(route)})
            print(f'[TEST] {ds} {cid+1}/{len(cases)} full replay checks PASS',flush=True)
            # Atomic replacement per case; interruption leaves recoverable partial tables.
            for filename,data in [('evaluation_per_case.csv',rows),
                                  ('selected_plan_actions.csv',action_rows),
                                  ('selected_plan_routes.csv',route_rows)]:
                tmp=dest/(filename+'.tmp')
                write_csv(tmp,data)
                tmp.replace(dest/filename)
    summary,pairs=analysis(rows)
    write_csv(dest/'evaluation_summary.csv',summary)
    write_csv(dest/'paired_statistics.csv',pairs)
    prov={'alpha_selected_on_validation_only':alpha, 'alpha_grid':ALPHA_GRID,
          'weights_sha256':hashes,'test_seed':TEST_SEED,'validation_seed':VAL_SEED,
          'k':a.k,'test_each':n,'official':0 if a.smoke else a.official,
          'smoke':a.smoke,'search_protocol':'offline full TaskEnv simulation; greedy + K-1 stochastic',
          'plan_replay':'exact events, decision order, routes, makespan and reward checked on independent copied initial env',
          'published_attention_protocol':'original weights, common TaskEnv; not necessarily original paper full search pipeline'}
    (dest/'provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
    print('[DONE] replayable guided-coalition evaluation complete',flush=True)
    for s in summary:
        print('[SUMMARY]',s['dataset'],s['model'],s['method'],
              'complete',s['completed'],'/',s['cases'],
              'makespan',round(s['penalized_makespan'],4),'time',round(s['mean_search_cpu_s'],3),flush=True)


if __name__=='__main__':main()
