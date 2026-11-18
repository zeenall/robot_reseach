#!/usr/bin/env python3
"""Reconstruct one stored plan using the ORIGINAL TaskEnv and verify it exactly."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from guided_coalition_search import run_logged_episode, mixed_env, official_env


def run_verify(directory:Path,plan_id:str):
    with (directory/'evaluation_per_case.csv').open(newline='') as f:
        found=[r for r in csv.DictReader(f) if r['plan_id']==plan_id]
    if len(found)!=1:raise ValueError(f'Expected exactly one matching plan_id, got {len(found)}')
    meta=found[0]
    with (directory/'selected_plan_actions.csv').open(newline='') as f:
        actions=[r for r in csv.DictReader(f) if r['plan_id']==plan_id]
    actions.sort(key=lambda x:int(x['decision']))
    with (directory/'selected_plan_routes.csv').open(newline='') as f:
        routes={r['agent_id']:json.loads(r['route_json']) for r in csv.DictReader(f) if r['plan_id']==plan_id}
    if meta['dataset']=='reused_official_DIAGNOSTIC':
        env=official_env(int(meta['case']))
    else:
        prefix='new_heldout_'
        if not meta['dataset'].startswith(prefix):raise ValueError(f'Unexpected dataset {meta["dataset"]}')
        env=mixed_env(int(meta['seed']),meta['dataset'][len(prefix):])
    env.init_state()
    res,new_actions,new_routes=run_logged_episode(env,None,int(meta['selected_seed']),scripted=actions)
    if len(new_actions)!=len(actions):raise AssertionError('Trace length differs')
    if new_routes!=routes:raise AssertionError('Saved robot routes differ')
    for k in ('makespan','waiting_time'):
        if not math.isclose(float(res[k]),float(meta[k]),rel_tol=1e-8,abs_tol=1e-7):
            raise AssertionError(f'{k} mismatch: {res[k]} vs {meta[k]}')
    if int(meta['done'])!=int(res['done']):raise AssertionError('Success mismatch')
    print('VERIFIED plan_id:',plan_id)
    print('Decisions:',len(new_actions),'Robots:',len(new_routes),
          'Success:',bool(res['done']),'Makespan:',res['makespan'])
    return res


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--plan-id',default=None)
    args=p.parse_args()
    out=Path(args.out)
    if args.plan_id is None:
        with (out/'evaluation_per_case.csv').open(newline='') as f:
            args.plan_id=next(csv.DictReader(f))['plan_id']
    run_verify(out,args.plan_id)
