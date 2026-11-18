"""Reproducible HeteroMRTA native-environment benchmark.

Source environment and original AttentionNet are loaded directly from
https://github.com/marmotlab/HeteroMRTA (Apache-2.0).
No home-built substitute environment is used.

Modes: smoke/pilot/full. Full train uses group-centered REINFORCE, the same
algorithm family as the paper. The shared simplified trainer is NOT the
original distributed Ray trainer; the published checkpoint IS unmodified.
"""
import copy
import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F

from kan_policy import KANPolicy

REPO = 'https://github.com/marmotlab/HeteroMRTA.git'


@dataclass
class Config:
    mode: str = 'full'          # smoke | pilot | full
    train_seeds: int = 1
    start_seed_index: int = 0
    base_seed: int = 20261104
    output: str = '/kaggle/working/heteromrta_kan_benchmark'
    repo_path: str = '/kaggle/working/HeteroMRTA'
    local_repo_zip: str = ''
    internet_clone: bool = True
    retrain_attention: bool = True    # Train ORIGINAL author's AttentionNet BEFORE KAN
    run_kan_training: bool = True  # False: evaluate the published reference without retraining KAN
    require_published_attention: bool = False  # True: fail rather than silently replace published weights
    kan_checkpoint_path: str = ''  # optional prior KAN best.pt file for evaluation only
    run_published_attention: bool = True
    run_native_greedy: bool = True
    run_task_greedy: bool = True
    device: str = 'auto'
    max_time: float = 200.0
    max_actions: int = 400
    max_decision_rounds: int = 500
    updates_every_problems: int = 8
    val_every_problems: int = 50
    val_cases: int = 12
    learning_rate: float = 1e-4  # legacy fallback
    attention_learning_rate: float = 1e-5  # authors' network reference LR
    kan_learning_rate: float = 1e-4        # KAN reference LR
    eval_max_cases: int = 0   # 0 = whole official dataset
    verbose: bool = True

    def problems(self):
        return {'smoke': 2, 'pilot': 60, 'full': 600}[self.mode]

    def pomos(self):
        return {'smoke': 2, 'pilot': 4, 'full': 6}[self.mode]


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_repository(cfg):
    root = Path(cfg.repo_path).resolve()
    if not (root / 'env' / 'task_env.py').exists():
        if cfg.local_repo_zip:
            import zipfile
            archive = Path(cfg.local_repo_zip)
            assert archive.exists(), f'Missing ZIP: {archive}'
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    candidate = Path(member.filename)
                    assert not candidate.is_absolute() and '..' not in candidate.parts, 'Unsafe ZIP path'
                scratch = root.parent / '_hmrta_extract'
                scratch.mkdir(parents=True, exist_ok=True)
                z.extractall(scratch)
            matching = list(scratch.glob('*/env/task_env.py'))
            if not matching:
                raise FileNotFoundError('Original GitHub repo env/task_env.py not in ZIP')
            import shutil
            shutil.copytree(matching[0].parents[1], root, dirs_exist_ok=True)
        elif cfg.internet_clone:
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git','clone','--depth','1',REPO,str(root)], check=True, timeout=300)
        else:
            raise RuntimeError('Original HeteroMRTA source is unavailable. Enable Kaggle Internet or set local_repo_zip.')
    assert (root/'attention.py').exists() and (root/'env/task_env.py').exists()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Verify source provenance and published artifacts; never overwrite source files.
    try:
        commit = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = 'unavailable_local_archive'
    from env.task_env import TaskEnv
    from attention import AttentionNet
    from parameters import EnvParams, TrainParams
    return root, commit, TaskEnv, AttentionNet, EnvParams, TrainParams


def get_native_dataset(root, TaskEnv, cfg):
    files = sorted((root/'RALTestSet').glob('env_*.pkl'), key=lambda p: int(p.stem.split('_')[-1]))
    if files:
        if cfg.eval_max_cases > 0: files = files[:cfg.eval_max_cases]
        return [(p.stem, p) for p in files], 'official_published_RALTestSet_pickle'
    # Explicitly labelled fallback, not misrepresented as official saved dataset.
    print('WARNING: The original RALTestSet/*.pkl is missing. Using regenerated test seeds 0..49.')
    n = cfg.eval_max_cases if cfg.eval_max_cases > 0 else 50
    return [(f'regenerated_env_{i}', i) for i in range(n)], 'regenerated_from_original_TaskEnv_seeds'


class OfficialTestSetUnpickler(pickle.Unpickler):
    """Compatibility loader for the *original authors' trusted* RALTestSet.

    Some published files were pickled while TaskEnv was bound in ``__main__``.
    Jupyter/papermill does not expose that class under the same module name.
    Only that one historical name is remapped to the **unchanged** original
    ``env.task_env.TaskEnv`` implementation imported from the authors' repo.

    WARNING: pickle is not safe for untrusted input. Use exclusively the
    pinned/verified files from marmotlab/HeteroMRTA, never user-supplied PKLs.
    """
    def __init__(self, source, task_env_class):
        super().__init__(source)
        self.task_env_class = task_env_class

    def find_class(self, module, name):
        if module == '__main__' and name == 'TaskEnv':
            return self.task_env_class
        if module == '__main__':
            raise pickle.UnpicklingError(
                f'Unexpected historical __main__ class: {name!r}. '
                'Inspect the ORIGINAL test file and class definitions before mapping it.')
        return super().find_class(module, name)


def load_case(item, TaskEnv):
    name, target = item
    if isinstance(target, Path):
        # Trust only the pinned original repository; this loader is NOT safe
        # for arbitrary third-party pickle files.
        with target.open('rb') as f:
            env = OfficialTestSetUnpickler(f, TaskEnv).load()
        if not isinstance(env, TaskEnv):
            raise TypeError(f'{name}: expected original TaskEnv, got {type(env).__name__}')
    else:
        env = TaskEnv((3,3),(5,5),(20,20),5,30,seed=int(target),plot_figure=False)
    env.init_state()
    env.plot_figure = False
    return env


def make_training_env(TaskEnv, seed):
    # Same robot counts, task counts, trait dimensions as the native RAL test set.
    env = TaskEnv((3,3),(5,5),(20,20),5,30,seed=seed,plot_figure=False)
    env.init_state()
    return env


def original_greedy_action(env, agent_id, mask, task_first=False):
    # Mirrors TaskEnv.execute_greedy_action's geometric selection on agent_observe mask.
    # task_first=True is our additional diagnostic (not a published algorithm).
    agent = env.agent_dic[agent_id]
    ids = np.flatnonzero(~np.asarray(mask[0], dtype=bool))
    if len(ids) == 0: return 0
    if task_first:
        others = ids[ids > 0]
        if len(others): ids = others
    def cost(i):
        target = env.task_dic[int(i)-1] if int(i)>0 else env.depot_dic[agent['species']]
        return float(np.linalg.norm(agent['location'] - target['location']))
    return int(min(ids, key=cost))



def fixed_greedy_action(env, agent_id, mask, skill_aware=False):
    """New task-first baseline; never call it the authors' original Greedy.

    Action 0 is depot. Original nearest-action Greedy picks depot immediately
    when a robot starts at the depot (zero distance), causing a no-progress loop.
    Task-first restriction only applies when some task action is legal.
    When no task action is available, depot action remains permitted.

    The skill-aware version is an untuned, clearly labelled diagnostic baseline.
    It uses only CURRENT visible task deficits, robot ability and travel time.
    """
    mask=np.asarray(mask[0], dtype=np.bool_)
    ids=np.flatnonzero(~mask)
    legal_tasks=[int(a) for a in ids if a > 0]
    if not legal_tasks: return 0
    agent=env.agent_dic[agent_id]
    vel=max(float(agent.get('velocity',1.)), 1e-8)
    def key(action):
        task=env.task_dic[int(action)-1]
        travel=float(np.linalg.norm(np.asarray(agent['location']) - np.asarray(task['location'])))/vel
        if not skill_aware:
            return (travel, int(action))
        deficit=np.maximum(np.asarray(task['status'], dtype=float),0.)
        ability=np.maximum(np.asarray(agent['abilities'], dtype=float),0.)
        useful=float(np.minimum(deficit, ability).sum()) / max(float(deficit.sum()),1e-8)
        waiting_bonus=0.5 if len(task.get('members',[]))>0 else 0.
        # Lower score preferred. Coefficients are fixed a priori; no test tuning.
        score=travel - 2.*useful - waiting_bonus
        return (score,travel,int(action))
    return min(legal_tasks,key=key)

def metrics(env, max_time, runtime_s, n_actions, n_rounds, status):
    reward, completed = env.get_episode_reward(max_time)
    finished = np.asarray(completed, dtype=float)
    starts = np.asarray(env.get_matrix(env.task_dic, 'time_start'), dtype=float)
    waiting = np.asarray(env.get_matrix(env.agent_dic,'sum_waiting_time'), dtype=float)
    distances = np.asarray(env.get_matrix(env.agent_dic,'travel_dist'), dtype=float)
    return dict(native_reward=float(reward), success_rate=float(np.mean(finished)),
                makespan=float(env.current_time) if np.isfinite(env.current_time) else float(max_time),
                time_cost=float(np.nanmean(starts)) if starts.size else 0.,
                waiting_time=float(np.mean(waiting)), travel_dist=float(np.sum(distances)),
                efficiency=float(env.get_efficiency()),
                done=int(bool(env.finished)), actions=int(n_actions), rounds=int(n_rounds),
                runtime_s=float(runtime_s), status=status)


def run_episode(env, policy, cfg, device, seed, sample=False, record=False):
    """Unmodified original TaskEnv transitions, original published action mask.

    Independent safeguards on stuck/long executions; safeguard events logged.
    """
    rng = random.Random(seed)
    decisions = []
    count, rounds, status = 0, 0, 'completed'
    start = time.perf_counter()
    while not env.finished and float(env.current_time) < cfg.max_time:
        if rounds >= cfg.max_decision_rounds or count >= cfg.max_actions:
            status = 'safety_iteration_limit'
            break
        release, current_time = env.next_decision()
        if not np.isfinite(current_time):
            status = 'nonfinite_event_time'
            break
        env.current_time = float(current_time)
        ready, blocked = list(release[0]), list(release[1])
        rng.shuffle(ready)
        ready += blocked
        if not ready:
            env.finished = env.check_finished()
            rounds += 1
            if not env.finished:
                status = 'no_agents_to_schedule'
                break
            continue
        for agent_id in ready:
            if count >= cfg.max_actions:
                status = 'safety_iteration_limit'
                break
            agent = env.agent_dic[agent_id]
            task, agents, mask = env.agent_observe(agent_id, max_waiting=(policy == 'native_greedy'))
            all_assigned = bool(np.all(env.get_matrix(env.task_dic,'feasible_assignment')))
            no_task_choice = bool(np.asarray(mask[0,1:], dtype=bool).all())
            if no_task_choice and not all_assigned:
                agent['no_choice'] = True
                continue
            if no_task_choice and all_assigned and int(agent['current_task']) < 0:
                continue
            if policy == 'native_greedy':
                action = original_greedy_action(env, agent_id, mask, task_first=False)
            elif policy == 'task_first_greedy':
                action = original_greedy_action(env, agent_id, mask, task_first=True)
            elif policy == 'fixed_nearest_greedy':
                action = fixed_greedy_action(env, agent_id, mask, skill_aware=False)
            elif policy == 'skill_aware_greedy':
                action = fixed_greedy_action(env, agent_id, mask, skill_aware=True)
            else:
                t = torch.as_tensor(task, dtype=torch.float32, device=device)
                a = torch.as_tensor(agents, dtype=torch.float32, device=device)
                m = torch.as_tensor(mask, dtype=torch.bool, device=device)
                ix = torch.tensor([[[agent_id]]], dtype=torch.long, device=device)
                with torch.no_grad():
                    p, _ = policy(t, a, m, ix)
                if not torch.isfinite(p).all() or p.sum() < .999:
                    raise RuntimeError('Policy returned invalid action probability')
                if sample:
                    action = int(torch.multinomial(p[0],1).item())
                else:
                    action = int(torch.argmax(p, -1).item())
            if mask[0,action]:
                raise AssertionError(f'Invalid action {action}, agent={agent_id}')
            if record:
                decisions.append((task[0].astype(np.float32).copy(),
                                  agents[0].astype(np.float32).copy(),
                                  mask[0].astype(np.bool_).copy(), int(agent_id), int(action)))
            _, workable, _ = env.agent_step(agent_id, action, rounds)
            if not workable:
                status = 'invalid_transition'
                break
            count += 1
        if status != 'completed': break
        env.finished = env.check_finished()
        rounds += 1
    if not env.finished and status == 'completed':
        status = 'time_horizon'
    row = metrics(env, cfg.max_time, time.perf_counter()-start, count, rounds, status)
    return row, decisions


def stack_decisions(records, device):
    if not records:
        raise RuntimeError('empty trajectory batch')
    b = len(records)
    nt = max(x[0].shape[0] for x in records)
    na = max(x[1].shape[0] for x in records)
    td, ad = records[0][0].shape[1], records[0][1].shape[1]
    tasks = np.zeros((b,nt,td), np.float32)
    agents = np.zeros((b,na,ad), np.float32)
    mask = np.ones((b,nt), np.bool_)
    agent_idx = np.zeros((b,1,1), np.int64)
    actions = np.zeros(b, np.int64)
    for j,(t,a,m,idx,action) in enumerate(records):
        tasks[j,:len(t),:] = t
        agents[j,:len(a),:] = a
        mask[j,:len(m)] = m
        agent_idx[j,0,0] = idx
        actions[j] = action
    return (torch.from_numpy(tasks).to(device), torch.from_numpy(agents).to(device),
            torch.from_numpy(mask).to(device), torch.from_numpy(agent_idx).to(device),
            torch.from_numpy(actions).to(device))


def train_model(model, tag, TaskEnv, cfg, seed, output_dir, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    best_file = output_dir/'best.pt'
    resume_file = output_dir/'resume.pt'
    done_file = output_dir/'DONE.json'
    if done_file.exists() and best_file.exists():
        print(f'{tag} seed={seed}: already complete; skipping training')
        loaded = torch.load(best_file, map_location=device, weights_only=True)
        model.load_state_dict(loaded['state_dict'])
        return model
    model_lr = cfg.attention_learning_rate if tag.startswith('ATTENTION') else cfg.kan_learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=model_lr)
    history_file = output_dir/'train_history.csv'
    history = []
    first_problem = 0
    best = -float('inf')
    cfg_fingerprint = json.dumps(dict(tag=tag, seed=seed, mode=cfg.mode,
                                      lr=model_lr, problems=cfg.problems(),
                                      pomos=cfg.pomos()), sort_keys=True)
    if resume_file.exists():
        ck = torch.load(resume_file, map_location='cpu', weights_only=True)
        assert ck['fingerprint'] == cfg_fingerprint, 'Resume checkpoint incompatible with config'
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        first_problem = ck['next_problem']
        best = ck['best']
        history = ck['history']
        print(f'{tag} seed={seed} resuming at problem {first_problem}')
    else:
        set_seeds(seed)
    model.to(device)
    model.train()
    experiences = []
    train_returns = []
    last_grad_norm = float('nan')
    last_action_entropy = float('nan')
    n_problems = cfg.problems()
    pomos = cfg.pomos()
    for ip in range(first_problem, n_problems):
        env_seed = 10000000 + cfg.base_seed + 15485863*ip  # identical training instance across architectures
        template = make_training_env(TaskEnv, env_seed)
        group = []
        for j in range(pomos):
            env = copy.deepcopy(template)
            row, trajectory = run_episode(env, model, cfg, device,
                                          seed=seed+1000003*ip+j, sample=True, record=True)
            group.append((row['native_reward'], trajectory))
        rewards = np.array([x[0] for x in group], dtype=np.float64)
        baseline = rewards.mean()
        # Relative POMO baseline: same policy-gradient family as the paper.
        # Normalize for numerical stability, never use expert actions/labels.
        std = max(float(rewards.std()), 1.0)
        for r,trajectory in group:
            adv = float(np.clip((r - baseline)/std,-5,5))
            for decision in trajectory:
                experiences.append((decision,adv))
        train_returns.extend(rewards.tolist())
        need_update = ((ip+1) % cfg.updates_every_problems == 0 or ip+1 == n_problems)
        if need_update and experiences:
            random.Random(seed + ip).shuffle(experiences)
            losses=[]
            for start in range(0,len(experiences),96):
                batch=experiences[start:start+96]
                tensors=stack_decisions([x[0] for x in batch], device)
                advantages=torch.tensor([x[1] for x in batch],dtype=torch.float32,device=device)
                probs,logs=model(tensors[0],tensors[1],tensors[2],tensors[3])
                chosen=logs.gather(-1,tensors[4].unsqueeze(-1)).squeeze(-1)
                entropy=-(probs*logs).sum(-1).mean()
                loss=-(chosen*advantages).mean() - .001*entropy
                if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss in {tag}')
                optimizer.zero_grad()
                loss.backward()
                grad_measure=clip_grad_norm_(model.parameters(), 5.)
                last_grad_norm=float(grad_measure.detach().cpu())
                last_action_entropy=float(entropy.detach().cpu())
                optimizer.step()
                losses.append(float(loss.item()))
            experiences=[]
        if (ip+1) % cfg.val_every_problems==0 or ip+1 == n_problems:
            model.eval()
            values=[]
            for j in range(cfg.val_cases if cfg.mode!='smoke' else 2):
                val_env=make_training_env(TaskEnv, 40000000 + 211*j)
                val_row,_=run_episode(val_env,model,cfg,device,seed=990001+j,sample=False)
                values.append(val_row)
            m_reward=float(np.mean([row['native_reward'] for row in values]))
            hist=dict(model=tag,seed=seed,problems=ip+1,episodes=(ip+1)*pomos,
                      validation_native_reward=m_reward,
                      validation_success=float(np.mean([r['success_rate'] for r in values])),
                      training_reward_recent=float(np.mean(train_returns[-min(48,len(train_returns)):])),
                      last_grad_norm=last_grad_norm, last_action_entropy=last_action_entropy,
                      learning_rate=model_lr )
            history.append(hist)
            write_csv(history_file,history)
            if cfg.verbose:
                print(f'{tag:31} seed={seed} problem={ip+1:4}/{n_problems} '
                      f'episodes={(ip+1)*pomos:5} val_reward={m_reward:.4f} '
                      f'val_success={hist["validation_success"]:.3f}',flush=True)
            if m_reward>best or not best_file.exists():
                best=m_reward
                torch.save(dict(state_dict={k:v.cpu().clone() for k,v in model.state_dict().items()},
                                val_reward=best,problem=ip+1,tag=tag,seed=seed),best_file)
            model.train()
        if need_update:
            ck=dict(model={k:v.cpu().clone() for k,v in model.state_dict().items()},
                    optimizer=optimizer.state_dict(),next_problem=ip+1,best=best,
                    fingerprint=cfg_fingerprint,history=history)
            torch.save(ck,resume_file)
    done_file.write_text(json.dumps(dict(tag=tag,seed=seed,episodes=n_problems*pomos,
                                         best_val_reward=best),indent=2))
    model.load_state_dict(torch.load(best_file,map_location=device,weights_only=True)['state_dict'])
    model.eval()
    return model


def write_csv(path,rows):
    if not rows: return
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def load_published_checkpoint(model, root, device):
    """Safely load the original author's torch checkpoint with NumPy metadata.

    The original checkpoint contains NumPy scalars (e.g., epoch/best score)
    that are rejected by PyTorch >= 2.6 weights_only defaults. Trust only
    the identified author's commit, allowlist *specific* NumPy value classes,
    retain weights_only=True, and use strict model state-dict matching.
    No weights_only=False or generic pickle fallback is used.
    """
    root=Path(root)
    commit_cmd=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],
                              capture_output=True,text=True)
    sha=commit_cmd.stdout.strip() if commit_cmd.returncode==0 else ''
    known='db51e29535e34bcaa8cb75f70be3fd28b0027988'
    if sha != known:
        return None, (f'Published checkpoint not loaded: unexpected repository SHA {sha!r}; '
                      f'verified commit is {known}. Pin/review source before trusting serialized metadata.')
    paths=[root/'model'/'save'/'checkpoint.pth', root/'model'/'save_1'/'checkpoint.pth']
    checkpoint_file=next((p for p in paths if p.is_file()),None)
    if checkpoint_file is None:
        return None, f'Missing original checkpoint at {[str(p) for p in paths]}'
    # NumPy 1.x serialized objects use numpy.core.*; NumPy 2.x uses numpy._core.*.
    # The class objects are the same, but old pickled module names need aliases.
    core=np._core.multiarray if hasattr(np,'_core') else np.core.multiarray
    safes=[
        (core.scalar, 'numpy.core.multiarray.scalar'),
        (core.scalar, 'numpy._core.multiarray.scalar'),
        (core._reconstruct, 'numpy.core.multiarray._reconstruct'),
        (core._reconstruct, 'numpy._core.multiarray._reconstruct'),
        np.ndarray, np.dtype,
    ]
    for dt in (np.float16,np.float32,np.float64,np.int8,np.int16,np.int32,np.int64,
               np.uint8,np.uint16,np.uint32,np.uint64,np.bool_):
        typ=type(np.dtype(dt))
        if typ not in safes: safes.append(typ)
    try:
        with torch.serialization.safe_globals(safes):
            ck=torch.load(checkpoint_file,map_location='cpu',weights_only=True)
    except Exception as exc:
        return None, (f'PyTorch restricted checkpoint load failed after allowlisting known NumPy '
                      f'metadata types: {type(exc).__name__}: {str(exc)[:1200]}')
    weights=ck.get('best_model',ck.get('model')) if isinstance(ck,dict) else None
    if not isinstance(weights,(dict,)):
        return None, 'Checkpoint does not contain a model/best_model state dict'
    try:
        model.load_state_dict(weights,strict=True)
    except Exception as exc:
        return None, f'Checkpoint/network shape mismatch: {str(exc)[:500]}'
    print(f'Published checkpoint loaded from {checkpoint_file}; best_model keys={len(weights)}')
    return model.to(device).eval(), ''


def evaluate_policy(policy, model, cases, TaskEnv, cfg, device, source_label, seed=None):
    rows=[]
    if isinstance(model,torch.nn.Module): model.eval()
    for i,item in enumerate(cases):
        env=load_case(item,TaskEnv)
        r,_=run_episode(env,model if model is not None else policy,cfg,device,
                        seed=100000+i,sample=False,record=False)
        r.update(method=policy,case_id=item[0],training_seed=(seed if seed is not None else 'published_or_fixed'),
                 dataset_provenance=source_label)
        rows.append(r)
    return rows


def analyze_results(rows, out):
    import pandas as pd
    df=pd.DataFrame(rows)
    if df.empty: return
    cols=['native_reward','success_rate','makespan','time_cost','waiting_time','travel_dist','efficiency','done','runtime_s']
    s=df.groupby(['method','training_seed'],dropna=False)[cols].agg(['mean','std']).reset_index()
    s.columns=['_'.join(str(v) for v in c if v!='') if isinstance(c,tuple) else c for c in s.columns]
    s.to_csv(Path(out)/'summary.csv',index=False)
    # Paired comparisons matched by same case; do not pretend repeated test instances are independent.
    baselines=df[df['method']=='native_greedy'][['case_id','native_reward','makespan']].drop_duplicates('case_id')
    if not baselines.empty:
        paired=df.merge(baselines,on='case_id',how='left',suffixes=('','_baseline'))
        paired['reward_gain_vs_native_greedy']=paired['native_reward']-paired['native_reward_baseline']
        paired['makespan_gain_vs_native_greedy']=paired['makespan_baseline']-paired['makespan']
        paired.to_csv(Path(out)/'paired_vs_native_greedy.csv',index=False)
    print('\n=== Native HeteroMRTA benchmark summary ===')
    print(df.groupby('method')[cols].mean(numeric_only=True).round(4).to_string())
    print('Results saved to:',out)


def main(cfg=None):
    cfg=cfg or Config()
    assert cfg.mode in ('smoke','pilot','full')
    device=torch.device('cuda' if cfg.device=='auto' and torch.cuda.is_available() else
                        'cpu' if cfg.device=='auto' else cfg.device)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    root,commit,TaskEnv,AttentionNet,EnvParams,TrainParams=setup_repository(cfg)
    assert TrainParams.AGENT_INPUT_DIM==11 and TrainParams.TASK_INPUT_DIM==15, \
        'Original input dimension changed: re-audit KAN feature normalizations'
    # Preserve original native environment. No monkeypatching or changing its reward.
    cases, provenance=get_native_dataset(root,TaskEnv,cfg)
    out=Path(cfg.output)/cfg.mode
    out.mkdir(parents=True,exist_ok=True)
    meta=dict(config=asdict(cfg), original_source=REPO, source_commit=commit,
              test_set=provenance, case_count=len(cases),
              original_environment='env.task_env.TaskEnv (unmodified)',
              native_reward='TaskEnv.get_episode_reward (unmodified)',
              policy_action='TaskEnv.agent_observe mask and TaskEnv.agent_step',
              training_method='shared group-baseline REINFORCE (not authors Ray driver.py)',
              training_order='AttentionNet original architecture first, then KAN',
              attention_lr=cfg.attention_learning_rate,kan_lr=cfg.kan_learning_rate,
              fairness_note='published checkpoint training budget may differ from KAN')
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Source SHA:',commit,'Test set:',provenance,'Cases:',len(cases),'Device:',device)
    results=[]
    if cfg.run_native_greedy:
        results+=evaluate_policy('native_greedy',None,cases,TaskEnv,cfg,device,provenance)
    if cfg.run_task_greedy:
        results+=evaluate_policy('task_first_greedy',None,cases,TaskEnv,cfg,device,provenance)
    published_unavailable = False
    if cfg.run_published_attention:
        original=AttentionNet(11,15,TrainParams.EMBEDDING_DIM)
        original,error=load_published_checkpoint(original,root,device)
        if error:
            published_unavailable = True
            (out/'published_checkpoint_error.txt').write_text(error)
            print('WARNING published Attention unavailable:',error)
            if cfg.require_published_attention:
                raise RuntimeError('Published Attention is REQUIRED for this benchmark. '+error)
            print('Published weights missing; retraining architecture uses SHARED trainer and is NOT the published model.')
        else:
            print('Loaded unmodified published Attention checkpoint.')
            results+=evaluate_policy('attention_published',original,cases,TaskEnv,cfg,device,provenance)
    # IMPORTANT TRAINING ORDER: authors' EXACT AttentionNet architecture FIRST, KAN SECOND.
    # The shared trainer is group-baseline REINFORCE, not the original distributed Ray driver.
    # The ORIGINAL published checkpoint (above) is evaluated separately if successfully loaded.
    order_path = out/'training_order.csv'
    order_rows = []
    for ix in range(cfg.start_seed_index,cfg.start_seed_index+cfg.train_seeds):
        seed = cfg.base_seed+10007*ix

        if cfg.retrain_attention:
            print(f'\n=== STAGE 1/2: AUTHORS ORIGINAL AttentionNet ARCHITECTURE | seed={seed} | mode={cfg.mode} ===', flush=True)
            set_seeds(seed)
            author_model = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM,
                                        TrainParams.EMBEDDING_DIM).to(device)
            print('Original AttentionNet parameters:', sum(p.numel() for p in author_model.parameters()),
                  'learning_rate:',cfg.attention_learning_rate, flush=True)
            order_rows.append(dict(training_seed=seed, stage=1, method='attention_retrained_shared',
                                   network_origin='authors_original_attention_py',
                                   trainer='shared_group_REINFORCE_not_authors_Ray_driver',
                                   learning_rate=cfg.attention_learning_rate))
            write_csv(order_path,order_rows)
            author_model=train_model(author_model,'ATTENTION_RETRAINED_SHARED_REINFORCE', TaskEnv,
                                     cfg,seed,out/f'attention_retrained_seed_{seed}',device)
            results+=evaluate_policy('attention_retrained_shared', author_model, cases, TaskEnv,
                                     cfg,device,provenance,seed)
            write_csv(out/'all_test_results.csv',results)
            print('Authors AttentionNet: TRAINED, evaluated, and saved. KAN comes NEXT.',flush=True)
        else:
            print('Authors AttentionNet training disabled explicitly: RETRAIN_ATTENTION=False')

        if cfg.run_kan_training or cfg.kan_checkpoint_path:
            print(f'\n=== STAGE 2/2: CUBIC B-SPLINE KAN | seed={seed} | mode={cfg.mode} ===',flush=True)
            set_seeds(seed)
            kan=KANPolicy(15,11,hidden=24,grid=5).to(device)
            print('KAN parameters:',sum(p.numel() for p in kan.parameters()),
                  'learning_rate:',cfg.kan_learning_rate,flush=True)
            order_rows.append(dict(training_seed=seed, stage=2, method='kan_reinforce',
                                   network_origin='new_cubic_bspline_kan',
                                   trainer='shared_group_REINFORCE_not_authors_Ray_driver',
                                   learning_rate=cfg.kan_learning_rate))
            write_csv(order_path,order_rows)
            if cfg.kan_checkpoint_path:
                saved=torch.load(cfg.kan_checkpoint_path,map_location='cpu',weights_only=True)
                kan.load_state_dict(saved['state_dict'],strict=True)
                kan=kan.to(device).eval()
                print('Loaded earlier KAN policy; no retraining.',flush=True)
            else:
                kan=train_model(kan,'KAN_REINFORCE',TaskEnv,cfg,seed,out/f'kan_seed_{seed}',device)
            results+=evaluate_policy('kan_reinforce',kan,cases,TaskEnv,cfg,device,provenance,seed)
            write_csv(out/'all_test_results.csv',results)
        else:
            print('KAN training disabled explicitly: RUN_KAN_TRAINING=False')

    write_csv(out/'all_test_results.csv',results)
    analyze_results(results,out)
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--mode',default='smoke',choices=['smoke','pilot','full'])
    p.add_argument('--train-seeds',type=int,default=1)
    p.add_argument('--start-seed-index',type=int,default=0)
    p.add_argument('--base-seed',type=int,default=20261104)
    p.add_argument('--repo-path',default='/kaggle/working/HeteroMRTA')
    p.add_argument('--output',default='/kaggle/working/heteromrta_kan_benchmark')
    p.add_argument('--local-repo-zip',default='')
    p.add_argument('--retrain-attention',action='store_true')
    p.add_argument('--max-cases',type=int,default=0)
    args=p.parse_args()
    d=Config(mode=args.mode,train_seeds=args.train_seeds,start_seed_index=args.start_seed_index,
             base_seed=args.base_seed,repo_path=args.repo_path,output=args.output,
             local_repo_zip=args.local_repo_zip,retrain_attention=args.retrain_attention,
             eval_max_cases=args.max_cases)
    main(d)
