#!/usr/bin/env python3
"""Paper 2: selective fine-tuning of Stage A from simulator-assisted teacher plans.

RESEARCH PROTOCOL
- Keep ORIGINAL authors' TaskEnv transitions/action masks intact.
- Four *full* simulated rollout candidates per teacher instance: 2 Guided +
  2 Scarcity-guided. Pick the best complete schedule using the actual Cmax.
- Collect features and actions by EXACT replay of the winning trace.
- Compare against a self-behavior-cloning control trained on original Stage A
  guided-greedy traces, using matched data instances/epochs/batches/seeds.
- Freeze every layer except edge_bias, adapter, kan.kan2 (5,498 params).
- Select checkpoints by SUCCESS then makespan on NEW validation instances.
- Evaluate after all tuning on independent synthetic held-out instances; NO
  official test instances are used for teacher, training, or validation.
- Offline simulator-assisted *planning* is not claimed as online deployment.

Resume: teacher data per-instance saved atomically; training checkpoints per
completed epoch with optimizer state; validated best model stored separately.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import guided_coalition_search as g
import scarcity_bottleneck_search as sc
from train_temporal_spatial_experiments import mixed_env, MODES
from warm_relational_kan import WarmRelationalKAN, count_parameters
from attention import AttentionNet

# Fresh seeds, disjoint from training/validation/test seeds used in experiments 23-28.
TRAIN_SEED = 2306110000
VAL_SEED = 2406110000
TEST_SEED = 2506110000
MODEL_SEED = 20261110
SELECTIVE_PREFIXES = ('adapter.', 'edge_bias.', 'kan.kan2.')
DEFAULT_ALPHA = {'stage_a': 1.1, 'attention': 0.7}
DEFAULT_BETA = {'stage_a': 0.8, 'attention': 0.4}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_torch(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def atomic_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_dict_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_stage_a(path=None):
    path = Path(path or HERE / 'reference' / 'stage_a_best_model.pt')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('model_name') != 'WarmRelationalKAN' or ck.get('stage') != 'A':
        raise RuntimeError(f'Expected trained Stage A WarmRelationalKAN checkpoint: {path}')
    model = WarmRelationalKAN().cpu()
    model.load_state_dict(ck['weights'], strict=True)
    return model


def load_attention():
    f = HERE / 'weights' / 'AttentionNet_published.pt'
    if not f.exists():
        raise FileNotFoundError(f'Missing ORIGINAL published checkpoint: {f}')
    data = torch.load(f, map_location='cpu', weights_only=True)
    if data.get('model_name') != 'AttentionNet_published':
        raise RuntimeError('Attention checkpoint identity mismatch')
    model = AttentionNet(11, 15, 128)
    model.load_state_dict(data['weights'], strict=True)
    return model.eval().cpu()


def seed_for(mode, index, base):
    return int(base + MODES.index(mode) * 100003 + int(index) * 1009)


def choose_candidates(env, model, seed_base, alpha=1.1, beta=0.8):
    """Exactly K=4 full simulated episodes. Stage A guided-greedy is control."""
    choice = sc.Choice(alpha=alpha, beta=beta)
    # 0: guided greedy, 1: guided sample, 2: scarcity greedy, 3: scarcity sample.
    cases = [
        ('guided_greedy', sc.candidate(env, model, seed_base, choice, scarcity=False, sample=False)),
        ('guided_sample', sc.candidate(env, model, seed_base + 7919, choice, scarcity=False, sample=True)),
        ('scarcity_greedy', sc.candidate(env, model, seed_base + 2 * 7919, choice, scarcity=True, sample=False)),
        ('scarcity_sample', sc.candidate(env, model, seed_base + 3 * 7919, choice, scarcity=True, sample=True)),
    ]
    winner = min(cases, key=lambda name_and_data: g._rank(name_and_data[1]))
    # 'self' is the ORIGINAL greedy policy with the ORIGINAL model and guided heuristic.
    return winner, cases[0], cases


def capture_replay(template, winner):
    """Extract exact visible observation, mask, index and teacher label per step.

    Intercepts agent_observe/agent_step ONLY for passive logging; original TaskEnv
    methods and legality checks still execute unmodified. Replays strict
    (agent,time,round,action) trace, checks reward, Cmax and routes afterward.
    """
    run = copy.deepcopy(template)
    orig_observe = run.agent_observe
    orig_step = run.agent_step
    most_recent = {}
    obs_rows = []

    def observe(agent_id, *args, **kwargs):
        task, agent, mask = orig_observe(agent_id, *args, **kwargs)
        most_recent[int(agent_id)] = (np.array(task, copy=True),
                                     np.array(agent, copy=True),
                                     np.array(mask, copy=True))
        return task, agent, mask

    def step(agent_id, action, decision_round):
        aid = int(agent_id)
        if aid not in most_recent:
            raise AssertionError('Unobserved agent at a decision')
        task, agent, mask = most_recent[aid]
        if bool(np.asarray(mask).reshape(-1)[int(action)]):
            raise AssertionError('Teacher tried masked action')
        obs_rows.append((task.reshape(-1, 15), agent.reshape(-1, 11),
                         mask.reshape(-1), aid, int(action)))
        return orig_step(agent_id, action, decision_round)

    run.agent_observe = observe
    run.agent_step = step
    result, trace, routes = g.run_logged_episode(run, None, winner['seed'], scripted=winner['trace'])
    if trace != winner['trace'] or routes != winner['routes']:
        raise AssertionError('Teacher plan replay changed trace or routes')
    for col in ('makespan', 'waiting_time', 'native_reward'):
        if not math.isclose(float(result[col]), float(winner['result'][col]), abs_tol=1e-7, rel_tol=1e-8):
            raise AssertionError(f'Teacher replay mismatch: {col}')
    if int(result['done']) != int(winner['result']['done']):
        raise AssertionError('Teacher completion differed on replay')
    if len(obs_rows) != len(trace):
        raise AssertionError('Number of collected states differs from actions')
    if not obs_rows:
        raise RuntimeError('No observable action in teacher trajectory')
    return {
        'tasks': np.stack([r[0] for r in obs_rows]).astype(np.float32),
        'agents': np.stack([r[1] for r in obs_rows]).astype(np.float32),
        'mask': np.stack([r[2] for r in obs_rows]).astype(np.bool_),
        'agent_id': np.asarray([r[3] for r in obs_rows], np.int64),
        'action': np.asarray([r[4] for r in obs_rows], np.int64),
    }


def save_npz(path, arrays, metadata):
    path = Path(path)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays, meta_json=json.dumps(metadata))
    os.replace(tmp, path)


def make_dataset(args, out):
    out = Path(out)
    root = out / 'demonstrations'
    root.mkdir(parents=True, exist_ok=True)
    stage = load_stage_a().eval()
    set_seed(args.seed)
    records = []
    # Strictly deterministic allocation across onehot/multihot/hard.
    for idx in range(args.teacher_cases):
        mode = MODES[idx % len(MODES)]
        seed = seed_for(mode, idx // len(MODES), TRAIN_SEED + args.seed - MODEL_SEED)
        key = f'{idx:05d}_{mode}_{seed}'
        teacher_path = root / f'{key}_teacher.npz'
        self_path = root / f'{key}_self.npz'
        if teacher_path.exists() and self_path.exists():
            with np.load(teacher_path) as existing:
                meta = json.loads(str(existing['meta_json'].item()))
            records.append(meta)
            continue
        env = mixed_env(seed, mode)
        env.init_state()
        winner, baseline, candidates = choose_candidates(env, stage, seed + 99331)
        # Train only on completed plans, and never pretend an unsuccessful plan is a target.
        if not winner[1]['result']['done'] or not baseline[1]['result']['done']:
            raise RuntimeError(f'Noncompleted training case {idx}. Use independent valid generator.')
        teacher = capture_replay(env, winner[1])
        baseline_data = capture_replay(env, baseline[1])
        improvement = float(baseline[1]['result']['makespan'] - winner[1]['result']['makespan'])
        meta = {'case': idx, 'mode': mode, 'seed': seed,
                'teacher_kind': winner[0], 'teacher_seed': winner[1]['seed'],
                'teacher_makespan': float(winner[1]['result']['makespan']),
                'baseline_makespan': float(baseline[1]['result']['makespan']),
                'improvement': improvement,
                'teacher_actions': len(teacher['action']), 'self_actions': len(baseline_data['action']),
                'teacher_K': 4, 'replay_verified': True}
        save_npz(teacher_path, teacher, meta)
        save_npz(self_path, baseline_data, meta)
        records.append(meta)
        if (idx + 1) % args.log_every == 0 or idx + 1 == args.teacher_cases:
            atomic_csv(records, out / 'teacher_collection.csv')
            mean_gain = float(np.mean([r['improvement'] for r in records]))
            print(f'[TEACHER] {idx+1}/{args.teacher_cases} saved, average greedy-to-best ΔCmax={mean_gain:.4f}', flush=True)
    atomic_csv(records, out / 'teacher_collection.csv')
    return root, records


def freeze_selectively(model):
    grouped = {'adapter': [], 'edge': [], 'kan2': []}
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(SELECTIVE_PREFIXES))
        if not param.requires_grad:
            continue
        if name.startswith('adapter.'):
            grouped['adapter'].append(param)
        elif name.startswith('edge_bias.'):
            grouped['edge'].append(param)
        elif name.startswith('kan.kan2.'):
            grouped['kan2'].append(param)
        else:
            raise AssertionError(f'Unexpected trainable parameter {name}')
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = count_parameters(model)
    if trainable != 5498 or total != 25138:
        raise AssertionError(f'Unexpected topology: trainable={trainable}, total={total}')
    return grouped, trainable, total


def evaluate_greedy(model, n_each, seed_base, *, alpha=1.1):
    """Fixed predeclared synthetic VAL instances; both guided and plain greedy."""
    was_training = model.training
    device = next(model.parameters()).device
    model.cpu().eval()
    values = []
    successes = 0
    by_mode = {}
    with torch.inference_mode():
        for mode in MODES:
            mode_vals = []
            for i in range(n_each):
                seed = seed_for(mode, i, seed_base)
                env = mixed_env(seed, mode)
                env.init_state()
                for use_guided in (False, True):
                    c = g._candidate(env, model, seed + 19881,
                                     guided=use_guided, sample=False,
                                     settings=g.HeuristicSettings(alpha=alpha))
                    success = bool(c['result']['done'])
                    successes += int(success)
                    penalty = float(c['result']['makespan']) if success else 200.0
                    values.append(penalty)
                    mode_vals.append(penalty)
            by_mode[mode] = float(np.mean(mode_vals))
    model.to(device).train(was_training)
    return {'success': successes, 'total': len(values),
            'penalized_makespan': float(np.mean(values)), 'by_mode': by_mode}


def make_optimizer(model, grouped, args):
    return torch.optim.AdamW([
        {'params': grouped['adapter'], 'lr': args.lr_adapter},
        {'params': grouped['edge'], 'lr': args.lr_edge},
        {'params': grouped['kan2'], 'lr': args.lr_kan2},
    ], weight_decay=0.0, eps=1e-8)


def iter_batches(files, batch_size, epoch_seed, pack=8):
    """Randomize file order and observations, with bounded memory use."""
    rng = np.random.default_rng(epoch_seed)
    files = list(files)
    rng.shuffle(files)
    bucket = []
    for filename in files:
        with np.load(filename, allow_pickle=False) as f:
            bucket.append({k: np.array(f[k], copy=True) for k in
                           ('tasks', 'agents', 'mask', 'agent_id', 'action')})
        if len(bucket) < pack:
            continue
        merged = {k: np.concatenate([item[k] for item in bucket], axis=0) for k in bucket[0]}
        indexes = rng.permutation(len(merged['action']))
        for start in range(0, len(indexes), batch_size):
            ix = indexes[start:start + batch_size]
            yield {k: v[ix] for k, v in merged.items()}
        bucket.clear()
    if bucket:
        merged = {k: np.concatenate([item[k] for item in bucket], axis=0) for k in bucket[0]}
        indexes = rng.permutation(len(merged['action']))
        for start in range(0, len(indexes), batch_size):
            ix = indexes[start:start + batch_size]
            yield {k: v[ix] for k, v in merged.items()}


def train_arm(args, out, arm, run_seed):
    assert arm in ('teacher', 'self')
    arm_dir = Path(out) / 'runs' / f'seed_{run_seed}' / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    model = load_stage_a()
    initial = state_dict_cpu(model)
    grouped, trainable, total = freeze_selectively(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model.to(device)
    optimizer = make_optimizer(model, grouped, args)
    ckpath = arm_dir / 'checkpoint_resume.pt'
    history = []
    best_score = None
    start_epoch = 0
    best_epoch = -1
    config = {'arm':arm, 'run_seed':run_seed, 'epochs':args.epochs,
              'teacher_cases':args.teacher_cases, 'batch_size':args.batch_size,
              'lr_adapter':args.lr_adapter, 'lr_edge':args.lr_edge,
              'lr_kan2':args.lr_kan2,'val_each':args.val_each,
              'updates_per_epoch':args.updates_per_epoch,
              'weights_sha256':sha256(HERE / 'reference' / 'stage_a_best_model.pt')}
    if ckpath.exists():
        ck = torch.load(ckpath, map_location=device, weights_only=False)
        if ck['config'] != config:
            raise RuntimeError('Resume config mismatch; use a new output directory')
        model.load_state_dict(ck['weights'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck['epoch']
        best_score = tuple(ck['best_score'])
        best_epoch = ck['best_epoch']
        history = ck['history']
        print(f'[RESUME] {arm} seed={run_seed} epoch={start_epoch}', flush=True)
    else:
        validation = evaluate_greedy(model, args.val_each, VAL_SEED + args.seed - MODEL_SEED)
        best_score = (int(validation['success']), -float(validation['penalized_makespan']))
        best_epoch = 0
        atomic_torch({'weights':state_dict_cpu(model),'epoch':0,'score':best_score,'arm':arm,
                      'run_seed':run_seed, 'trainable':trainable},arm_dir / 'best_model.pt')
        print(f'[INITIAL VAL] {arm} seed={run_seed} {validation}', flush=True)
    files = sorted((Path(out) / 'demonstrations').glob('*_' + arm + '.npz'))
    if len(files) != args.teacher_cases:
        raise RuntimeError(f'Missing {arm} demonstrations {len(files)} != {args.teacher_cases}')
    print(f'[MODEL] {arm} trainable={trainable}/{total}, device={device}',flush=True)
    for epoch in range(start_epoch, args.epochs):
        set_seed(run_seed + epoch * 137)
        model.train()
        losses = []
        steps = 0
        for batch in iter_batches(files, args.batch_size, run_seed + epoch * 7919):
            task = torch.from_numpy(batch['tasks']).to(device)
            agent = torch.from_numpy(batch['agents']).to(device)
            mask = torch.from_numpy(batch['mask']).to(device)
            index = torch.from_numpy(batch['agent_id']).to(device).reshape(-1,1,1)
            labels = torch.from_numpy(batch['action']).to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logp = model(task, agent, mask, index)
            if not bool(torch.isfinite(logp.gather(-1, labels[:,None])).all()):
                raise RuntimeError('Nonfinite selected action log probabilities')
            loss = F.nll_loss(logp, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite supervised imitation loss')
            loss.backward()
            grads = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(grads, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if steps >= args.updates_per_epoch:
                break
        if steps != args.updates_per_epoch:
            raise RuntimeError(f'Not enough demo steps: {steps} vs {args.updates_per_epoch}. Increase teacher cases or reduce --updates-per-epoch.')
        validation = evaluate_greedy(model, args.val_each, VAL_SEED + args.seed - MODEL_SEED)
        score = (int(validation['success']), -float(validation['penalized_makespan']))
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            atomic_torch({'weights':state_dict_cpu(model),'epoch':best_epoch,'score':best_score,
                          'arm':arm,'run_seed':run_seed,'trainable':trainable},arm_dir/'best_model.pt')
            print(f'[BEST] arm={arm} seed={run_seed} epoch={best_epoch}, Cmax={validation["penalized_makespan"]:.4f}',flush=True)
        # Verify that frozen tensors remain bitwise equal to the saved Stage A weights.
        for name, param in model.named_parameters():
            if not param.requires_grad and not torch.equal(param.detach().cpu(), initial[name]):
                raise AssertionError(f'Frozen parameter changed: {name}')
        row={'arm':arm,'seed':run_seed,'epoch':epoch+1,'steps':steps,
             'loss':float(np.mean(losses)),'val_success':validation['success'],
             'val_total':validation['total'],'val_cmax':validation['penalized_makespan'],
             'best_epoch':best_epoch}
        history.append(row)
        atomic_csv(history, arm_dir/'training_history.csv')
        atomic_torch({'config':config, 'epoch':epoch+1,'weights':state_dict_cpu(model),
                      'optimizer':optimizer.state_dict(), 'best_score':best_score,
                      'best_epoch':best_epoch,'history':history},ckpath)
        print(f'[TRAIN] arm={arm} seed={run_seed} epoch={epoch+1}/{args.epochs} '
              f'steps={steps} loss={row["loss"]:.5f} val={row["val_cmax"]:.4f} '
              f'success={row["val_success"]}/{row["val_total"]} best={best_epoch}',flush=True)
    return arm_dir/'best_model.pt'


def load_arm_weights(path):
    model = load_stage_a()
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(ck['weights'], strict=True)
    return model.eval()


def eval_candidate_set(env, model, seed, alpha, beta, k=4):
    """Use 1 Guided greedy + exactly K=4 hybrid candidates, no hidden rollouts."""
    if k != 4:
        raise ValueError('Teacher evaluation currently requires exactly K=4')
    t0 = time.perf_counter()
    win, baseline, cases = choose_candidates(env, model, seed, alpha, beta)
    four_time = time.perf_counter()-t0
    # Plain greedy separately: a distinct K=1 evaluation, not counted into K=4.
    t1 = time.perf_counter()
    plain = g._candidate(env,model,seed, guided=False, sample=False,
                         settings=g.HeuristicSettings(alpha=0.0))
    plain_time = time.perf_counter()-t1
    return [('plain_greedy_K1', plain, plain_time),
            ('guided_greedy_K1', baseline[1], float(baseline[1]['result']['runtime_s'])),
            ('hybrid_guided_scarcity_K4', win[1], four_time)]


def evaluate_final(args, out):
    """Test is first touched after ALL training and validation are done."""
    out = Path(out)
    models = {'stage_a': load_stage_a().eval()}
    for run_seed in args.seeds:
        for arm in ('self','teacher'):
            path = out/'runs'/f'seed_{run_seed}'/arm/'best_model.pt'
            if not path.exists(): raise FileNotFoundError(f'Missing best weights: {path}')
            models[f'{arm}_seed{run_seed}'] = load_arm_weights(path)
    # Published baseline MUST be present; never silently replace its weights.
    models['attention'] = load_attention()
    rows = []
    for mode in MODES:
        for i in range(args.test_each):
            case_seed = seed_for(mode,i,TEST_SEED + args.seed - MODEL_SEED)
            env = mixed_env(case_seed,mode)
            env.init_state()
            for name,model in models.items():
                aa = DEFAULT_ALPHA['attention' if name=='attention' else 'stage_a']
                bb = DEFAULT_BETA['attention' if name=='attention' else 'stage_a']
                chosen = eval_candidate_set(env,model,case_seed+27911,aa,bb, k=4)
                for method, x, elapsed in chosen:
                    r = x['result']
                    # Independently verify winner. During smoke and full runs.
                    g.replay_and_verify(env, x)
                    rows.append({'mode':mode,'case':i,'seed':case_seed,'model':name,
                                 'method':method,'done':int(r['done']),
                                 'makespan':float(r['makespan']),
                                 'penalized_makespan':float(r['makespan'] if r['done'] else 200.),
                                 'waiting_time':float(r['waiting_time']),
                                 'native_reward':float(r['native_reward']),
                                 'candidate_search_elapsed_s':elapsed,
                                 'replay_verified':1})
            atomic_csv(rows,out/'evaluation_per_case.csv')
            print(f'[TEST] {mode} {i+1}/{args.test_each} all models, K=1/K=4 replay OK',flush=True)
    summary = []
    for name,method in sorted({(r['model'],r['method']) for r in rows}):
        for mode in ('ALL',)+MODES:
            rs=[r for r in rows if r['model']==name and r['method']==method and (mode=='ALL' or r['mode']==mode)]
            summary.append({'model':name,'method':method,'mode':mode,'cases':len(rs),
                            'success':sum(r['done'] for r in rs),
                            'mean_penalized_makespan':float(np.mean([r['penalized_makespan'] for r in rs])),
                            'mean_waiting':float(np.mean([r['waiting_time'] for r in rs])),
                            'mean_search_elapsed_s':float(np.mean([r['candidate_search_elapsed_s'] for r in rs]))})
    atomic_csv(summary,out/'evaluation_summary.csv')
    paired=[]
    for method in sorted({r['method'] for r in rows}):
        for aa,bb in [('teacher_seed'+str(s),'stage_a') for s in args.seeds]+[
                ('teacher_seed'+str(s),'self_seed'+str(s)) for s in args.seeds]+[
                ('teacher_seed'+str(s),'attention') for s in args.seeds]:
            a={(r['mode'],r['case']):r['penalized_makespan'] for r in rows if r['model']==aa and r['method']==method}
            b={(r['mode'],r['case']):r['penalized_makespan'] for r in rows if r['model']==bb and r['method']==method}
            ix=sorted(set(a)&set(b)); diff=np.asarray([a[i]-b[i] for i in ix])
            rng=np.random.default_rng(21110)
            bymode={m:np.asarray([j for j,k in enumerate(ix) if k[0]==m]) for m in MODES}
            boot=[]
            for _ in range(1000):
                subs=np.concatenate([rng.choice(inds, size=len(inds),replace=True) for inds in bymode.values()])
                boot.append(float(np.mean(diff[subs])))
            paired.append({'method':method,'model_a':aa,'model_b':bb,'cases':len(ix),
                           'delta_a_minus_b':float(np.mean(diff)),
                           'ci95_lower':float(np.quantile(boot,.025)),
                           'ci95_upper':float(np.quantile(boot,.975))})
    atomic_csv(paired,out/'paired_statistics.csv')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='/kaggle/working/SELECTIVE_DISTILLATION_RESULTS')
    p.add_argument('--teacher-cases',type=int,default=1500)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--updates-per-epoch',type=int,default=300,
                   help='Exactly matched optimizer updates for self-BC and teacher arms')
    p.add_argument('--val-each',type=int,default=12)
    p.add_argument('--test-each',type=int,default=50)
    p.add_argument('--seeds',type=int,nargs='+',default=[20261110,20261111,20261112])
    p.add_argument('--seed',type=int,default=MODEL_SEED)
    p.add_argument('--lr-adapter',type=float,default=1e-5)
    p.add_argument('--lr-edge',type=float,default=2e-5)
    p.add_argument('--lr-kan2',type=float,default=2e-6)
    p.add_argument('--log-every',type=int,default=25)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.smoke:
        a.teacher_cases=3; a.epochs=1; a.val_each=1; a.test_each=1; a.seeds=[20261110]; a.updates_per_epoch=2
    if min(a.teacher_cases,a.epochs,a.batch_size,a.updates_per_epoch,a.val_each,a.test_each)<1:
        raise ValueError('Positive dimensions required')
    torch.set_num_threads(1)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    stage_file=HERE/'reference'/'stage_a_best_model.pt'
    provenance={'train_seed_base':TRAIN_SEED,'validation_seed_base':VAL_SEED,
                'heldout_test_seed_base':TEST_SEED,'seeds':a.seeds,
                'teacher_cases':a.teacher_cases,'epochs':a.epochs,'batch_size':a.batch_size,
                'updates_per_epoch':a.updates_per_epoch,
                'teacher_search_budget':4, 'teacher_methods':[
                  'guided_greedy','guided_sample','scarcity_greedy','scarcity_sample'],
                'teacher_alpha':1.1,'teacher_beta':0.8,
                'trainable_names':SELECTIVE_PREFIXES,'trainable_parameters':5498,
                'checkpoint_stage_A_sha256':sha256(stage_file),
                'methodology':'offline simulator-assisted policy distillation; self-BC supervised control; no new RL reward used',
                'test_claim':'held-out synthetic only; no official test for tuning or test',
                'smoke':bool(a.smoke)}
    atomic_json(provenance,out/'provenance.json')
    print('[START] selective Stage A 5498/25138 params; Teacher K=4; official excluded',flush=True)
    _, teacher_rows = make_dataset(a,out)
    print('[DATASET] cases=',len(teacher_rows),'saved trace teacher/self verified',flush=True)
    for seed in a.seeds:
        for arm in ('self','teacher'):
            train_arm(a,out,arm,seed)
    summary = evaluate_final(a,out)
    print('[DONE] completed teacher distillation + matched self-BC + heldout evaluation',flush=True)
    for row in summary:
        if row['mode']=='ALL':
            print('[SUMMARY]',row['model'],row['method'],row['mean_penalized_makespan'],
                  'success',row['success'],'/',row['cases'],flush=True)


if __name__=='__main__':
    main()
