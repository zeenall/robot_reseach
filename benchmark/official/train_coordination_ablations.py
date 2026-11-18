#!/usr/bin/env python3
"""Paper 2 follow-on: matched control, coalition-arrival, waiting penalty; search at evaluation.

Three INDEPENDENT training arms each warm-start from the same Stage A best_model.pt:
  control -- exact Stage-A network, original native reward
  arrival -- Stage-A network + a zero-initialized coordination logit residual
  waiting -- exact Stage-A network, native reward - lambda * mean waiting
Training instances/seeds, validation cases, jobs, PPO-like? NO: original group-centered
REINFORCE + POMO worker retained. Original authors' TaskEnv untouched.
At test-time, search is a separate inference-only best-of-K rollout comparison.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, random, shutil, signal, sys, time, traceback
from collections import deque
from pathlib import Path
import numpy as np
import torch

from env.task_env import TaskEnv
from attention import AttentionNet
from warm_relational_kan import WarmRelationalKAN, count_parameters
from coalition_arrival_kan import CoalitionArrivalKAN
from train_temporal_spatial_experiments import (MixedRewardWorker, mixed_env, MODES,
    evaluate_dataset, profile_for_job)
from train_warm_hybrid import (Trainer as BaseTrainer, seed_all, cpustate,
    atomic_torch_save, write_csv)
from benchmark_core import run_episode
from train_temporal_spatial_experiments import Limits, official_env
from evaluate_coordination_search import evaluate_final

HERE=Path(__file__).resolve().parent
STAGE_A=HERE/'reference'/'stage_a_best_model.pt'
ARM_NAMES=('control','arrival','waiting')
TRAIN_SEED_START=917000000
VALID_SEED_START=731000000


def load_stage_a():
    blob=torch.load(STAGE_A,map_location='cpu',weights_only=True)
    if blob.get('model_name')!='WarmRelationalKAN' or blob.get('stage')!='A':
        raise RuntimeError('Expected results(25) Stage-A best_model.pt (not original hybrid)')
    if blob.get('best_job')!=3600:
        raise RuntimeError(f"Unexpected Stage A best job {blob.get('best_job')}")
    model=WarmRelationalKAN()
    model.load_state_dict(blob['weights'],strict=True)
    return model


def create_arm(arm):
    stage_a=load_stage_a()
    if arm!='arrival':return stage_a
    model=CoalitionArrivalKAN()
    model.transfer_stage_a(stage_a)
    stage_a.eval();model.eval()
    e=mixed_env(4117,'hard');e.init_state()
    t,a,m=e.agent_observe(0)
    with torch.inference_mode():
        old,_=stage_a(torch.as_tensor(t,dtype=torch.float32),torch.as_tensor(a,dtype=torch.float32),
                      torch.as_tensor(m),torch.tensor([[[0]]]))
        new,_=model(torch.as_tensor(t,dtype=torch.float32),torch.as_tensor(a,dtype=torch.float32),
                    torch.as_tensor(m),torch.tensor([[[0]]]))
    err=float((old-new).abs().max())
    if err>1.e-6:raise RuntimeError(f'Invalid warmstart, probability delta={err}')
    print(f'[TRANSFER] Stage-A -> CoalitionArrivalKAN; probability delta={err:.2g}',flush=True)
    return model


def val_seeds(n):
    # These validation environments differ from all previous Stage-A validation/test seeds.
    return {mode:[VALID_SEED_START+1000003*j+1009*i for i in range(n)]
            for j,mode in enumerate(MODES)}


class Collector:
    def __init__(self,actor_id,arm,waiting_penalty,seed):
        torch.set_num_threads(1)
        seed_all(700000+actor_id)
        self.actor_id=actor_id
        self.arm=arm
        self.model=WarmRelationalKAN() if arm!='arrival' else CoalitionArrivalKAN()
        self.model.cpu().eval()
        self.waiting_penalty=waiting_penalty if arm=='waiting' else 0.
        self.base_seed=TRAIN_SEED_START + int(seed)*13

    def collect(self,weights,job):
        seed=self.base_seed+104729*int(job)
        seed_all(seed)
        self.model.load_state_dict(weights,strict=True)
        mode=profile_for_job(job)
        wk=MixedRewardWorker(self.actor_id,self.model,job,seed,mode,self.waiting_penalty)
        wk.work(job)
        return wk.experience,wk.perf_metrics,{'job':job,'mode':mode}


class ArmTrainer(BaseTrainer):
    def __init__(self,args,arm):
        self.args=args;self.arm=arm
        self.stage=arm
        self.work=Path(args.out).resolve()/arm
        self.work.mkdir(parents=True,exist_ok=True)
        self.device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
        torch.set_num_threads(1)
        seed_all(args.seed)
        self.model=create_arm(arm).to(self.device)
        self.initial=cpustate(self.model)
        assert all(p.requires_grad for p in self.model.parameters())
        extra=[p for n,p in self.model.named_parameters() if not n.startswith('kan.')]
        self.opt=torch.optim.Adam([
          {'params':list(self.model.kan.parameters()),'lr':args.lr_kan,'base_lr':args.lr_kan},
          {'params':extra,'lr':args.lr_attention,'base_lr':args.lr_attention}],eps=1e-8)
        self.completed=set();self.buffer={i:[] for i in range(7)}
        self.history=[];self.validation=[];self.best=cpustate(self.model)
        self.best_score=None;self.best_step=0;self.updates=0
        self.last_loss=float('nan');self.last_grad=float('nan');self.last_entropy=float('nan')
        self.recent=deque(maxlen=10);self.stop=False;self.start_time=time.monotonic()
        self.checkpoint_path=self.work/'checkpoint_resume.pt'
        config={'version':'COALITION_ABLATION_V3','arm':arm,'jobs':args.jobs,'seed':args.seed,
                'batch_size':args.batch_size,'lr_kan':args.lr_kan,'lr_attention':args.lr_attention,
                'entropy_coef':args.entropy_coef,'clip_grad':args.clip_grad,
                'waiting_penalty':args.waiting_penalty if arm=='waiting' else 0.,
                'val_each':args.val_each,'val_every':args.val_every,
                'save_every':args.save_every,'log_every':args.log_every,
                'actors':args.actors,'backend':args.backend,
                'validation_seeds':VALID_SEED_START,
                'training_seeds':TRAIN_SEED_START,'profile':'4onehot4multi2hard',
                'pomo':10,'stage_a_sha256':hashlib.sha256(STAGE_A.read_bytes()).hexdigest()}
        self.fingerprint=json.dumps(config,sort_keys=True)
        if self.checkpoint_path.exists():
            self._resume()
            # A previous Kaggle output may contain only checkpoint_resume.pt.
            # Restore the SELECTED best model from its embedded state rather
            # than accidentally replacing it with the latest weights.
            selected=self.work/'best_model.pt'
            if not selected.exists() and self.best is not None and self.best_score is not None:
                atomic_torch_save({'arm':self.arm,'model_name':type(self.model).__name__,
                    'weights':self.best,'best_job':self.best_step,
                    'validation_score':self.best_score,
                    'source_sha256':hashlib.sha256(STAGE_A.read_bytes()).hexdigest()},selected)
                print(f'[RESTORED BEST] arm={arm} job={self.best_step} from resume checkpoint',flush=True)
        self.actors=[]
        print(f'[MODEL] arm={arm} parameters={count_parameters(self.model)} '
              f'waiting_lambda={args.waiting_penalty if arm=="waiting" else 0.} '
              f'device={self.device} done={len(self.completed)}/{args.jobs}',flush=True)

    def _validate(self,label):
        rows=[]
        for mode,seeds in val_seeds(self.args.val_each).items():
            metr,_=evaluate_dataset(self.model,mode,seeds,self.device,'validation')
            metr['label']=label;metr['jobs_completed']=len(self.completed)
            rows.append(metr)
        self.validation.extend(rows)
        successes=sum(x['completed'] for x in rows)
        n=sum(x['cases'] for x in rows)
        mean_penalized=sum(x['penalized_makespan']*x['cases'] for x in rows)/n
        score=(successes,-mean_penalized)
        print(f'[VAL] arm={self.arm} jobs={len(self.completed)} success={successes}/{n} '
              f'penalized_makespan={mean_penalized:.4f} '+
              ' '.join(f'{r["mode"]}:{r["penalized_makespan"]:.2f}' for r in rows),flush=True)
        self.model.train()
        if self.best_score is None or score>self.best_score:
            self.best=cpustate(self.model)
            self.best_score=score;self.best_step=len(self.completed)
            atomic_torch_save({'arm':self.arm,'model_name':type(self.model).__name__,
                'weights':self.best,'best_job':self.best_step,'validation_score':self.best_score,
                'source_sha256':hashlib.sha256(STAGE_A.read_bytes()).hexdigest()},
                self.work/'best_model.pt')
            print(f'[BEST] arm={self.arm} job={self.best_step} on fixed validation',flush=True)

    def _finalize(self):
        if self.buffer[0]:self._update(len(self.buffer[0]))
        self._validate('after_all_updates')
        delta=sum((x.detach().cpu()-self.initial[k]).abs().sum().item()
                  for k,x in self.model.state_dict().items())
        (self.work/'training_provenance.json').write_text(json.dumps({
            'arm':self.arm,'start':'Stage A best_model.pt, job=3600','jobs':len(self.completed),
            'pomo':10*len(self.completed),'optimizer_updates':self.updates,
            'best_job':self.best_step,'initial_to_last_L1':delta,
            'parameter_count':count_parameters(self.model),
            'wait_lambda':self.args.waiting_penalty if self.arm=='waiting' else 0.,
            'seed':self.args.seed,'checkpoint_selection':'success then validation penalized makespan',
            'training_seed_start':TRAIN_SEED_START,'validation_seed_start':VALID_SEED_START},indent=2))
        self._save('training_complete')
        print(f'[DONE] arm={self.arm} jobs={len(self.completed)}/{self.args.jobs} '
              f'updates={self.updates} best_job={self.best_step} L1={delta:.3f}',flush=True)

    def run(self):
        if self.best_score is None:
            self._validate('stage_a_initial')
            self._save('initial')
        if self.args.smoke:
            actor=Collector(0,self.arm,self.args.waiting_penalty,self.args.seed)
            for job in range(1,self.args.smoke_jobs+1):
                if job not in self.completed:
                    self._consume(job,actor.collect(cpustate(self.model),job))
            if self.buffer[0]:self._update(len(self.buffer[0]))
            self._save('smoke_done')
            print(f'[SMOKE] {self.arm} finished {self.args.smoke_jobs} real Worker/POMO jobs',flush=True)
            return
        from importlib.util import find_spec
        if self.args.backend=='local' or (self.args.backend=='auto' and find_spec('ray') is None):
            print('[BACKEND] sequential local',flush=True)
            actor=Collector(0,self.arm,self.args.waiting_penalty,self.args.seed)
            for job in range(1,self.args.jobs+1):
                if job in self.completed:continue
                if self.stop or time.monotonic()-self.start_time>=self.args.session_hours*3600:
                    self._save('safe_stop');break
                self._consume(job,actor.collect(cpustate(self.model),job))
        else:
            import ray
            ray.init(num_cpus=self.args.actors,include_dashboard=False,
                     log_to_driver=False,ignore_reinit_error=True)
            @ray.remote(num_cpus=1,max_restarts=0)
            class RemoteActor:
                def __init__(self,idx,arm,waiting_penalty,seed):
                    self.worker=Collector(idx,arm,waiting_penalty,seed)
                def collect(self,weights,job):return self.worker.collect(weights,job)
            actors=[RemoteActor.remote(i,self.arm,self.args.waiting_penalty,self.args.seed)
                    for i in range(self.args.actors)]
            todo=deque(job for job in range(1,self.args.jobs+1) if job not in self.completed)
            active={}
            try:
                for i in range(min(len(todo),len(actors))):
                    job=todo.popleft()
                    active[actors[i].collect.remote(cpustate(self.model),job)]=(i,job)
                while active:
                    if self.stop or time.monotonic()-self.start_time>=self.args.session_hours*3600:
                        self._save('safe_stop');break
                    ready,_=ray.wait(list(active),num_returns=1,timeout=15)
                    if not ready:continue
                    ref=ready[0]
                    i,job=active.pop(ref)
                    self._consume(job,ray.get(ref))
                    if todo:
                        nxt=todo.popleft()
                        active[actors[i].collect.remote(cpustate(self.model),nxt)]=(i,nxt)
            finally:
                for actor in actors:
                    try:ray.kill(actor,no_restart=True)
                    except Exception:pass
                ray.shutdown()
        if len(self.completed)==self.args.jobs:self._finalize()


def import_previous(out,source):
    if not source:return
    root=Path(source)
    if not root.is_dir():raise FileNotFoundError(root)
    for arm in ARM_NAMES:
        dest=out/arm/'checkpoint_resume.pt'
        if dest.exists():continue
        matches=[p for p in root.rglob('checkpoint_resume.pt') if p.parent.name==arm]
        if len(matches)>1:raise RuntimeError(f'Ambiguous resume for {arm}: {matches}')
        if matches:
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(matches[0],dest)
            print('[IMPORTED RESUME]',matches[0],flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='/kaggle/working/COALITION_ABLATION_RESULTS')
    p.add_argument('--arm',choices=('all','control','arrival','waiting','eval'),default='all')
    p.add_argument('--jobs',type=int,default=1200,help='per arm')
    p.add_argument('--seed',type=int,default=20261117)
    p.add_argument('--actors',type=int,default=3)
    p.add_argument('--backend',choices=('auto','ray','local'),default='auto')
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--lr-kan',type=float,default=3e-6)
    p.add_argument('--lr-attention',type=float,default=2e-5)
    p.add_argument('--waiting-penalty',type=float,default=0.2)
    p.add_argument('--entropy-coef',type=float,default=.001)
    p.add_argument('--clip-grad',type=float,default=10.)
    p.add_argument('--val-each',type=int,default=20)
    p.add_argument('--val-every',type=int,default=100)
    p.add_argument('--save-every',type=int,default=25)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--test-each',type=int,default=50,help='per heldout mode')
    p.add_argument('--official-cases',type=int,default=50)
    p.add_argument('--search-k',type=int,default=4)
    p.add_argument('--session-hours',type=float,default=8.)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-jobs',type=int,default=1)
    p.add_argument('--resume-dir',default='')
    args=p.parse_args()
    if min(args.jobs,args.actors,args.batch_size,args.val_each,args.val_every,args.search_k)<1:
        p.error('All counts must be positive')
    if args.waiting_penalty<0:p.error('Waiting penalty must be nonnegative')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    import_previous(out,args.resume_dir)
    try:
        arms=ARM_NAMES if args.arm=='all' else () if args.arm=='eval' else (args.arm,)
        for arm in arms:
            t=ArmTrainer(args,arm)
            def graceful(sig,frame):
                t.stop=True
                print(f'[SIGNAL] {sig} graceful stop requested',flush=True)
            signal.signal(signal.SIGINT,graceful)
            signal.signal(signal.SIGTERM,graceful)
            t.run()
            if args.smoke:continue
            if len(t.completed)!=args.jobs:
                print('[INCOMPLETE] Resume from checkpoint before final evaluation',flush=True)
                return
        if not args.smoke and args.arm in ('all','eval'):
            evaluate_final(out,args)
        print('[DONE] training/evaluation operation complete',flush=True)
    except BaseException as exc:
        print('[ERROR]',repr(exc),flush=True)
        traceback.print_exc()
        if 't' in locals():t._save('exception')
        raise

if __name__=='__main__':main()
