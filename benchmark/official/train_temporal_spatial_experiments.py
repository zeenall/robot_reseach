#!/usr/bin/env python3
"""Paper 2: controlled Stage A (data-only) and Stage B (data+time/space edges).

Both start from IDENTICAL published-to-us best_hybrid.pt, never from each other.
The authors' original TaskEnv transitions/observation mask and Worker/POMO
are untouched. Only our training-instance generator subclasses TaskEnv.
Official RALTestSet is DIAGNOSTIC after training, never for checkpoint selection.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, os, random, shutil, signal, sys, time, traceback
from collections import deque
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from env.task_env import TaskEnv
from worker import Worker
from parameters import TrainParams, SaverParams
from attention import AttentionNet
from kan_policy import KANPolicy
from warm_relational_kan import WarmRelationalKAN, count_parameters
from temporal_spatial_kan import TemporalSpatialKAN, verify_transfer
from benchmark_core import run_episode
from train_warm_hybrid import (Trainer as BaseTrainer, Limits, seed_all, cpustate,
    atomic_torch_save, write_csv, mean_safe, official_env)

TrainParams.POMO_SIZE=10
TrainParams.USE_GPU=False
TrainParams.NUM_GPU=0
SaverParams.SAVE_IMG=False
SEED_BASE=616060000


def training_seed_base(run_seed):
    # Matched A/B runs share identical training-instance seeds, while distinct
    # run_seed values produce genuinely different training experiences.
    return SEED_BASE + 10000019 * (int(run_seed) - 20261109)
VALIDATION_SEED=822000000
TEST_SEED=969000000
MODES=('onehot','multihot','hard')


class MixedCapabilityEnv(TaskEnv):
    """Same TaskEnv state/action/reward, only generating capability profiles.

    onehot: 5 unique species with exactly one ability each, 3 robots/species.
    multihot: original TaskEnv random binary capability generator.
    hard: onehot robots, requirements demanding 2-3 different specialists.
    """
    def __init__(self,*args,mode='multihot',**kwargs):
        if mode not in MODES:raise ValueError(mode)
        self.capability_mode=mode
        super().__init__(*args,**kwargs)

    def generate_agent(self,species_num):
        if self.capability_mode in ('onehot','hard'):
            if species_num != self.traits_dim:
                raise ValueError('Onehot generator needs one species per trait; select 5 species')
            order=np.asarray(self.random_choice(self.traits_dim,size=species_num,replace=False),dtype=int)
            return np.eye(self.traits_dim,dtype=int)[order]
        return super().generate_agent(species_num)

    def generate_task(self,tasks_num):
        if self.capability_mode != 'hard':
            return super().generate_task(tasks_num)
        # Multi-trait coalitions; 2 or 3 different specialist skills/task.
        req=np.zeros((tasks_num,self.traits_dim),dtype=int)
        for i in range(tasks_num):
            k=int(self.random_int(2,4))
            idx=self.random_choice(self.traits_dim,size=k,replace=False)
            req[i,idx]=1
        return req


def profile_for_job(job:int):
    # 40% onehot, 40% multihot, 20% hard, over each block of 10 jobs.
    return ('onehot','onehot','onehot','onehot',
            'multihot','multihot','multihot','multihot',
            'hard','hard')[(job-1)%10]


def mixed_env(seed,mode):
    return MixedCapabilityEnv((3,3),(5,5),(20,20),5,30,
                              seed=int(seed),plot_figure=False,mode=mode)


def get_initial_hybrid():
    ck=torch.load(HERE/'reference'/'best_hybrid.pt',map_location='cpu',weights_only=True)
    if ck.get('model_name') != 'WarmRelationalKAN':
        raise RuntimeError('Wrong pretrained hybrid identity')
    model=WarmRelationalKAN()
    model.load_state_dict(ck['weights'],strict=True)
    return model


def create_model(stage):
    old=get_initial_hybrid()
    if stage=='A': return old
    if stage!='B':raise ValueError(stage)
    model=TemporalSpatialKAN()
    model.transfer_pretrained(old)
    e=mixed_env(1777,'onehot'); e.init_state()
    tasks,agents,mask=e.agent_observe(0)
    diff=verify_transfer(old,model,torch.tensor(tasks,dtype=torch.float32),
        torch.tensor(agents,dtype=torch.float32),torch.tensor(mask),
        torch.tensor([[[0]]],dtype=torch.long))
    print(f'[TRANSFER] stage B from ORIGINAL best_hybrid.pt; exact action-probability max delta={diff:.9g}',flush=True)
    return model


class MixedRewardWorker(Worker):
    def __init__(self,actor_id,model,job,seed,mode,waiting_penalty):
        # Use the exact authors' decision procedure and group-centered POMO.
        super().__init__(actor_id,model,model,job,device='cpu',
                         save_image=False,seed=seed,
                         env_params=[(3,3),(5,5),(20,20)])
        self.env=mixed_env(seed,mode)
        self.baseline_env=copy.deepcopy(self.env)
        self.waiting_penalty=float(waiting_penalty)

    def run_episode(self,*args,**kwargs):
        reward,buffer,metrics=super().run_episode(*args,**kwargs)
        original=float(reward)
        waiting=float(np.mean(metrics['waiting_time']))
        shaped=original - self.waiting_penalty * waiting
        metrics['native_reward']=[original]
        metrics['terminal_reward']=[shaped]
        return shaped,buffer,metrics


class Collector:
    def __init__(self,actor_id,stage,waiting_penalty,run_seed):
        torch.set_num_threads(1)
        seed_all(700000+actor_id)
        self.actor_id=actor_id
        self.model=WarmRelationalKAN() if stage=='A' else TemporalSpatialKAN()
        self.model.cpu().eval()
        self.waiting_penalty=waiting_penalty
        self.training_seed_base=training_seed_base(run_seed)

    def collect(self,weights,job):
        seed=self.training_seed_base + 104729*int(job)
        seed_all(seed)
        self.model.load_state_dict(weights,strict=True)
        mode=profile_for_job(job)
        wk=MixedRewardWorker(self.actor_id,self.model,job,seed,mode,self.waiting_penalty)
        wk.work(job)
        return wk.experience,wk.perf_metrics,{'job':job,'mode':mode}


def validation_seeds(n):
    return {m:[VALIDATION_SEED + 1000003*j + 1009*i for i in range(n)]
            for j,m in enumerate(MODES)}


def evaluate_dataset(model,mode,seeds,device,group='val'):
    model.eval()
    rows=[]
    with torch.inference_mode():
        for i,s in enumerate(seeds):
            e=mixed_env(s,mode)
            e.init_state()
            outcome,_=run_episode(e,model,Limits,device,seed=20261900+i,
                                  sample=False,record=False)
            rows.append(outcome)
    completed=sum(int(r['done']) for r in rows)
    return {'group':group,'mode':mode,'cases':len(rows),'completed':completed,
            'penalized_makespan':mean_safe(r['makespan'] if r['done'] else 200. for r in rows),
            'makespan_completed':mean_safe(r['makespan'] for r in rows if r['done']),
            'waiting':mean_safe(r['waiting_time'] for r in rows),
            'reward':mean_safe(r['native_reward'] for r in rows)},rows


class ExperimentTrainer(BaseTrainer):
    def __init__(self,args,stage):
        # Deliberately do not call BaseTrainer.__init__; it would warmstart from
        # the ancient KAN checkpoint instead of the trained hybrid.
        self.stage=stage
        self.args=args
        self.work=Path(args.out).resolve()/f'stage_{stage.lower()}'
        self.work.mkdir(parents=True,exist_ok=True)
        self.device=torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
        torch.set_num_threads(1)
        seed_all(args.seed)
        self.model=create_model(stage).to(self.device)
        self.initial=cpustate(self.model)
        if not all(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Some parameters frozen')
        print(f'[MODEL] stage={stage} parameters={count_parameters(self.model)} '
              f'no_frozen_layers=True device={self.device}',flush=True)
        extra=[p for name,p in self.model.named_parameters() if not name.startswith('kan.')]
        self.opt=torch.optim.Adam([
            {'params':list(self.model.kan.parameters()),'lr':args.lr_kan,'base_lr':args.lr_kan},
            {'params':extra,'lr':args.lr_attention,'base_lr':args.lr_attention}],eps=1e-8)
        self.completed=set()
        self.buffer={i:[] for i in range(7)}
        self.history=[]; self.validation=[]
        self.best=cpustate(self.model)
        self.best_score=None; self.best_step=0; self.updates=0
        self.last_loss=float('nan'); self.last_grad=float('nan'); self.last_entropy=float('nan')
        self.recent=deque(maxlen=10)
        self.stop=False; self.start_time=time.monotonic()
        self.checkpoint_path=self.work/'checkpoint_resume.pt'
        # Strict configuration fingerprint prevents accidental resume with changed experiment.
        config={'stage':stage,'jobs':args.jobs,'seed':args.seed,'batch_size':args.batch_size,
                'lr_kan':args.lr_kan,'lr_attention':args.lr_attention,
                'entropy_coef':args.entropy_coef,'clip_grad':args.clip_grad,
                'waiting_penalty':args.waiting_penalty,'val_each':args.val_each,
                'seed_base':training_seed_base(args.seed),'mode_cycle':'4onehot4multi2hard','source':'best_hybrid.pt',
                'source_sha256':hashlib.sha256((HERE/'reference'/'best_hybrid.pt').read_bytes()).hexdigest(),
                'pomo':10,'arch_version':'tsk_2026_11_09_v1'}
        self.fingerprint=json.dumps(config,sort_keys=True)
        if self.checkpoint_path.exists():self._resume()
        self.actors=[]

    def _validate(self,label):
        rows=[]
        for mode,seeds in validation_seeds(self.args.val_each).items():
            metric,_=evaluate_dataset(self.model,mode,seeds,self.device,'validation')
            metric['label']=label
            metric['jobs_completed']=len(self.completed)
            rows.append(metric)
        self.validation.extend(rows)
        successes=sum(r['completed'] for r in rows)
        n=sum(r['cases'] for r in rows)
        mean_penalized=sum(r['penalized_makespan']*r['cases'] for r in rows)/n
        score=(successes,-mean_penalized)
        print(f'[VAL] stage={self.stage} jobs={len(self.completed)} '
              f'success={successes}/{n} penalized_makespan={mean_penalized:.4f} '
              + ' '.join(f"{r['mode']}:{r['penalized_makespan']:.2f}" for r in rows),flush=True)
        self.model.train()
        if self.best_score is None or score>self.best_score:
            self.best=cpustate(self.model)
            self.best_score=score
            self.best_step=len(self.completed)
            atomic_torch_save({'model_name':'WarmRelationalKAN' if self.stage=='A' else 'TemporalSpatialKAN',
                               'stage':self.stage,'weights':self.best,'best_job':self.best_step,
                               'validation_score':self.best_score,
                               'source_checkpoint':'best_hybrid.pt'},
                              self.work/'best_model.pt')
            print(f'[BEST] stage={self.stage} new best at {self.best_step} jobs',flush=True)

    def _consume(self,job,result):
        super()._consume(job,result)

    def _collect_params(self,job):
        return None

    def _test_models(self):
        # Run at top-level after stages, NOT inside base Trainer.run().
        return None

    def _finalize(self):
        if self.buffer[0]:self._update(len(self.buffer[0]))
        self._validate('after_all_updates')
        delta=float(sum((v.detach().cpu()-self.initial[k]).abs().sum().item()
                        for k,v in self.model.state_dict().items()))
        (self.work/'training_provenance.json').write_text(json.dumps({
            'stage':self.stage,'source':'reference/best_hybrid.pt',
            'initial_to_last_weights_L1':delta,
            'training_jobs':len(self.completed),'pomo':10*len(self.completed),
            'optimizer_updates':self.updates,'selected_best_job':self.best_step,
            'parameters':count_parameters(self.model),
            'waiting_penalty':self.args.waiting_penalty,
            'validation_only_checkpoint_selection':True,
            'train_mode_ratio':{'onehot':.4,'multihot':.4,'hard':.2}},indent=2))
        self._save('training_complete')
        print(f'[DONE] stage={self.stage} training={len(self.completed)}/{self.args.jobs} '
              f'updates={self.updates} best_job={self.best_step} L1_change={delta:.5f}',flush=True)

    def run(self):
        if self.best_score is None:
            self._validate('pretrained_hybrid_initial')
            self._save('initial')
        if self.args.smoke:
            # Real original Worker and POMO, with ONE weight update and save.
            collector=Collector(0,self.stage,self.args.waiting_penalty,self.args.seed)
            for job in range(1,self.args.smoke_jobs+1):
                if job not in self.completed:
                    self._consume(job,collector.collect(cpustate(self.model),job))
            if self.buffer[0]:self._update(len(self.buffer[0]))
            self._save('smoke_done')
            return
        # CPU-local fallback keeps this script usable offline (and testable without Ray).
        # Kaggle's notebook installs Ray and prefers three parallel workers.
        import importlib.util
        if self.args.backend == 'local' or (self.args.backend == 'auto' and importlib.util.find_spec('ray') is None):
            print('[BACKEND] local sequential collector; Ray not used',flush=True)
            collector=Collector(0,self.stage,self.args.waiting_penalty,self.args.seed)
            for job in range(1,self.args.jobs+1):
                if job in self.completed:continue
                if self.stop or time.monotonic()-self.start_time>=self.args.session_hours*3600:
                    self._save('session_timeout_or_stop')
                    break
                self._consume(job,collector.collect(cpustate(self.model),job))
            if len(self.completed)==self.args.jobs:self._finalize()
            return
        import ray
        ray.init(num_cpus=self.args.actors,include_dashboard=False,log_to_driver=False,
                 ignore_reinit_error=True)
        @ray.remote(num_cpus=1,max_restarts=0)
        class Actor:
            def __init__(self,actor_id,stage,waiting_penalty,run_seed):
                self.worker=Collector(actor_id,stage,waiting_penalty,run_seed)
            def collect(self,weights,job):
                return self.worker.collect(weights,job)
        self.actors=[Actor.remote(i,self.stage,self.args.waiting_penalty,self.args.seed)
                     for i in range(self.args.actors)]
        todo=deque(job for job in range(1,self.args.jobs+1) if job not in self.completed)
        active={}
        try:
            for i in range(min(len(todo),len(self.actors))):
                job=todo.popleft()
                active[self.actors[i].collect.remote(cpustate(self.model),job)]=(i,job)
            while active:
                if self.stop or time.monotonic()-self.start_time>=self.args.session_hours*3600:
                    print('[SAFE STOP] Checkpointing; attach output to next Kaggle run to resume.',flush=True)
                    self._save('session_timeout_or_stop')
                    break
                ready,_=ray.wait(list(active),num_returns=1,timeout=15)
                if not ready:continue
                ref=ready[0]
                i,job=active.pop(ref)
                self._consume(job,ray.get(ref))
                if todo:
                    nxt=todo.popleft()
                    active[self.actors[i].collect.remote(cpustate(self.model),nxt)]=(i,nxt)
            if len(self.completed)==self.args.jobs:
                self._finalize()
        finally:
            for actor in self.actors:
                try:ray.kill(actor,no_restart=True)
                except Exception:pass
            ray.shutdown()


def read_best(path,stage):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('stage')!=stage:
        raise RuntimeError(f'Wrong stage in checkpoint {path}')
    model=WarmRelationalKAN() if stage=='A' else TemporalSpatialKAN()
    model.load_state_dict(ck['weights'],strict=True)
    return model.cpu().eval()


def model_collection(out):
    base=get_initial_hybrid()
    models={'OriginalHybrid_1200':base}
    for stage,name in [('A','StageA_MixedData'),('B','StageB_TemporalSpatial')]:
        p=out/f'stage_{stage.lower()}'/'best_model.pt'
        if p.exists():models[name]=read_best(p,stage)
    pub=AttentionNet(11,15,128)
    ck=torch.load(HERE/'weights'/'AttentionNet_published.pt',map_location='cpu',weights_only=True)
    if ck.get('model_name')!='AttentionNet_published':raise RuntimeError('Wrong AttentionNet weights')
    pub.load_state_dict(ck['weights'],strict=True)
    models['AttentionNet_published']=pub
    for m in models.values():m.cpu().eval()
    return models


def paired_ci(x,seed=20261109,bootstrap=3000):
    x=np.asarray(x,dtype=np.float64)
    if len(x)<2:return [None,None]
    rng=np.random.default_rng(seed)
    means=[]
    for _ in range((bootstrap+99)//100):
        idx=rng.integers(0,len(x),size=(min(100,bootstrap-len(means)),len(x)))
        means.extend(x[idx].mean(1).tolist())
    return [float(v) for v in np.quantile(means,[.025,.975])]


def evaluate_final(out,args):
    models=model_collection(out)
    print('[FINAL] models='+','.join(models),flush=True)
    sets={m:[TEST_SEED + j*1000003 + 1009*i for i in range(args.test_each)]
          for j,m in enumerate(MODES)}
    rows=[]
    with torch.inference_mode():
        for mode,seeds in sets.items():
            for i,s in enumerate(seeds):
                template=mixed_env(s,mode)
                template.init_state()
                for name,model in models.items():
                    r,_=run_episode(copy.deepcopy(template),model,Limits,torch.device('cpu'),
                                    seed=773000+i,sample=False,record=False)
                    rows.append({'dataset':f'heldout_{mode}','case':i,'seed':int(s),
                                 'model':name,**r,
                                 'penalized_makespan':float(r['makespan']) if r['done'] else 200.})
                if (i+1)%20==0:print(f'[TEST] {mode} {i+1}/{len(seeds)}',flush=True)
        # Already repeatedly inspected by our project, hence DIAGNOSTIC only.
        for i in range(args.official_cases):
            template=official_env(i)
            for name,model in models.items():
                r,_=run_episode(copy.deepcopy(template),model,Limits,torch.device('cpu'),
                                seed=773000+i,sample=False,record=False)
                rows.append({'dataset':'official_reused_DIAGNOSTIC','case':i,
                             'seed':f'env_{i}','model':name,**r,
                             'penalized_makespan':float(r['makespan']) if r['done'] else 200.})
            if (i+1)%10==0:print(f'[TEST] official {i+1}/{args.official_cases}',flush=True)
    write_csv(out/'evaluation_per_case.csv',rows)
    datasets=sorted(set(r['dataset'] for r in rows))
    summary=[]
    for ds in datasets:
        for name,model in models.items():
            arr=[r for r in rows if r['dataset']==ds and r['model']==name]
            done=[r for r in arr if r['done']]
            summary.append({'dataset':ds,'model':name,'cases':len(arr),'completed':len(done),
                'mean_makespan_completed':mean_safe(r['makespan'] for r in done),
                'penalized_makespan':mean_safe(r['penalized_makespan'] for r in arr),
                'mean_waiting':mean_safe(r['waiting_time'] for r in arr),
                'mean_runtime_s':mean_safe(r['runtime_s'] for r in arr),
                'parameters':count_parameters(model)})
    write_csv(out/'evaluation_summary.csv',summary)
    lookup={(r['dataset'],r['case'],r['model']):r for r in rows}
    pairs=[]; percase=[]
    for ds in datasets:
        cases=sorted(set(r['case'] for r in rows if r['dataset']==ds))
        for a,b in [('StageB_TemporalSpatial','StageA_MixedData'),
                    ('StageB_TemporalSpatial','AttentionNet_published'),
                    ('StageA_MixedData','OriginalHybrid_1200'),
                    ('StageA_MixedData','AttentionNet_published')]:
            if a not in models or b not in models:continue
            diffs=[]; comp=[]; aa=0;bb=0
            for i in cases:
                x=lookup[ds,i,a];y=lookup[ds,i,b]
                diff=float(x['penalized_makespan']-y['penalized_makespan'])
                diffs.append(diff);aa+=int(x['done']);bb+=int(y['done'])
                if x['done'] and y['done']:comp.append(float(x['makespan']-y['makespan']))
                percase.append({'dataset':ds,'case':i,'a':a,'b':b,
                                'a_makespan':x['makespan'],'b_makespan':y['makespan'],
                                'a_completed':x['done'],'b_completed':y['done'],
                                'delta_penalized_a_minus_b':diff})
            pairs.append({'dataset':ds,'model_a':a,'model_b':b,
                          'cases':len(cases),'a_completed':aa,'b_completed':bb,
                          'penalized_delta_a_minus_b':mean_safe(diffs),
                          'ci95_penalized_bootstrap':str(paired_ci(diffs)),
                          'completed_pair_count':len(comp),
                          'completed_delta_a_minus_b':mean_safe(comp),
                          'ci95_completed_bootstrap':str(paired_ci(comp))})
    write_csv(out/'paired_statistics.csv',pairs)
    write_csv(out/'paired_per_case.csv',percase)
    print('[EVALUATION SUMMARY]\n'+json.dumps(summary,indent=2),flush=True)


def import_resume_if_requested(out,resume):
    if not resume:return
    path=Path(resume)
    if not path.is_dir():raise FileNotFoundError(path)
    for stage in ('a','b'):
        dest=out/f'stage_{stage}'/'checkpoint_resume.pt'
        if dest.exists():continue
        # We require separate stage_a and stage_b source directories.
        files=list(path.rglob('checkpoint_resume.pt'))
        candidates=[x for x in files if x.parent.name==f'stage_{stage}']
        if len(candidates)>1:
            raise RuntimeError(f'Multiple resume candidates for stage {stage}: {candidates}')
        if candidates:
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(candidates[0],dest)
            print(f'[RESUME IMPORT] {candidates[0]} -> {dest}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='TSK_MIXED_EXPERIMENTS')
    p.add_argument('--stage',choices=('A','B','both','eval'),default='both')
    p.add_argument('--jobs',type=int,default=3600,help='per stage, equal in A and B')
    p.add_argument('--actors',type=int,default=3)
    p.add_argument('--backend',choices=('auto','ray','local'),default='auto')
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--lr-kan',type=float,default=3e-6)
    p.add_argument('--lr-attention',type=float,default=2e-5)
    p.add_argument('--waiting-penalty',type=float,default=0.,
                   help='0 preserves original reward and isolates architecture. Try 0.2 in separate matched experiments.')
    p.add_argument('--entropy-coef',type=float,default=0.001)
    p.add_argument('--clip-grad',type=float,default=10.)
    p.add_argument('--seed',type=int,default=20261109)
    p.add_argument('--save-every',type=int,default=25)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--val-every',type=int,default=200)
    p.add_argument('--val-each',type=int,default=20)
    p.add_argument('--test-each',type=int,default=100)
    p.add_argument('--official-cases',type=int,default=50)
    p.add_argument('--session-hours',type=float,default=8.0)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-jobs',type=int,default=2)
    p.add_argument('--resume-dir',default='')
    args=p.parse_args()
    if args.jobs < 1 or args.actors <1 or args.batch_size<1 or args.val_each<1:
        p.error('jobs, actors, batch-size, val-each must be positive')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    import_resume_if_requested(out,args.resume_dir)
    try:
        for stage in (('A','B') if args.stage=='both' else () if args.stage=='eval' else (args.stage,)):
            trainer=ExperimentTrainer(args,stage)
            def stop_now(sig,frame):
                trainer.stop=True
                print(f'[SIGNAL] {sig} requested graceful checkpoint',flush=True)
            signal.signal(signal.SIGINT,stop_now)
            signal.signal(signal.SIGTERM,stop_now)
            trainer.run()
            if args.smoke:continue
            if len(trainer.completed)<args.jobs:
                print('[INCOMPLETE] Only a partial stage. Not running final evaluation.',flush=True)
                return
        if not args.smoke:evaluate_final(out,args)
        else:print('[SMOKE DONE] no final benchmark was run',flush=True)
        print('[DONE] TSK experiments and/or evaluation complete',flush=True)
    except BaseException as e:
        print('[ERROR]',repr(e),flush=True)
        traceback.print_exc()
        if 'trainer' in locals():trainer._save('exception')
        raise


if __name__=='__main__':main()
