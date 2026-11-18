#!/usr/bin/env python3
"""Resumable POMO REINFORCE warm fine-tuning on the ORIGINAL HeteroMRTA TaskEnv.

Trainable: pretrained small KAN (11,473 parameters) + NEW relational attention.
Published AttentionNet is a frozen BENCHMARK ONLY, never a teacher or trainer.
Original Worker.work collects 10 POMO rollouts per randomized task problem.

Changes vs authors' historical trainer: smaller GPU updates, finite budget,
shorter held-out validation, custom hybrid architecture and checkpoint scheme.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, os, random, signal, sys, time, traceback
from collections import deque
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import numpy as np
import torch
from torch.distributions import Categorical

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from kan_policy import KANPolicy
from warm_relational_kan import WarmRelationalKAN, count_parameters, verify_exact_warmstart
from benchmark_core import run_episode, OfficialTestSetUnpickler
from env.task_env import TaskEnv
from worker import Worker
from parameters import EnvParams, TrainParams, SaverParams
from attention import AttentionNet

TrainParams.POMO_SIZE=10
TrainParams.USE_GPU=False     # rollout actors evaluate on CPU
TrainParams.NUM_GPU=0
SaverParams.SAVE_IMG=False

TRAIN_SEED_START=616060000
VAL_SEEDS=[515000000+17389*i for i in range(24)]
TEST_SEEDS=[935000000+9973*i for i in range(100)]


class Limits:
    max_time=200.0
    max_actions=400
    max_decision_rounds=500


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32-1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def cpustate(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def atomic_torch_save(obj, path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,temp)
    os.replace(temp,path)


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader();w.writerows(rows)


def mean_safe(seq):
    x=np.asarray(list(seq),dtype=np.float64)
    x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float('nan')


def make_env(seed,params=None):
    if params is None:params=((3,3),(5,5),(20,20))
    return TaskEnv(*params, 5, 30, seed=int(seed), plot_figure=False)


def official_env(case):
    p=HERE/'RALTestSet'/f'env_{case}.pkl'
    with p.open('rb') as f:env=OfficialTestSetUnpickler(f,TaskEnv).load()
    env.init_state();env.plot_figure=False
    return env


def eval_policy(model,device,seeds,caption):
    model.eval()
    rows=[]
    with torch.inference_mode():
        for i,seed in enumerate(seeds):
            env=make_env(seed)
            env.init_state()
            r,_=run_episode(env,model,Limits,device,seed=20260900+i,sample=False,record=False)
            rows.append(r)
    completed=sum(int(r['done']) for r in rows)
    metric={'tag':caption,'cases':len(rows),'done':completed,
            'reward':mean_safe(r['native_reward'] for r in rows),
            'makespan_completed':mean_safe(r['makespan'] for r in rows if r['done']),
            'penalized_makespan':mean_safe(r['makespan'] if r['done'] else 200. for r in rows),
            'waiting':mean_safe(r['waiting_time'] for r in rows)}
    print(f'[EVAL] {caption} | success={completed}/{len(rows)} '
          f'| reward={metric["reward"]:.4f} | makespan_completed={metric["makespan_completed"]:.4f} '
          f'| penalized={metric["penalized_makespan"]:.4f}',flush=True)
    model.train()
    return metric,rows


def checkpoint_validation(score):
    # Completion first, then reward; avoids selecting a failure-prone model.
    return (int(score['done']),float(score['reward']))


class RewardWorker(Worker):
    """No change to transitions/rewards; expose the author's terminal reward in logs."""
    def run_episode(self,*args,**kwargs):
        reward,buffer,metrics=super().run_episode(*args,**kwargs)
        metrics['terminal_reward']=[float(reward)]
        return reward,buffer,metrics


class LocalCollector:
    def __init__(self,actor_id):
        torch.set_num_threads(1)
        seed_all(700000+actor_id)
        self.actor_id=actor_id
        self.model=WarmRelationalKAN().cpu().eval()

    def collect(self,weights,job,params):
        seed=TRAIN_SEED_START + 104729*int(job)
        seed_all(seed)
        self.model.load_state_dict(weights,strict=True)
        wk=RewardWorker(self.actor_id,self.model,self.model,job,device='cpu',
                        save_image=False,seed=seed,env_params=params)
        wk.work(job)
        return wk.experience,wk.perf_metrics,{'job':job,'actor_id':self.actor_id}


class Trainer:
    def __init__(self,args):
        self.args=args
        self.work=Path(args.out).resolve()
        self.work.mkdir(parents=True,exist_ok=True)
        self.device=torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
        torch.set_num_threads(1)
        seed_all(args.seed)
        self.kan=KANPolicy(hidden=24,grid=5)
        initial=torch.load(HERE/'weights'/'KAN_V71_selected.pt',map_location='cpu',weights_only=True)
        if initial.get('model_name')!='KAN_V71_selected':
            raise RuntimeError('Wrong warmstart identity')
        self.kan.load_state_dict(initial['weights'],strict=True)
        self.original=cpustate(self.kan)
        self.model=WarmRelationalKAN().to(self.device)
        self.model.kan.load_state_dict(self.original,strict=True)
        # Verify predictions and weights BEFORE optimizer is ever stepped.
        test_env=make_env(1777);test_env.init_state()
        task,agents,mask=test_env.agent_observe(0)
        t=torch.tensor(task,dtype=torch.float32,device=self.device)
        a=torch.tensor(agents,dtype=torch.float32,device=self.device)
        m=torch.tensor(mask,device=self.device)
        ix=torch.tensor([[[0]]],device=self.device)
        warm_err=verify_exact_warmstart(self.kan.to(self.device),self.model,t,a,m,ix)
        assert self.model._initial_kan_params==11473
        assert all(p.requires_grad for p in self.model.parameters())
        print(f'[WARMSTART] KAN 11,473 parameters copied STRICTLY | action maximum delta={warm_err:.9g}',flush=True)
        print(f'[MODEL] hybrid_parameters={count_parameters(self.model)} attention_adapter='
              f'{count_parameters(self.model)-11473} | kan_frozen=False | cuda={self.device}',flush=True)
        attn_params=[p for name,p in self.model.named_parameters() if not name.startswith('kan.')]
        self.opt=torch.optim.Adam([
            {'params':list(self.model.kan.parameters()),'lr':args.lr_kan,'base_lr':args.lr_kan},
            {'params':attn_params,'lr':args.lr_attention,'base_lr':args.lr_attention}],eps=1e-8)
        self.completed=set()
        self.buffer={i:[] for i in range(7)}
        self.history=[]
        self.validation=[]
        self.best=None
        self.best_score=None
        self.best_step=0
        self.updates=0
        self.last_loss=float('nan')
        self.last_grad=float('nan')
        self.last_entropy=float('nan')
        self.recent=deque(maxlen=10)
        self.stop=False
        self.start_time=time.monotonic()
        self.checkpoint_path=self.work/'checkpoint_resume.pt'
        self.fingerprint=json.dumps({'seed':args.seed,'jobs':args.jobs,'pomo':10,
                                      'batch_size':args.batch_size,'lr_kan':args.lr_kan,
                                      'lr_attention':args.lr_attention,'d':32,'heads':4,
                                      'train_seed':TRAIN_SEED_START},sort_keys=True)
        if self.checkpoint_path.exists():
            self._resume()
        else:
            # initial best model is the old KAN with identity-initialized adapter
            self.best=cpustate(self.model)
        self.actors=[]

    def _resume(self):
        ck=torch.load(self.checkpoint_path,map_location='cpu',weights_only=False)
        if ck['fingerprint']!=self.fingerprint:raise RuntimeError('Incompatible checkpoint: training configuration changed')
        self.model.load_state_dict(ck['model'],strict=True)
        self.opt.load_state_dict(ck['optimizer'])
        for state in self.opt.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(self.device)
        self.completed=set(ck['completed'])
        self.buffer=ck['buffer']
        self.history=ck['history']
        self.validation=ck['validation']
        self.best=ck['best']
        self.best_score=tuple(ck['best_score']) if ck['best_score'] is not None else None
        self.best_step=int(ck['best_step'])
        self.updates=int(ck['updates'])
        if 'rng_python' in ck:random.setstate(ck['rng_python'])
        if 'rng_numpy' in ck:np.random.set_state(ck['rng_numpy'])
        if 'rng_torch' in ck:torch.set_rng_state(ck['rng_torch'])
        print(f'[RESUME] continuing from {len(self.completed)}/{self.args.jobs} COMPLETED jobs, '
              f'optimizer updates={self.updates}',flush=True)

    def _save(self,why):
        cp={'fingerprint':self.fingerprint,'model':cpustate(self.model),'optimizer':self.opt.state_dict(),
            'completed':sorted(self.completed),'buffer':self.buffer,'history':self.history,
            'validation':self.validation,'best':self.best,'best_score':self.best_score,
            'best_step':self.best_step,'updates':self.updates,
            'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
            'rng_torch':torch.get_rng_state(),'reason':why}
        atomic_torch_save(cp,self.checkpoint_path)
        write_csv(self.work/'train_metrics.csv',self.history)
        write_csv(self.work/'validation_metrics.csv',self.validation)
        print(f'[CHECKPOINT] {why} | done={len(self.completed)}/{self.args.jobs} '
              f'| size={self.checkpoint_path.stat().st_size:,} bytes',flush=True)

    def _update(self,batch):
        sliced={k:self.buffer[k][:batch] for k in self.buffer}
        for k in self.buffer:del self.buffer[k][:batch]
        agents=torch.stack(sliced[0],dim=0).to(self.device)
        tasks=torch.stack(sliced[1],dim=0).to(self.device)
        actions=torch.stack(sliced[2],dim=0).reshape(-1).long().to(self.device)
        mask=torch.stack(sliced[3],dim=0).to(self.device)
        index=torch.stack(sliced[5],dim=0).to(self.device)
        adv=torch.stack(sliced[6],dim=0).reshape(-1).to(self.device)
        # Group-centered REINFORCE advantages from original Worker; scale only.
        std=adv.std(unbiased=False).clamp_min(1.)
        adv=(adv/std).clamp(-5,5)
        self.model.train()
        probs,logprobs=self.model(tasks,agents,mask,index)
        chosen=logprobs.gather(1,actions.unsqueeze(-1)).squeeze(-1)
        entropy=-(probs*logprobs).sum(-1).mean()
        loss=-(chosen*adv).mean() - self.args.entropy_coef*entropy
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.args.clip_grad)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        self.opt.step()
        self.updates+=1
        self.last_loss=float(loss.detach().cpu())
        self.last_grad=float(grad.detach().cpu())
        self.last_entropy=float(entropy.detach().cpu())

    def _phase(self):
        fraction=len(self.completed)/self.args.jobs
        if fraction < .33:return 1,1.
        if fraction < .75:return 2,.7
        return 3,.4

    def _adjust_lr(self):
        stage,scale=self._phase()
        for group in self.opt.param_groups:
            group['lr']=group['base_lr']*scale
        return stage

    def _collect_params(self,job):
        # Same ranges as the original authors, per-instance randomized sizes.
        g=np.random.default_rng(TRAIN_SEED_START+104729*job+42)
        agents=int(g.integers(3,4))
        species=int(g.integers(3,6))
        tasks=int(g.integers(15,51))
        return [(agents,agents),(species,species),(tasks,tasks)]

    def _consume(self,job,result):
        replay,metrics,info=result
        if job in self.completed:raise RuntimeError('Duplicate job')
        self.completed.add(job)
        for key in self.buffer:self.buffer[key].extend(replay[key])
        self.recent.append({k:mean_safe(metrics.get(k,[])) for k in
                            ('terminal_reward','makespan','success_rate','waiting_time')})
        self._adjust_lr()
        while len(self.buffer[0])>=self.args.batch_size:
            self._update(self.args.batch_size)
        done=len(self.completed)
        if done%self.args.log_every==0 or done==self.args.jobs:
            averages={k:mean_safe(v[k] for v in self.recent) for k in self.recent[0]}
            row={'jobs_completed':done,'total_jobs':self.args.jobs,'pomo_attempts':10*done,
                 'stage':self._phase()[0],'reward':averages['terminal_reward'],
                 'makespan':averages['makespan'],'success_rate':averages['success_rate'],
                 'waiting':averages['waiting_time'],'loss':self.last_loss,
                 'gradient_norm':self.last_grad,'entropy':self.last_entropy,
                 'lr_kan':self.opt.param_groups[0]['lr'],
                 'lr_attention':self.opt.param_groups[1]['lr'],'updates':self.updates,
                 'elapsed_s':round(time.monotonic()-self.start_time,1)}
            self.history.append(row)
            print(f'[TRAIN] Stage={row["stage"]}/3 | Jobs={done}/{self.args.jobs} | '
                  f'POMO={10*done}/{10*self.args.jobs} | Reward={row["reward"]:.4f} | '
                  f'Makespan={row["makespan"]:.4f} | Success={row["success_rate"]:.2%} | '
                  f'Waiting={row["waiting"]:.4f} | Loss={self.last_loss:.5f} | '
                  f'KAN_LR={row["lr_kan"]:.2e} | ATT_LR={row["lr_attention"]:.2e}',flush=True)
        if done%self.args.save_every==0:
            self._save('periodic')
        if done%self.args.val_every==0 or done==self.args.jobs:
            self._validate(f'after_{done}_jobs')
            self._save('post_validation')

    def _validate(self,label):
        out,_=eval_policy(self.model,self.device,VAL_SEEDS,label)
        out['jobs_completed']=len(self.completed)
        self.validation.append(out)
        score=checkpoint_validation(out)
        if self.best_score is None or score>self.best_score:
            self.best=cpustate(self.model)
            self.best_score=score
            self.best_step=len(self.completed)
            atomic_torch_save({'model_name':'WarmRelationalKAN',
                               'weights':self.best,'best_job':self.best_step,
                               'score':self.best_score},self.work/'best_hybrid.pt')
            print(f'[BEST] snapshot at {self.best_step} jobs | '
                  f'success={score[0]}/{len(VAL_SEEDS)} reward={score[1]:.4f}',flush=True)

    def _test_models(self):
        # Final checkpoint-selected hybrid vs unmodified published AttentionNet,
        # plus original KAN; NO gradients, same TaskEnv instances/action mask.
        models={
            'WarmRelationalKAN_best':WarmRelationalKAN(),
            'KAN_V71_pretrained':KANPolicy(hidden=24,grid=5),
            'AttentionNet_published':AttentionNet(11,15,128)}
        models['WarmRelationalKAN_best'].load_state_dict(self.best,strict=True)
        models['KAN_V71_pretrained'].load_state_dict(self.original,strict=True)
        blob=torch.load(HERE/'weights'/'AttentionNet_published.pt',map_location='cpu',weights_only=True)
        if blob.get('model_name')!='AttentionNet_published':
            raise RuntimeError('Expected ORIGINAL published AttentionNet checkpoint')
        models['AttentionNet_published'].load_state_dict(blob['weights'],strict=True)
        # Equal hardware mode for inference times (single CPU thread).
        for p in models.values():p.cpu().eval()
        self.model.cpu()
        rows=[]
        for dataset,n in [('generated_test',self.args.eval_new_cases),('official',self.args.eval_official_cases)]:
            print(f'[TEST] {dataset}: {n} paired problems * 3 models',flush=True)
            for i in range(n):
                template=make_env(TEST_SEEDS[i]) if dataset=='generated_test' else official_env(i)
                template.init_state()
                for name,model in models.items():
                    with torch.inference_mode():
                        r,_=run_episode(copy.deepcopy(template),model,Limits,torch.device('cpu'),
                                        seed=773000+i,sample=False,record=False)
                    rows.append({'dataset':dataset,'case':i,'seed':int(TEST_SEEDS[i]) if dataset=='generated_test' else f'env_{i}',
                                 'model':name,**r,
                                 'penalized_makespan':r['makespan'] if r['done'] else 200.})
                if (i+1)%10==0:print(f'[TEST] {dataset} {i+1}/{n}',flush=True)
        write_csv(self.work/'evaluation_per_case.csv',rows)
        summaries=[]
        for dataset in ['generated_test','official']:
            for name in models:
                rr=[r for r in rows if r['dataset']==dataset and r['model']==name]
                done=[r for r in rr if r['done']]
                summaries.append({'dataset':dataset,'model':name,'cases':len(rr),'completed':len(done),
                                  'mean_makespan_completed':mean_safe(r['makespan'] for r in done),
                                  'penalized_makespan':mean_safe(r['penalized_makespan'] for r in rr),
                                  'mean_native_reward':mean_safe(r['native_reward'] for r in rr),
                                  'mean_waiting':mean_safe(r['waiting_time'] for r in rr),
                                  'mean_runtime':mean_safe(r['runtime_s'] for r in rr),
                                  'parameters':count_parameters(models[name])})
        write_csv(self.work/'evaluation_summary.csv',summaries)
        pairs=[];pair_rows=[]
        by={(r['dataset'],r['case'],r['model']):r for r in rows}
        for dataset,n in [('generated_test',self.args.eval_new_cases),('official',self.args.eval_official_cases)]:
            for a,b in [('WarmRelationalKAN_best','KAN_V71_pretrained'),
                        ('WarmRelationalKAN_best','AttentionNet_published')]:
                all_d=[];complete_d=[]
                for i in range(n):
                    x,y=by[(dataset,i,a)],by[(dataset,i,b)]
                    diff=float(x['penalized_makespan']-y['penalized_makespan'])
                    all_d.append(diff)
                    both=bool(x['done'] and y['done'])
                    if both:complete_d.append(float(x['makespan']-y['makespan']))
                    pair_rows.append({'dataset':dataset,'case':i,'model_a':a,'model_b':b,
                                      'a_done':x['done'],'b_done':y['done'],
                                      'penalized_delta_a_minus_b':diff,
                                      'both_completed_delta_a_minus_b':
                                          float(x['makespan']-y['makespan']) if both else ''})
                def ci(values):
                    if len(values)<2:return [None,None]
                    v=np.asarray(values,dtype=float)
                    rng=np.random.default_rng(20261108)
                    boots=np.empty(2000)
                    for j in range(0,2000,100):
                        picks=rng.integers(0,len(v),size=(100,len(v)))
                        boots[j:j+100]=v[picks].mean(axis=1)
                    return [float(q) for q in np.quantile(boots,[0.025,0.975])]
                pairs.append({'dataset':dataset,'model_a':a,'model_b':b,'cases':n,
                              'both_completed':len(complete_d),
                              'mean_penalized_delta':mean_safe(all_d),'ci95_penalized':str(ci(all_d)),
                              'mean_completed_delta':mean_safe(complete_d),
                              'ci95_completed':str(ci(complete_d)),
                              'note':'A - B; negative favors A; failures penalized at 200'})
        write_csv(self.work/'paired_statistics.csv',pairs)
        write_csv(self.work/'paired_per_case.csv',pair_rows)
        print('[FINAL SUMMARY]',json.dumps(summaries,indent=2),flush=True)
        self.model.to(self.device)

    def run(self):
        if self.best_score is None:
            self._validate('warm_start_initial')
            self._save('initial')
        if self.args.smoke:
            # No Ray dependency; runs one job using original Worker/POMO and updates.
            self.args.jobs=1
            collector=LocalCollector(0)
            job=1
            result=collector.collect(cpustate(self.model),job,self._collect_params(job))
            self._consume(job,result)
            self._save('smoke_done')
            return
        import ray
        ray.init(num_cpus=self.args.actors,include_dashboard=False,log_to_driver=False,
                 ignore_reinit_error=True)
        @ray.remote(num_cpus=1, max_restarts=0)
        class Actor:
            def __init__(self,actor_id):
                # Each actor imports the same work directory, pinned original source.
                self.local=LocalCollector(actor_id)
            def collect(self,weights,job,params):
                return self.local.collect(weights,job,params)
        self.actors=[Actor.remote(i) for i in range(self.args.actors)]
        todo=deque(job for job in range(1,self.args.jobs+1) if job not in self.completed)
        active={}
        try:
            for i in range(min(len(todo),len(self.actors))):
                job=todo.popleft()
                active[self.actors[i].collect.remote(cpustate(self.model),job,self._collect_params(job))]=(i,job)
            while active:
                if self.stop or time.monotonic()-self.start_time>=self.args.session_hours*3600:
                    print('[SAFE STOP] Will checkpoint completed jobs; pending jobs will be rerun on resume',flush=True)
                    self._save('session_timeout_or_stop')
                    break
                ready,_=ray.wait(list(active),num_returns=1,timeout=15)
                if not ready:continue
                ref=ready[0]
                i,job=active.pop(ref)
                result=ray.get(ref)
                self._consume(job,result)
                if todo:
                    nxt=todo.popleft()
                    active[self.actors[i].collect.remote(cpustate(self.model),nxt,self._collect_params(nxt))]=(i,nxt)
            if len(self.completed)==self.args.jobs:
                # Some choices may remain in replay; flush them to avoid wasting data.
                if self.buffer[0]:self._update(len(self.buffer[0]))
                self._validate('after_all_updates')
                kan_change=sum(float((v.detach().cpu()-self.original[k]).abs().sum())
                               for k,v in self.model.kan.state_dict().items())
                adapter_change=float(self.model.adapter.weight.detach().abs().sum().cpu())
                print(f'[TRAINABLE VERIFIED] KAN weight L1 change={kan_change:.6f} '
                      f'| adapter nonzero L1={adapter_change:.6f}',flush=True)
                (self.work/'training_provenance.json').write_text(json.dumps({
                    'warmstart':'KAN V7.1 selected exact weights',
                    'published_attention':'evaluation_only',
                    'trained_parameters':count_parameters(self.model),
                    'old_kan_parameter_count':11473,
                    'old_kan_weight_change_L1':kan_change,
                    'learned_residual_adapter_weight_L1':adapter_change,
                    'optimizer_updates':self.updates,
                    'training_completed_jobs':len(self.completed),
                    'training_pomo_attempts':len(self.completed)*10,
                    'selected_validation_job':self.best_step},indent=2))
                self._save('training_complete')
                self._test_models()
                self._save('evaluation_complete')
                print('[DONE] Training+evaluation completed',flush=True)
        finally:
            for actor in self.actors:
                try:ray.kill(actor,no_restart=True)
                except Exception:pass
            ray.shutdown()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='WARM_RELATIONAL_KAN_RUN')
    p.add_argument('--jobs',type=int,default=1200)
    p.add_argument('--actors',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--lr-kan',type=float,default=2e-5)
    p.add_argument('--lr-attention',type=float,default=1e-4)
    p.add_argument('--clip-grad',type=float,default=10.)
    p.add_argument('--entropy-coef',type=float,default=0.001)
    p.add_argument('--seed',type=int,default=20261108)
    p.add_argument('--save-every',type=int,default=25)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--val-every',type=int,default=100)
    p.add_argument('--session-hours',type=float,default=8.5)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--eval-new-cases',type=int,default=100)
    p.add_argument('--eval-official-cases',type=int,default=50)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    assert a.jobs>0 and a.actors>0 and a.batch_size>=1
    trainer=Trainer(a)
    def stop_req(sig,frame):
        trainer.stop=True
        print(f'[SIGNAL] Request graceful shutdown via {sig}',flush=True)
    signal.signal(signal.SIGTERM,stop_req)
    signal.signal(signal.SIGINT,stop_req)
    try:trainer.run()
    except BaseException as error:
        print('[ERROR]',repr(error),flush=True)
        traceback.print_exc()
        trainer._save('exception')
        raise


if __name__=='__main__':main()
