#!/usr/bin/env python3
"""Paper2: Attention vs capacity-matched No-Attention KAN, controlled two-arm ablation.

Both arms: same exact pretrained KAN V7.1 weights, identical initial action
probabilities, identical POMO job seeds and train/validation/test definitions,
same optimizer groups, same budgets, and near-equal total parameter counts.
The ONLY architectural intervention is learned relational attention vs a
parameter-matched feedforward relational pooling module.

Primary: validation-selected K1 greedy and heldout K1, per training seed.
Secondary: frozen Guided+Scarcity K4 with identical search parameters.
No previous 50 official cases are used for training, validation, or final test.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, os, random, signal, time, traceback, zipfile
from pathlib import Path
from collections import deque
import numpy as np
import torch
from warm_relational_kan import WarmRelationalKAN, count_parameters
from matched_noattention_kan import MatchedNoAttentionKAN
from train_warm_hybrid import Trainer as BaseTrainer, seed_all, cpustate, atomic_torch_save, write_csv, mean_safe
from train_temporal_spatial_experiments import MixedRewardWorker, mixed_env, MODES, evaluate_dataset, profile_for_job

HERE=Path(__file__).resolve().parent
KAN_CK=HERE/'weights'/'KAN_V71_selected.pt'
NAMES=('attention','no_attention')
VAL_BASE=1439000000
TEST_BASE=1539000000
TRAIN_BASE=1307000000
VERSION='MATCHED_ATTENTION_ABLATION_V1'


def make_model(name, seed):
    seed_all(seed)
    if name=='attention': m=WarmRelationalKAN()
    elif name=='no_attention': m=MatchedNoAttentionKAN()
    else:raise ValueError(name)
    ck=torch.load(KAN_CK,weights_only=True,map_location='cpu')
    if ck.get('model_name')!='KAN_V71_selected':
        raise RuntimeError('Invalid original KAN weights')
    m.kan.load_state_dict(ck['weights'],strict=True)
    if len([p for p in m.parameters() if p.requires_grad])==0:raise RuntimeError('All layers frozen')
    return m


def env_seeds(seed,per_mode, base):
    # arm-independent, run-seed dependent and reproducible
    return {mode:[base+1000003*j+1009*i+331*(int(seed)-20261115)
                  for i in range(per_mode)] for j,mode in enumerate(MODES)}


class Collector:
    def __init__(self, idx, variant, seed):
        torch.set_num_threads(1)
        self.idx=idx; self.variant=variant;self.seed=seed
        self.model=make_model(variant,seed).cpu().eval()

    def collect(self, weights, job):
        # SAME environment seed for the two architectures, at every job.
        envseed=TRAIN_BASE+10000019*(self.seed-20261115)+104729*int(job)
        seed_all(envseed)
        self.model.load_state_dict(weights,strict=True)
        mode=profile_for_job(job)
        worker=MixedRewardWorker(self.idx,self.model,job,envseed,mode,0.0)
        worker.work(job)
        return worker.experience,worker.perf_metrics,{'job':int(job),'mode':mode,'train_seed':envseed}


class Trainer(BaseTrainer):
    def __init__(self,args,variant,seed):
        self.args=args; self.variant=variant; self.seed=seed
        self.work=Path(args.out).resolve()/f'seed_{seed}'/variant
        self.work.mkdir(parents=True,exist_ok=True)
        self.device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
        torch.set_num_threads(1);seed_all(seed)
        self.model=make_model(variant,seed).to(self.device)
        self.initial=cpustate(self.model)
        self.opt=torch.optim.Adam([
            {'params':list(self.model.kan.parameters()),'lr':args.lr_kan,'base_lr':args.lr_kan},
            {'params':[p for n,p in self.model.named_parameters() if not n.startswith('kan.')],
             'lr':args.lr_extra,'base_lr':args.lr_extra}],eps=1e-8)
        self.completed=set();self.buffer={i:[] for i in range(7)}
        self.history=[];self.validation=[];self.best=cpustate(self.model)
        self.best_score=None;self.best_step=0;self.updates=0
        self.last_loss=float('nan');self.last_grad=float('nan');self.last_entropy=float('nan')
        self.recent=deque(maxlen=10);self.stop=False;self.start_time=time.monotonic()
        self.checkpoint_path=self.work/'checkpoint_resume.pt'
        protocol={'version':VERSION,'variant':variant,'seed':seed,'jobs':args.jobs,
                  'batch_size':args.batch_size,'lr_kan':args.lr_kan,'lr_extra':args.lr_extra,
                  'entropy_coef':args.entropy_coef,'clip_grad':args.clip_grad,
                  'save_every':args.save_every,'val_every':args.val_every,'val_each':args.val_each,
                  'train_seed_base':TRAIN_BASE,'val_seed_base':VAL_BASE,
                  'KAN_sha256':hashlib.sha256(KAN_CK.read_bytes()).hexdigest(),
                  'POMO':10,'profile':'4onehot4multihot2hard',
                  'parameter_count':count_parameters(self.model)}
        self.fingerprint=json.dumps(protocol,sort_keys=True)
        if self.checkpoint_path.exists():self._resume()
        self.actors=[]
        (self.work/'protocol.json').write_text(json.dumps(protocol,indent=2))
        print(f'[MODEL] arm={variant} seed={seed} params={count_parameters(self.model)} '
              f'initial_KAN=11473 device={self.device} completed={len(self.completed)}/{args.jobs}',flush=True)

    def _validate(self,label):
        results=[]
        for mode,seeds in env_seeds(self.seed,self.args.val_each,VAL_BASE).items():
            met,_=evaluate_dataset(self.model,mode,seeds,self.device,'validation')
            met['label']=label;met['jobs_completed']=len(self.completed)
            results.append(met)
        successes=sum(x['completed'] for x in results)
        n=sum(x['cases'] for x in results)
        penalized=sum(x['penalized_makespan']*x['cases'] for x in results)/n
        self.validation.extend(results)
        score=(int(successes),-float(penalized))
        print(f'[VAL] arm={self.variant} seed={self.seed} jobs={len(self.completed)} '
              f'success={successes}/{n} makespan={penalized:.4f} '
              + ' '.join(f"{r['mode']}:{r['penalized_makespan']:.2f}" for r in results),flush=True)
        self.model.train()
        if self.best_score is None or score>self.best_score:
            self.best=cpustate(self.model);self.best_score=score
            self.best_step=len(self.completed)
            self._write_best()

    def _write_best(self):
        atomic_torch_save({'model_name':type(self.model).__name__,'variant':self.variant,
          'seed':self.seed,'weights':self.best,'best_job':self.best_step,
          'validation_score':self.best_score,'protocol':VERSION,
          'KAN_sha256':hashlib.sha256(KAN_CK.read_bytes()).hexdigest()},self.work/'best_model.pt')
        print(f'[BEST] {self.variant} seed={self.seed} job={self.best_step}',flush=True)

    def run(self):
        if self.best_score is None:
            self._validate('initial_pretrained_KAN')
            self._save('initial')
        elif not (self.work/'best_model.pt').exists():
            self._write_best()  # restore best from resume checkpoint, not latest
        if len(self.completed)==self.args.jobs:
            print(f'[SKIP] {self.variant} seed={self.seed} training complete',flush=True)
            return True
        if self.args.smoke:
            collector=Collector(0,self.variant,self.seed)
            # true Worker + POMO update, but no final test or promotion
            for job in range(1,min(self.args.smoke_jobs,self.args.jobs)+1):
                if job not in self.completed:
                    self._consume(job,collector.collect(cpustate(self.model),job))
            if self.buffer[0]:self._update(len(self.buffer[0]))
            self._save('smoke_done')
            print(f'[SMOKE PASS] {self.variant} seed={self.seed} params={count_parameters(self.model)}',flush=True)
            return False
        if self.args.backend=='local':
            collector=Collector(0,self.variant,self.seed)
            for job in range(1,self.args.jobs+1):
                if job in self.completed:continue
                if self.stop or time.monotonic()-self.start_time>self.args.session_hours*3600:
                    self._save('session_stop');return False
                self._consume(job,collector.collect(cpustate(self.model),job))
        else:
            try:
                import ray
            except ImportError:
                print('[BACKEND] ray unavailable; falling back to local',flush=True)
                self.args.backend='local'
                return self.run()
            ray.init(num_cpus=self.args.actors,include_dashboard=False,
                     log_to_driver=False,ignore_reinit_error=True)
            @ray.remote(num_cpus=1,max_restarts=0)
            class Actor:
                def __init__(self,idx,variant,seed):self.worker=Collector(idx,variant,seed)
                def collect(self,weights,job):return self.worker.collect(weights,job)
            workers=[Actor.remote(i,self.variant,self.seed) for i in range(self.args.actors)]
            remaining=[j for j in range(1,self.args.jobs+1) if j not in self.completed]
            try:
                # Synchronous waves. Consume in sorted job order regardless of
                # asynchronous Ray scheduling, supporting seed-matched arms.
                for off in range(0,len(remaining),len(workers)):
                    if self.stop or time.monotonic()-self.start_time>self.args.session_hours*3600:
                        self._save('session_stop');return False
                    group=remaining[off:off+len(workers)]
                    state=cpustate(self.model)
                    results=ray.get([workers[i].collect.remote(state,job)
                                     for i,job in enumerate(group)])
                    for job,result in zip(group,results):self._consume(job,result)
            finally:
                for w in workers:
                    try:ray.kill(w,no_restart=True)
                    except Exception:pass
                ray.shutdown()
        if len(self.completed)==self.args.jobs:
            if self.buffer[0]:self._update(len(self.buffer[0]))
            self._validate('after_all_updates')
            self._save('training_complete')
            print(f'[DONE] {self.variant} seed={self.seed} '
                  f'jobs={len(self.completed)} updates={self.updates} '
                  f'selected_best_job={self.best_step}',flush=True)
            return True
        return False


def paired_bootstrap(rows,method,model_a,model_b,n_boot=5000):
    from collections import defaultdict
    by=defaultdict(dict)
    for r in rows:
        if r['method']==method and r['variant'] in (model_a,model_b):
            by[(r['seed'],r['mode'],r['case'])][r['variant']]=float(r['penalized_makespan'])
    pairs={m:[] for m in MODES}
    for (s,m,i),record in by.items():
        if set(record)=={model_a,model_b}:
            pairs[m].append(record[model_a]-record[model_b])
    # Resample cases within each (mode, training seed) to protect against
    # mode imbalance; repeated seeds can share test cases and remain dependent.
    if not all(len(pairs[m]) for m in MODES):return {}
    vals=np.concatenate([pairs[m] for m in MODES]); rng=np.random.default_rng(156789)
    out=np.empty(n_boot)
    for j in range(n_boot):
        out[j]=np.mean(np.concatenate([rng.choice(v,len(v),replace=True) for v in pairs.values()]))
    return {'method':method,'model_a':model_a,'model_b':model_b,
            'n_paired_model_seed_cases':len(vals),'mean_delta_a_minus_b':float(np.mean(vals)),
            'ci95_lower':float(np.quantile(out,.025)),'ci95_upper':float(np.quantile(out,.975)),
            'note':'exploratory case bootstrap; does not capture independent training-seed uncertainty'}


def evaluate(args):
    import train_selective_distillation as sd
    import guided_coalition_search as g
    out=Path(args.out); f=out/'evaluation_per_case.csv'
    rows=[]
    if f.exists():
        with f.open(encoding='utf-8-sig',newline='') as fp:rows=list(csv.DictReader(fp))
    existing={(int(r['seed']),r['mode'],int(r['case']),r['variant']) for r in rows}
    for seed in args.seeds:
        for variant in NAMES:
            path=out/f'seed_{seed}'/variant/'best_model.pt'
            cp=torch.load(path,map_location='cpu',weights_only=True)
            if cp['seed']!=seed or cp['variant']!=variant or cp['protocol']!=VERSION:
                raise RuntimeError('Incompatible best checkpoint')
            model=make_model(variant,seed)
            model.load_state_dict(cp['weights'],strict=True);model.cpu().eval()
            for j,mode in enumerate(MODES):
                for case in range(args.test_each):
                    if (seed,mode,case,variant) in existing:continue
                    case_seed=TEST_BASE+1000003*j+1009*case
                    env=mixed_env(case_seed,mode);env.init_state()
                    # Same four teacher-independent search candidates as Paper 2:
                    # guided greedy, guided sample, scarcity greedy, scarcity sample.
                    start=time.perf_counter()
                    chosen,baseline,candidates=sd.choose_candidates(
                        env,model,case_seed+99331,1.1,0.8)
                    runtime=time.perf_counter()-start
                    assert len(candidates)==4
                    record=[]
                    for method,candidate,search_time,n_rollouts in (
                        ('guided_K1',baseline[1],float(baseline[1]['result']['runtime_s']),1),
                        ('guided_scarcity_K4',chosen[1],runtime,4)):
                        g.replay_and_verify(env,candidate)
                        r=candidate['result'];complete=bool(r['done'])
                        record.append({'seed':seed,'mode':mode,'case':case,'case_seed':case_seed,
                           'variant':variant,'method':method,'K':n_rollouts,
                           'success':int(complete),'makespan':float(r['makespan']),
                           'penalized_makespan':float(r['makespan']) if complete else 200.,
                           'waiting':float(r['waiting_time']),'search_s':search_time,
                           'replay_verified':1,'selected_best_job':cp['best_job'],
                           'parameters':count_parameters(model)})
                    rows.extend(record)
                    write_csv(f,rows)
                    if (case+1)%10==0:
                        print(f'[TEST] seed={seed} arm={variant} mode={mode} '
                              f'{case+1}/{args.test_each}',flush=True)
    summary=[]
    for seed in args.seeds:
        for variant in NAMES:
            for method in ('guided_K1','guided_scarcity_K4'):
                for mode in ('ALL',)+MODES:
                    sel=[x for x in rows if int(x['seed'])==seed and x['variant']==variant
                         and x['method']==method and (mode=='ALL' or x['mode']==mode)]
                    if len(sel)!=args.test_each*(3 if mode=='ALL' else 1):
                        raise RuntimeError(f'Incomplete results {seed}/{variant}/{method}/{mode}')
                    summary.append({'seed':seed,'variant':variant,'method':method,'mode':mode,
                          'cases':len(sel),'success':sum(int(x['success']) for x in sel),
                          'mean_penalized_makespan':mean_safe(float(x['penalized_makespan']) for x in sel),
                          'mean_waiting':mean_safe(float(x['waiting']) for x in sel),
                          'mean_search_s':mean_safe(float(x['search_s']) for x in sel)})
    write_csv(out/'evaluation_summary.csv',summary)
    pairs=[paired_bootstrap(rows,method,'attention','no_attention')
           for method in ('guided_K1','guided_scarcity_K4')]
    write_csv(out/'paired_statistics.csv',pairs)
    print('[RESULT] Attention vs matched NoAttention (lower makespan is better)',flush=True)
    for row in pairs:print(row,flush=True)
    print('[DONE] Ablation heldout completed',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='PAPER2_ATTENTION_ABLATION')
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261115,20261116,20261117])
    parser.add_argument('--jobs',type=int,default=1200)
    parser.add_argument('--backend',choices=['local','ray'],default='ray')
    parser.add_argument('--actors',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--lr-kan',type=float,default=2e-5)
    parser.add_argument('--lr-extra',type=float,default=1e-4)
    parser.add_argument('--entropy-coef',type=float,default=.001)
    parser.add_argument('--clip-grad',type=float,default=10.)
    parser.add_argument('--save-every',type=int,default=25)
    parser.add_argument('--val-every',type=int,default=100)
    parser.add_argument('--val-each',type=int,default=12)
    parser.add_argument('--test-each',type=int,default=30)
    parser.add_argument('--session-hours',type=float,default=8.)
    parser.add_argument('--log-every',type=int,default=10)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-jobs',type=int,default=2)
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):raise ValueError('Duplicate seeds')
    if args.jobs<1 or args.test_each<1:raise ValueError('Invalid budgets')
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    old=root/'experiment_lock.json'
    protocol={'version':VERSION,'seeds':args.seeds,'jobs':args.jobs,
              'batch_size':args.batch_size,'lr_kan':args.lr_kan,'lr_extra':args.lr_extra,
              'val_each':args.val_each,'test_each':args.test_each,
              'source_kan_hash':hashlib.sha256(KAN_CK.read_bytes()).hexdigest(),
              'variants':list(NAMES),'source':'KAN V7.1 selected pretrained checkpoint'}
    if old.exists() and json.loads(old.read_text())!=protocol:
        raise RuntimeError('Locked experiment differs from previous run; choose a NEW --out')
    old.write_text(json.dumps(protocol,indent=2))
    for seed in args.seeds:
        x,y=make_model('attention',seed),make_model('no_attention',seed)
        assert count_parameters(x)==25138 and abs(count_parameters(x)-count_parameters(y))<=.005*count_parameters(x)
        e=mixed_env(1777,'onehot');e.init_state();t,a,m=e.agent_observe(0)
        with torch.no_grad():
            inp=(torch.tensor(t,dtype=torch.float32),torch.tensor(a,dtype=torch.float32),
                 torch.tensor(m),torch.tensor([[[0]]]))
            delta=float((x(*inp)[0]-y(*inp)[0]).abs().max())
        if delta>1e-6:raise RuntimeError(f'Initial policies differ: {delta}')
        print(f'[PARITY] seed={seed} params={count_parameters(x)}/{count_parameters(y)} '
              f'initial_action_prob_delta={delta:.2g}',flush=True)
    for seed in args.seeds:
        for variant in NAMES:
            trainer=Trainer(args,variant,seed)
            signal.signal(signal.SIGTERM,lambda *_:setattr(trainer,'stop',True))
            try:
                completed=trainer.run()
            except BaseException:
                traceback.print_exc();trainer._save('exception');raise
            if args.smoke:continue
            if not completed:
                print('[STOP] Partial checkpoints saved, rerun to resume',flush=True)
                return
    if not args.smoke:evaluate(args)
    else:print('[DONE] SMOKE passed; change SMOKE=False for full independent training',flush=True)

if __name__=='__main__':main()
