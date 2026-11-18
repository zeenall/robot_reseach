#!/usr/bin/env python3
"""Paper 2: advantage-weighted selective distillation with KL trust-region.

Frozen Stage A backbone; train only adapter + edge_bias + KAN2 (5,498/25,138).
Three controls with identical instances/budget: self-BC+KL, teacher uniform+KL,
teacher advantage-weighted+KL. Both best guided-K1 and best guided/scarcity-K4
weights are saved using disjoint validation. Final test never selects checkpoints.

Teacher labels come from strictly replay-verified full schedule simulations; no
claims of online planning. Search K=4 counts complete simulated trajectories.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

import train_selective_distillation as old

HERE=Path(__file__).resolve().parent
MODES=old.MODES
MODEL_SEED=20261110
VAL_SEED=2606110000     # new validation distribution, not from experiments 23-29
TEST_SEED=2706110000    # disjoint heldout test, used only after training is done
ARMS=('self_kl','teacher_uniform_kl','teacher_adv_kl')


def notify(msg):
    print(msg,flush=True)


def args_signature(args,seed,arm):
    return {'method':'advantage_kl_v1','seed':int(seed),'arm':arm,
            'model_sha':old.sha256(HERE/'reference'/'stage_a_best_model.pt'),
            'teacher_cases':args.teacher_cases,'epochs':args.epochs,
            'updates_per_epoch':args.updates_per_epoch,'batch_size':args.batch_size,
            'val_each':args.val_each,'val_seed':VAL_SEED,
            'lr_adapter':args.lr_adapter,'lr_edge':args.lr_edge,'lr_kan2':args.lr_kan2,
            'kl_weight':args.kl_weight,'adv_floor':args.adv_floor,
            'adv_temperature':args.adv_temperature,'adv_cap':args.adv_cap,
            'demo_seed':args.seed}


def get_weights(improvement,arm,args):
    if arm!='teacher_adv_kl':
        return 1.0
    if not math.isfinite(float(improvement)):
        raise ValueError('Nonfinite teacher advantage')
    # Best-of-K must not be worse than the matched guided-greedy candidate.
    if improvement < -1e-5:
        raise AssertionError(f'Best candidate worse than included greedy baseline: {improvement}')
    return float(np.clip(args.adv_floor+max(0.0,float(improvement))/args.adv_temperature,
                         args.adv_floor,args.adv_cap))


def weighted_batches(files,batch_size,epoch_seed,arm,args,pack=8):
    """Shuffle bounded-memory episode files; attach episode improvement to each action."""
    rng=np.random.default_rng(epoch_seed)
    files=list(files)
    rng.shuffle(files)
    bucket=[]
    keys=('tasks','agents','mask','agent_id','action')
    for path in files:
        with np.load(path,allow_pickle=False) as f:
            data={k:np.array(f[k],copy=True) for k in keys}
            meta=json.loads(str(f['meta_json'].item()))
        n=len(data['action'])
        if n==0:
            raise AssertionError(f'No actions in {path}')
        data['adv_weight']=np.full(n,get_weights(float(meta['improvement']),arm,args),dtype=np.float32)
        bucket.append(data)
        if len(bucket)<pack:
            continue
        merged={k:np.concatenate([x[k] for x in bucket]) for k in data}
        order=rng.permutation(len(merged['action']))
        for begin in range(0,len(order),batch_size):
            sel=order[begin:begin+batch_size]
            yield {k:v[sel] for k,v in merged.items()}
        bucket=[]
    if bucket:
        merged={k:np.concatenate([x[k] for x in bucket]) for k in bucket[0]}
        order=rng.permutation(len(merged['action']))
        for begin in range(0,len(order),batch_size):
            sel=order[begin:begin+batch_size]
            yield {k:v[sel] for k,v in merged.items()}


def evaluate_val(model,n_each,seed_base=VAL_SEED):
    """K1 guided greedy and K4 guided/scarcity assessed on SAME fixed instances."""
    device=next(model.parameters()).device
    training=model.training
    model.eval().cpu()
    totals={k:{'success':0,'count':0,'values':[],'by_mode':{}} for k in ('K1','K4')}
    with torch.inference_mode():
        for mode in MODES:
            mode_values={'K1':[],'K4':[]}
            for ix in range(n_each):
                seed=old.seed_for(mode,ix,seed_base)
                env=old.mixed_env(seed,mode); env.init_state()
                winner,baseline,_=old.choose_candidates(env,model,seed+99331,
                    old.DEFAULT_ALPHA['stage_a'],old.DEFAULT_BETA['stage_a'])
                for key,data in (('K1',baseline[1]),('K4',winner[1])):
                    r=data['result']
                    complete=bool(r['done'])
                    cmax=float(r['makespan']) if complete else 200.0
                    totals[key]['success']+=int(complete)
                    totals[key]['count']+=1
                    totals[key]['values'].append(cmax)
                    mode_values[key].append(cmax)
            for key in totals:
                totals[key]['by_mode'][mode]=float(np.mean(mode_values[key]))
    model.to(device).train(training)
    result={}
    for key,t in totals.items():
        result[key]={'success':int(t['success']),'total':int(t['count']),
                     'penalized_makespan':float(np.mean(t['values'])),
                     'by_mode':t['by_mode']}
    return result


def best_tuple(metrics):
    return (int(metrics['success']),-float(metrics['penalized_makespan']))


def save_best(model,path,epoch,arm,run_seed,metrics,name,trainable):
    old.atomic_torch({'weights':old.state_dict_cpu(model),'epoch':epoch,'arm':arm,
                      'run_seed':run_seed,'selection_method':name,'metrics':metrics,
                      'score':best_tuple(metrics),'trainable':trainable,
                      'initial_stage_a_sha':old.sha256(HERE/'reference'/'stage_a_best_model.pt')},path)


def ensure_demonstrations(args,out):
    demo_dir=out/'demonstrations'
    expected=args.teacher_cases
    teacher_files=list(demo_dir.glob('*_teacher.npz')) if demo_dir.exists() else []
    self_files=list(demo_dir.glob('*_self.npz')) if demo_dir.exists() else []
    if len(teacher_files)==expected and len(self_files)==expected:
        notify(f'[DATASET] Reusing verified {expected} previous teacher/self trace pairs')
        return demo_dir
    # Optional already-extracted previous data folder on Kaggle Add Input.
    if args.demo_dir:
        supplied=Path(args.demo_dir)
        if not supplied.is_dir(): raise FileNotFoundError(supplied)
        demo_dir.mkdir(parents=True,exist_ok=True)
        for suffix in ('teacher','self'):
            files=list(supplied.glob('*_'+suffix+'.npz'))
            if len(files)!=expected:
                raise RuntimeError(f'Provided demo dir has {len(files)} {suffix} files, expected {expected}')
            for item in files:
                target=demo_dir/item.name
                if not target.exists(): shutil.copyfile(item,target)
        notify(f'[DATASET] Imported {expected} old demonstration pairs from {supplied}')
        return demo_dir
    # Optional data import: results (29).zip, no need to recompute K4 teacher.
    if args.demo_zip:
        zpath=Path(args.demo_zip)
        if not zpath.is_file(): raise FileNotFoundError(zpath)
        z=zipfile.ZipFile(zpath)
        entries=[x for x in z.infolist() if '/demonstrations/' in x.filename and x.filename.endswith('.npz')]
        if len(entries)<2*expected:
            raise RuntimeError(f'ZIP has {len(entries)} demonstration files, expected >= {2*expected}')
        demo_dir.mkdir(parents=True,exist_ok=True)
        for item in entries:
            name=Path(item.filename).name
            if not name.endswith(('_teacher.npz','_self.npz')): continue
            target=demo_dir/name
            if not target.exists():
                with z.open(item) as stream, target.open('wb') as dst:
                    shutil.copyfileobj(stream,dst)
        notify(f'[DATASET] Imported demonstrations from {zpath.name}')
        teacher_files=list(demo_dir.glob('*_teacher.npz'))
        self_files=list(demo_dir.glob('*_self.npz'))
        if len(teacher_files)==expected and len(self_files)==expected:
            # Verify seed/case identity for all traces to prevent silent reuse of unrelated demos.
            for idx in (0, expected-1):
                t=sorted(teacher_files)[idx]
                with np.load(t,allow_pickle=False) as data:
                    meta=json.loads(str(data['meta_json'].item()))
                if int(meta.get('teacher_K',0))!=4: raise RuntimeError('Unsupported old teacher K')
            return demo_dir
        raise RuntimeError(f'Imported {len(teacher_files)} teacher and {len(self_files)} self files; incorrect count')
    notify('[DATASET] Prior demonstrations not attached, generating verified teacher rollouts (K=4)')
    old.make_dataset(args,out)
    return demo_dir


def train_one(args,out,arm,seed,reference,initial_validation):
    rundir=out/'runs'/f'seed_{seed}'/arm
    rundir.mkdir(parents=True,exist_ok=True)
    model=old.load_stage_a()
    initial=old.state_dict_cpu(model)
    grouped,trainable,total=old.freeze_selectively(model)
    dev=torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model.to(dev)
    opt=old.make_optimizer(model,grouped,args)
    cp=rundir/'checkpoint_resume.pt'
    sig=args_signature(args,seed,arm)
    best_metrics={'K1':initial_validation['K1'],'K4':initial_validation['K4']}
    scores={key:best_tuple(initial_validation[key]) for key in ('K1','K4')}
    best_epoch={'K1':0,'K4':0}
    start=0
    history=[]
    if cp.is_file():
        ck=torch.load(cp,map_location=dev,weights_only=False)
        if ck['signature']!=sig:
            raise RuntimeError(f'Checkpoint config changed for {rundir}; use another --out')
        model.load_state_dict(ck['weights'],strict=True)
        opt.load_state_dict(ck['optimizer'])
        scores={k:tuple(v) for k,v in ck['scores'].items()}
        best_epoch=ck['best_epoch']
        best_metrics=ck['best_metrics']
        history=ck['history']
        start=int(ck['epoch'])
        notify(f'[RESUME] {arm} seed={seed} from epoch {start}')
    else:
        for method in ('K1','K4'):
            save_best(model,rundir/f'best_{method}.pt',0,arm,seed,initial_validation[method],method,trainable)
    demos=sorted((out/'demonstrations').glob('*_'+('self' if arm=='self_kl' else 'teacher')+'.npz'))
    if len(demos)!=args.teacher_cases:
        raise RuntimeError(f'{arm}: available demos {len(demos)}, expected {args.teacher_cases}')
    notify(f'[MODEL] {arm} seed={seed} trainable={trainable}/{total} device={dev} KL={args.kl_weight}')
    reference=reference.to(dev).eval()
    for epoch in range(start,args.epochs):
        old.set_seed(seed+epoch*137)
        model.train()
        totals=[]
        nlls=[]; kls=[]; avgweights=[]
        nsteps=0
        for batch in weighted_batches(demos,args.batch_size,seed+epoch*7919,arm,args):
            tasks=torch.from_numpy(batch['tasks']).to(dev)
            agents=torch.from_numpy(batch['agents']).to(dev)
            mask=torch.from_numpy(batch['mask']).to(dev)
            index=torch.from_numpy(batch['agent_id']).to(dev).reshape(-1,1,1)
            action=torch.from_numpy(batch['action']).to(dev)
            weights=torch.from_numpy(batch['adv_weight']).to(dev)
            opt.zero_grad(set_to_none=True)
            _,logp=model(tasks,agents,mask,index)
            if not torch.isfinite(logp).all(): raise RuntimeError('Nonfinite student log probabilities')
            selected=logp.gather(-1,action.unsqueeze(-1)).squeeze(-1)
            if not torch.isfinite(selected).all(): raise RuntimeError('Nonfinite legal selected actions')
            # Divide by the SUM of weights, keeping the step-size comparable across arms.
            nll=(-(selected)*weights).sum()/weights.sum().clamp_min(1e-8)
            with torch.no_grad():
                _,ref_logp=reference(tasks,agents,mask,index)
                ref_prob=ref_logp.exp()
            # KL(pi_StageA || pi_student), on the exact ORIGINAL legal-action mask.
            # Masked terms have p_ref=0 and contribute zero.
            kl=(ref_prob*(ref_logp-logp)).sum(-1).mean()
            loss=nll+args.kl_weight*kl
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite weighted NLL / KL loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            opt.step()
            totals.append(float(loss.detach().cpu()))
            nlls.append(float(nll.detach().cpu()))
            kls.append(float(kl.detach().cpu()))
            avgweights.append(float(weights.mean().detach().cpu()))
            nsteps+=1
            if nsteps>=args.updates_per_epoch: break
        if nsteps!=args.updates_per_epoch:
            raise RuntimeError(f'{arm} produced only {nsteps} batches; need {args.updates_per_epoch}')
        validation=evaluate_val(model,args.val_each,VAL_SEED)
        for method in ('K1','K4'):
            score=best_tuple(validation[method])
            if score>scores[method]:
                scores[method]=score
                best_epoch[method]=epoch+1
                best_metrics[method]=validation[method]
                save_best(model,rundir/f'best_{method}.pt',epoch+1,arm,seed,validation[method],method,trainable)
                notify(f'[BEST] arm={arm} seed={seed} {method} epoch={epoch+1} Cmax={validation[method]["penalized_makespan"]:.4f}')
        for name,p in model.named_parameters():
            if not p.requires_grad and not torch.equal(p.detach().cpu(),initial[name]):
                raise AssertionError('Frozen tensor changed: '+name)
        record={'arm':arm,'seed':seed,'epoch':epoch+1,'updates':nsteps,
                'loss':float(np.mean(totals)),'weighted_nll':float(np.mean(nlls)),
                'kl_ref_student':float(np.mean(kls)),'mean_adv_weight':float(np.mean(avgweights)),
                'val_k1_success':validation['K1']['success'],'val_k1_cmax':validation['K1']['penalized_makespan'],
                'val_k4_success':validation['K4']['success'],'val_k4_cmax':validation['K4']['penalized_makespan'],
                'best_k1_epoch':best_epoch['K1'],'best_k4_epoch':best_epoch['K4']}
        history.append(record)
        old.atomic_csv(history,rundir/'training_history.csv')
        old.atomic_torch({'signature':sig,'epoch':epoch+1,'weights':old.state_dict_cpu(model),
                          'optimizer':opt.state_dict(),'scores':scores,'best_epoch':best_epoch,
                          'best_metrics':best_metrics,'history':history},cp)
        notify(f'[TRAIN] arm={arm} seed={seed} epoch={epoch+1}/{args.epochs} '
               f'loss={record["loss"]:.5f} NLL={record["weighted_nll"]:.5f} KL={record["kl_ref_student"]:.6f} '
               f'val_K1={record["val_k1_cmax"]:.4f} val_K4={record["val_k4_cmax"]:.4f} '
               f'best_K1={best_epoch["K1"]} best_K4={best_epoch["K4"]}')
    reference.cpu()
    return rundir


def test_models(args,out):
    """Evaluate only after all training and checkpoint decisions are completed."""
    refs={'stage_a':old.load_stage_a().eval(),'attention':old.load_attention().eval()}
    # For every arm, evaluate best_K4 at BOTH search budgets; and weighted best_K1
    # explicitly at K1 to check if the selection metric matters.
    for seed in args.seeds:
        for arm in ARMS:
            for checkpoint in ('K4',)+(('K1',) if arm=='teacher_adv_kl' else ()):
                fn=out/'runs'/f'seed_{seed}'/arm/f'best_{checkpoint}.pt'
                if not fn.is_file(): raise FileNotFoundError(fn)
                refs[f'{arm}_seed{seed}_best{checkpoint}']=old.load_arm_weights(fn).eval()
    rows=[]
    expected=len(refs)*args.test_each*len(MODES)*2
    for mode in MODES:
        for idx in range(args.test_each):
            case_seed=old.seed_for(mode,idx,TEST_SEED)
            env=old.mixed_env(case_seed,mode)
            env.init_state()
            for name,model in refs.items():
                aa=old.DEFAULT_ALPHA['attention' if name=='attention' else 'stage_a']
                bb=old.DEFAULT_BETA['attention' if name=='attention' else 'stage_a']
                begin=time.perf_counter()
                winner,baseline,candidates=old.choose_candidates(env,model,case_seed+99331,aa,bb)
                elapsed=time.perf_counter()-begin
                for method,x,t in (('guided_K1',baseline[1],float(baseline[1]['result'].get('runtime_s',0))),
                                   ('guided_scarcity_K4',winner[1],elapsed)):
                    r=x['result']
                    old.g.replay_and_verify(env,x)
                    rows.append({'mode':mode,'case':idx,'case_seed':case_seed,'model':name,
                                 'method':method,'done':int(r['done']),
                                 'makespan':float(r['makespan']),
                                 'penalized_makespan':float(r['makespan']) if r['done'] else 200.0,
                                 'waiting_time':float(r['waiting_time']),
                                 'wall_time_s':t,'replay_verified':1,'K':1 if method=='guided_K1' else 4})
            old.atomic_csv(rows,out/'evaluation_per_case.csv')
            notify(f'[TEST] {mode} {idx+1}/{args.test_each} evaluated {len(refs)} models; replay OK')
    assert len(rows)==expected,(len(rows),expected)
    summaries=[]
    for model in refs:
        for method in ('guided_K1','guided_scarcity_K4'):
            for mode in ('ALL',)+MODES:
                a=[r for r in rows if r['model']==model and r['method']==method and (mode=='ALL' or r['mode']==mode)]
                summaries.append({'model':model,'method':method,'mode':mode,'cases':len(a),
                                  'success':sum(r['done'] for r in a),
                                  'mean_penalized_makespan':float(np.mean([r['penalized_makespan'] for r in a])),
                                  'mean_waiting':float(np.mean([r['waiting_time'] for r in a])),
                                  'mean_wall_time_s':float(np.mean([r['wall_time_s'] for r in a]))})
    old.atomic_csv(summaries,out/'evaluation_summary.csv')
    pairs=[]
    for method in ('guided_K1','guided_scarcity_K4'):
        for seed in args.seeds:
            keys=[(f'teacher_adv_kl_seed{seed}_bestK4','stage_a'),
                  (f'teacher_adv_kl_seed{seed}_bestK4',f'teacher_uniform_kl_seed{seed}_bestK4'),
                  (f'teacher_adv_kl_seed{seed}_bestK4',f'self_kl_seed{seed}_bestK4'),
                  (f'teacher_adv_kl_seed{seed}_bestK4','attention')]
            if method=='guided_K1':
                keys.append((f'teacher_adv_kl_seed{seed}_bestK1','stage_a'))
            for a,b in keys:
                m={(r['mode'],r['case']):r['penalized_makespan'] for r in rows if r['model']==a and r['method']==method}
                n={(r['mode'],r['case']):r['penalized_makespan'] for r in rows if r['model']==b and r['method']==method}
                ix=sorted(set(m)&set(n))
                diff=np.array([m[i]-n[i] for i in ix],dtype=np.float64)
                if len(ix)!=len(MODES)*args.test_each:raise AssertionError('Cases do not match')
                strat={mode:np.array([j for j,k in enumerate(ix) if k[0]==mode]) for mode in MODES}
                rng=np.random.default_rng(260611)
                boot=[]
                for _ in range(args.bootstrap):
                    resampled=np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in strat.values()])
                    boot.append(float(diff[resampled].mean()))
                pairs.append({'method':method,'model_a':a,'model_b':b,'cases':len(ix),
                              'delta_a_minus_b':float(diff.mean()),
                              'ci95_low':float(np.quantile(boot,.025)),
                              'ci95_high':float(np.quantile(boot,.975))})
    old.atomic_csv(pairs,out/'paired_statistics.csv')
    notify(f'[TEST DONE] {len(rows)} rows, {len(refs)} models, paired CIs exported')
    return summaries


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=str,default='/kaggle/working/ADV_KL_RESULTS')
    ap.add_argument('--teacher-cases',type=int,default=1500)
    ap.add_argument('--demo-zip',type=str,default='')
    ap.add_argument('--demo-dir',type=str,default='')
    ap.add_argument('--epochs',type=int,default=4)
    ap.add_argument('--updates-per-epoch',type=int,default=300)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--val-each',type=int,default=8)
    ap.add_argument('--test-each',type=int,default=50)
    ap.add_argument('--seed',type=int,default=MODEL_SEED)
    ap.add_argument('--seeds',type=int,nargs='+',default=[20261110,20261111,20261112])
    ap.add_argument('--lr-adapter',type=float,default=1e-5)
    ap.add_argument('--lr-edge',type=float,default=2e-5)
    ap.add_argument('--lr-kan2',type=float,default=2e-6)
    ap.add_argument('--kl-weight',type=float,default=0.05)
    ap.add_argument('--adv-floor',type=float,default=0.35)
    ap.add_argument('--adv-temperature',type=float,default=1.5)
    ap.add_argument('--adv-cap',type=float,default=3.0)
    ap.add_argument('--bootstrap',type=int,default=1000)
    ap.add_argument('--log-every',type=int,default=25)
    ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.smoke:
        a.teacher_cases=3;a.epochs=1;a.updates_per_epoch=2;a.val_each=1;a.test_each=1
        a.seeds=[MODEL_SEED];a.demo_zip='';a.demo_dir='';a.bootstrap=100
    if min(a.teacher_cases,a.epochs,a.updates_per_epoch,a.batch_size,a.val_each,a.test_each,a.bootstrap)<1:
        raise ValueError('All dimensions must be strictly positive')
    if not (0<a.adv_floor<=a.adv_cap and a.adv_temperature>0 and a.kl_weight>=0):
        raise ValueError('Invalid advantage/KL settings')
    torch.set_num_threads(1)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    old.set_seed(a.seed)
    settings={'protocol':'advantage_weighted_selective_distillation_v1',
              'training_seed':a.seed,'training_seeds':a.seeds,'teacher_cases':a.teacher_cases,
              'training_arms':ARMS,'epochs':a.epochs,'updates_per_epoch':a.updates_per_epoch,
              'batch_size':a.batch_size,'val_each':a.val_each,'test_each':a.test_each,
              'val_seed':VAL_SEED,'heldout_seed':TEST_SEED,'KL_ref_student':a.kl_weight,
              'advantage_formula':'clip(adv_floor + max(0,Cmax_guided_greedy-Cmax_best_K4)/temperature, adv_floor, adv_cap)',
              'frozen_policy':'Stage A original frozen backbone, adapter/edge_bias/KAN2 trainable',
              'stageA_sha256':old.sha256(HERE/'reference'/'stage_a_best_model.pt'),
              'no_official_test':'official RALTestSet not used in selection or test',
              'checkpoint_selection':'success then penalized Cmax, independently for K1 and K4',
              'smoke':a.smoke}
    old.atomic_json(settings,out/'provenance.json')
    notify('[START] advantage-weighted teacher + KL; 5498/25138 parameters; best K1 and best K4 separately')
    demo_dir=ensure_demonstrations(a,out)
    teacher=list(demo_dir.glob('*_teacher.npz'))
    advantages=[]
    metadata_rows=[]
    for file in teacher:
        with np.load(file,allow_pickle=False) as f:
            meta=json.loads(str(f['meta_json'].item()))
            advantages.append(float(meta['improvement']))
            metadata_rows.append(meta)
    old.atomic_csv(sorted(metadata_rows,key=lambda r:int(r['case'])),out/'teacher_collection.csv')
    notify(f'[TEACHER] {len(advantages)} cases; mean improvement={np.mean(advantages):.5f} positive={(np.array(advantages)>0).mean():.2%}')
    reference=old.load_stage_a().eval()
    initial=evaluate_val(reference,a.val_each,VAL_SEED)
    notify(f'[INITIAL VAL] K1={initial["K1"]["penalized_makespan"]:.4f} K4={initial["K4"]["penalized_makespan"]:.4f} success={initial["K4"]["success"]}/{initial["K4"]["total"]}')
    # Same original initialization and validation for all arms/seeds.
    for run_seed in a.seeds:
        for arm in ARMS:
            train_one(a,out,arm,run_seed,reference,initial)
    summaries=test_models(a,out)
    notify('[DONE] full selective advantage/KL training + heldout evaluation')
    for r in summaries:
        if r['mode']=='ALL':
            notify(f'[SUMMARY] {r["model"]} {r["method"]} Cmax={r["mean_penalized_makespan"]:.5f} success={r["success"]}/{r["cases"]}')

if __name__=='__main__':
    main()
