import zipfile, io, json, hashlib, numpy as np, pandas as pd
from scipy import stats
from collections import Counter
Z=zipfile.ZipFile('/mnt/data/results (33).zip'); R='PAPER2_SADCHER_CROSSDATASET_300/'
df=pd.read_csv(io.BytesIO(Z.read(R+'evaluation_per_case.csv')))
assert df.shape[0]==1200 and df.case_index.nunique()==300
expected_models={'stage_a','attention_original'}; expected_methods={'guided_K1','guided_scarcity_K4'}
assert set(df.model)==expected_models and set(df.method)==expected_methods
assert (df.groupby('case_index').size()==4).all()
assert (df.complete==1).all() and (df.replay_verified==1).all()
assert (df.n_candidate_rollouts==df.K).all()
assert (df.completed_candidates==df.K).all()
assert not df.penalized_makespan.isna().any()
assert np.allclose(df.makespan,df.penalized_makespan)
assert (df.groupby('case_index').case_seed.nunique()==1).all()
assert (df.groupby('case_index').case_sha256.nunique()==1).all()
assert df.case_sha256.nunique()==300
assert df.search_seed.nunique()==300
for i, g in df.groupby('case_index'):
 assert int(g.case_seed.iloc[0]) == 2026121200+1009*i
 assert all(g.case_sha256 == g.case_sha256.iloc[0]); assert all(g.search_seed == g.search_seed.iloc[0])
for model in expected_models:
 a=df[(df.model==model)&(df.K==1)].set_index('case_index')
 b=df[(df.model==model)&(df.K==4)].set_index('case_index')
 assert (b.penalized_makespan<=a.penalized_makespan+1e-10).all()
 match=b.winner_candidate=='guided_greedy'
 assert np.allclose(a.penalized_makespan[match],b.penalized_makespan[match])
 print(model,'K4 first candidate wins',int(match.sum()),'K4 improved',int((b.penalized_makespan<a.penalized_makespan-1e-9).sum()), 'mean_k4_gain',float((a.penalized_makespan-b.penalized_makespan).mean()),'candidate counts',b.winner_candidate.value_counts().to_dict())
plans=[x for x in Z.namelist() if x.startswith(R+'plans/') and x.endswith('.json')]
assert len(plans)==1200
wrong=[]
for p in plans:
 x=json.loads(Z.read(p)); i=x['case_index']; s=df[(df.case_index==i)&(df.model==x['model'])&(df.method==x['method'])];assert len(s)==1
 r=s.iloc[0]
 assert x['K']==r.K and x['trace_sha256']==r.trace_sha256 and x['replay_verified'] and x['case_sha256']==r.case_sha256
 assert abs(x['result']['makespan']-r.makespan)<1e-8
 # Check trace hash as in source, simple canonical JSON
 h=hashlib.sha256(json.dumps(x['trace'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if h!=r.trace_sha256:wrong.append((p,h,r.trace_sha256))
print('trace canonical sha mismatches',len(wrong),wrong[:1]);
proto=json.loads(Z.read(R+'protocol_locked.json'))
for path,exp in [('Sadcher_official_repository/data_generation/problem_generator.py',proto['sadcher_generator_sha256']),('Paper2_TemporalSpatial_KAN/env/task_env.py',proto['HeteroMRTA_simulator_sha256']),('Paper2_TemporalSpatial_KAN/paper2_sadcher_external300.py',proto['evaluator_sha256']),('Paper2_TemporalSpatial_KAN/reference/stage_a_best_model.pt',proto['model_weights_sha256']['stage_a']),('Paper2_TemporalSpatial_KAN/weights/AttentionNet_published.pt',proto['model_weights_sha256']['attention_original'])]:
 h=hashlib.sha256(Z.read(path)).hexdigest(); assert h==exp,(path,h,exp)
print('all 5 SHA256 verified')
casefeatures=[]
for i in range(300):
 x=json.loads(Z.read(R+f'cases/case_{i:04d}.json')); d=x['data'];Q=np.array(d['Q']);T=np.array(d['T_e']);Rr=np.array(d['R']);P=d['precedence_constraints'];loc=np.array(d['task_locations']);
 assert x['sha256']==df[df.case_index==i].case_sha256.iloc[0]
 assert x['seed']==int(df[df.case_index==i].case_seed.iloc[0]); assert Q.shape==(15,5) and Rr.shape==(22,5) and len(T)==22 and loc.shape==(22,2)
 assert (P is None or len(P)==0) and np.array_equal(loc[0],loc[-1])
 assert (np.sum(Q,axis=1)>=1).all() and (np.sum(Rr[1:-1],axis=1)>=1).all()
 casefeatures.append({'case_index':i,'n_multiskill_tasks_calc':int(np.sum(np.sum(Rr[1:-1],axis=1)>1)),'n_multiskill_robots_calc':int(np.sum(np.sum(Q,axis=1)>1)),'mean_skills_task':float(np.sum(Rr[1:-1],axis=1).mean()),'mean_skills_robot':float(np.sum(Q,axis=1).mean())})
features=pd.DataFrame(casefeatures)
assert (df.merge(features,on='case_index').n_multiskill_tasks==df.merge(features,on='case_index').n_multiskill_tasks_calc).all()
assert (df.merge(features,on='case_index').n_multiskill_robots==df.merge(features,on='case_index').n_multiskill_robots_calc).all()
print('data shape all verified; multiskill tasks mean/range',features.n_multiskill_tasks_calc.mean(),features.n_multiskill_tasks_calc.min(),features.n_multiskill_tasks_calc.max(),'robots',features.n_multiskill_robots_calc.mean(),features.n_multiskill_robots_calc.min(),features.n_multiskill_robots_calc.max())

print('\n===PERFORMANCE===')
print(df.groupby(['model','method']).agg(n=('case_index','nunique'),mean=('penalized_makespan','mean'),sd=('penalized_makespan','std'),median=('penalized_makespan','median'),time=('search_elapsed_s','mean'),waiting=('waiting_time','mean'),decisions=('n_decisions','mean')).round(7).to_string())
rows=[]; RNG=np.random.default_rng(334455); B=30000
for K in [1,4]:
 a=df[(df.model=='stage_a')&(df.K==K)].set_index('case_index').sort_index();b=df[(df.model=='attention_original')&(df.K==K)].set_index('case_index').sort_index()
 diff=(a.penalized_makespan-b.penalized_makespan).to_numpy(); tdiff=(a.search_elapsed_s-b.search_elapsed_s).to_numpy();
 draws=RNG.integers(0,300,size=(B,300)); boots=diff[draws].mean(axis=1); bt=tdiff[draws].mean(axis=1)
 wins=int(np.sum(diff<-1e-9));losses=int(np.sum(diff>1e-9));ties=300-wins-losses
 t=stats.ttest_rel(a.penalized_makespan,b.penalized_makespan)
 binom=stats.binomtest(wins,wins+losses,p=0.5,alternative='greater')
 rows.append({'K':K,'n':300,'stage_a_mean':a.penalized_makespan.mean(),'attention_mean':b.penalized_makespan.mean(),'mean_delta':diff.mean(),'CI95_lower_recomputed':np.quantile(boots,.025),'CI95_upper_recomputed':np.quantile(boots,.975),'median_delta':np.median(diff),'paired_sd_delta':np.std(diff,ddof=1),'stage_a_wins':wins,'attention_wins':losses,'ties':ties,'sign_test_p_two_sided':stats.binomtest(wins,wins+losses,p=.5).pvalue,'paired_t_p_two_sided':t.pvalue,'stage_a_time':a.search_elapsed_s.mean(),'attention_time':b.search_elapsed_s.mean(),'time_reduction_percent':100*(1-a.search_elapsed_s.mean()/b.search_elapsed_s.mean()),'time_difference_CI95_low':np.quantile(bt,.025),'time_difference_CI95_high':np.quantile(bt,.975)})
 print('K',K,'diff summary percentiles',np.quantile(diff,[0,.05,.25,.5,.75,.95,1]))
 print('K',K,'K4 search time quantiles stage',np.quantile(a.search_elapsed_s,[.1,.5,.9,.99]),'attention',np.quantile(b.search_elapsed_s,[.1,.5,.9,.99]))
res=pd.DataFrame(rows);print(res.to_string(index=False))

for K in [1,4]:
 a=df[(df.model=='stage_a')&(df.K==K)].set_index('case_index').sort_index();b=df[(df.model=='attention_original')&(df.K==K)].set_index('case_index').sort_index()
 tmp=features.set_index('case_index').copy();tmp['delta']=a.penalized_makespan-b.penalized_makespan
 tmp['q']=pd.qcut(tmp.n_multiskill_tasks_calc,4,duplicates='drop')
 print('\n K=',K,'n_multiskill quartiles\n',tmp.groupby('q',observed=True).agg(n=('delta','size'),mean_delta=('delta','mean'),std_delta=('delta','std'),mean_multiskill=('n_multiskill_tasks_calc','mean')).to_string())
 print('first_150 & second_150',tmp.iloc[:150].delta.mean(),tmp.iloc[150:].delta.mean())
print('\noriginal 300 final summary and paired stats independent checks:')
print(pd.read_csv(io.BytesIO(Z.read(R+'paired_statistics.csv'))).to_string(index=False))

import os
O='/mnt/data/paper2_results33_audit'
os.makedirs(O,exist_ok=True)
res.to_csv(O+'/independent_paired_statistics.csv',index=False)
features.to_csv(O+'/case_features.csv',index=False)
df.groupby(['model','method']).agg(n=('case_index','nunique'),mean=('penalized_makespan','mean'),sd=('penalized_makespan','std'),median=('penalized_makespan','median'),mean_search_s=('search_elapsed_s','mean'),mean_wait=('waiting_time','mean')).to_csv(O+'/recomputed_summary.csv')
print('Saved audited summary in',O)
