"""Walk-forward reproducible. No lee etiquetas del test durante la selección.

python scripts/run_training_experiments.py --stage isolated
python scripts/run_training_experiments.py --stage combinations
python scripts/run_training_experiments.py --stage tune
python scripts/run_training_experiments.py --stage final
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import inspect
import itertools
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                             HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'include'))
from deadlock import schema
from deadlock.experiment_data import TRAIN_CUTS, TEST_CUT, STRESS_CUT, WINDOWS, sha256, write_json
from deadlock.shrink import shrink_rate

OUT = ROOT/'experiments/training_strategy'
MODELS = ['dummy','logistic','ridge','sgd','linear_svm','gaussian_nb','decision_tree',
          'random_forest','extra_trees','gradient_boosting','hist_gradient_boosting',
          'adaboost','mlp','rbf_approx','knn','lda','qda']
POLICIES = ['expanding','sliding_1','decay_1','decay_2']
BLOCKS = ['A1','A2','composition','all','A1_trend','all_trend']
NO_WEIGHT = {'knn','lda','qda'}
SEED = 42


def estimator(name, params=None):
    models = dict(
        dummy=lambda: DummyClassifier(strategy='prior'),
        logistic=lambda: LogisticRegression(C=.01, max_iter=2000, random_state=SEED),
        ridge=lambda: RidgeClassifier(alpha=100.),
        sgd=lambda: SGDClassifier(loss='log_loss',alpha=.01,max_iter=2000,tol=1e-5,
                                  average=True,random_state=SEED),
        linear_svm=lambda: LinearSVC(C=.01,dual=False,max_iter=3000,random_state=SEED),
        gaussian_nb=lambda: GaussianNB(var_smoothing=1e-5),
        decision_tree=lambda: DecisionTreeClassifier(max_depth=4,min_samples_leaf=100,random_state=SEED),
        random_forest=lambda: RandomForestClassifier(n_estimators=100,max_depth=7,min_samples_leaf=50,
                                                      n_jobs=1,random_state=SEED),
        extra_trees=lambda: ExtraTreesClassifier(n_estimators=100,max_depth=7,min_samples_leaf=50,
                                                  n_jobs=1,random_state=SEED),
        gradient_boosting=lambda: GradientBoostingClassifier(n_estimators=60,max_depth=2,
                                 min_samples_leaf=100,learning_rate=.05,random_state=SEED),
        hist_gradient_boosting=lambda: HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,
                                   min_samples_leaf=100,l2_regularization=10.,learning_rate=.05,
                                   early_stopping=False,random_state=SEED),
        adaboost=lambda: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,min_samples_leaf=100),
                                            n_estimators=60,learning_rate=.05,random_state=SEED),
        mlp=lambda: MLPClassifier(hidden_layer_sizes=(16,),alpha=1.,max_iter=120,early_stopping=False,
                                  batch_size=512,tol=1e-5,n_iter_no_change=12,random_state=SEED),
        rbf_approx=lambda: LogisticRegression(C=.1,max_iter=1000,random_state=SEED),
        knn=lambda: KNeighborsClassifier(n_neighbors=301,weights='distance',n_jobs=1),
        lda=lambda: LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'),
        qda=lambda: QuadraticDiscriminantAnalysis(reg_param=.5),
    )
    obj = models[name]()
    if params: obj.set_params(**params)
    return obj


def columns(block):
    mapping = dict(A1=['hero_winrate_adv_mean'], A2=schema.COUNTER+schema.SYNERGY+schema.HERO_WINRATE,
                   composition=schema.COMPOSICION, all=schema.FEATURE_COLUMNS)
    base = block.replace('_trend','')
    return mapping[base]+(['hero_trend_adv_mean'] if block.endswith('_trend') else [])


@lru_cache(maxsize=24)
def dataset(cut, days):
    folder = ROOT/'include/output/silver'/f'corte={cut}'/f'dias={days}'/'k=100'
    meta = json.loads((folder/'parametros.json').read_text(encoding='utf-8'))
    assert meta['features_sha256'] == sha256(folder/'features.parquet')
    assert (meta['dias_ventana_stats'],meta['k_shrinkage'],meta['match_mode']) == (days,100,'ranked')
    df = pd.read_parquet(folder/'features.parquet')
    hero = {}
    for window in (7,14):
        path = ROOT/'include/output/bronze'/f'corte={cut}'/'stats'/f'dias={window}'/'hero_stats.json'
        hero[window] = {r['hero_id']:r for r in json.loads(path.read_text(encoding='utf-8'))}
    trend = []
    for h,_ in schema.HEROES:
        recent, full = hero[7].get(h,{}), hero[14].get(h,{})
        n, w = full.get('matches',0)-recent.get('matches',0), full.get('wins',0)-recent.get('wins',0)
        assert 0 <= w <= n, (cut,h,n,w)
        trend.append(shrink_rate(recent.get('wins',0),recent.get('matches',0),100)-shrink_rate(w,n,100))
    df['hero_trend_adv_mean'] = df[schema.COMPOSICION].to_numpy() @ np.array(trend)/6
    df['corte'] = cut
    return df


def weights_for(groups, policy):
    if not policy.startswith('decay'): return None
    ordered = sorted(set(groups))
    half = float(policy.split('_')[1])
    age = {cut:len(ordered)-1-i for i,cut in enumerate(ordered)}
    weights = np.array([.5**(age[c]/half) for c in groups])
    # Misma suma de pesos que cantidad de filas: no cambiar C por cambiar la escala.
    return weights/weights.mean()


def fit_predict(config, train, valid):
    cols = columns(config['block'])
    x, z = train[cols].to_numpy(dtype=float), valid[cols].to_numpy(dtype=float)
    y = train.winner.eq('A').to_numpy(dtype=int)
    weight = weights_for(train.corte, config['policy'])
    scaler = StandardScaler().fit(x, sample_weight=weight)
    x, z = scaler.transform(x), scaler.transform(z)
    kernel = None
    if config['model']=='rbf_approx':
        kernel = Nystroem(kernel='rbf',gamma=1/len(cols),n_components=128,random_state=SEED)
        x, z = kernel.fit_transform(x), kernel.transform(z)
    model = estimator(config['model'], config.get('params'))
    kw = {} if weight is None else {'sample_weight':weight}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model.fit(x,y,**kw)
        predicted = model.predict(z)
        proba = model.predict_proba(z)[:,1] if hasattr(model,'predict_proba') else None
        score = proba if proba is not None else model.decision_function(z)
    actual = valid.winner.eq('A').to_numpy(dtype=int)
    assert np.isfinite(score).all()
    metrics = dict(auc=float(roc_auc_score(actual,score)),accuracy=float(accuracy_score(actual,predicted)),
                   brier=float(brier_score_loss(actual,proba)) if proba is not None else None,
                   logloss=float(log_loss(actual,np.clip(proba,1e-15,1-1e-15))) if proba is not None else None,
                   fit_predict_seconds=time.perf_counter()-started,
                   warnings=sorted(set(str(w.message) for w in caught)),
                   n_train=len(train),n_valid=len(valid),n_features=len(cols),
                   effective_n=float(weight.sum()**2/(weight@weight)) if weight is not None else len(train))
    artifact=dict(scaler=scaler,kernel=kernel,model=model,columns=cols,config=config)
    return metrics, score, proba, predicted, artifact


def config_id(c):
    return '|'.join(str(c[k]) for k in ('model','block','days','policy'))+'|'+json.dumps(c.get('params',{}),sort_keys=True)


def label_isolated(block, days, policy):
    if days==14:
        if policy=='expanding' and block in ('A1','A2','all','composition'): return block
        if policy=='sliding_1' and block in ('A1','all'): return 'B1' if block=='A1' else 'B2'
        if policy.startswith('decay') and block=='all': return 'C'
        if policy=='expanding' and block=='A1_trend': return 'E'
    if block=='A1' and policy=='expanding' and days in (7,20): return 'D'
    return None


def matrix():
    configs=[]
    for block, days, policy, model in itertools.product(BLOCKS,WINDOWS,POLICIES,MODELS):
        label = label_isolated(block,days,policy)
        c = dict(model=model,block=block,days=days,policy=policy,
                 phase='isolated' if label else 'combinations',alternative=label or 'combination')
        c['id']=config_id(c)
        configs.append(c)
    return configs


def run_config(config):
    if config['policy'].startswith('decay') and config['model'] in NO_WEIGHT:
        return dict(config=config,status='unsupported',reason='fit no admite sample_weight; no se sustituye por remuestreo')
    try:
        folds=[]
        if config['phase']=='stress':
            cuts=(TRAIN_CUTS+[TEST_CUT])[-1:] if config['policy']=='sliding_1' else TRAIN_CUTS+[TEST_CUT]
            train=pd.concat([dataset(c,config['days']) for c in cuts],ignore_index=True)
            valid=dataset(STRESS_CUT,config['days'])
            assert train.match_date.max()<valid.match_date.min()
            metrics,*_=fit_predict(config,train,valid)
            return dict(config=config,status='ok',folds=[dict(step=1,train_cuts=cuts,valid_cut=STRESS_CUT,**metrics)])
        for i in range(1,len(TRAIN_CUTS)):
            cuts = TRAIN_CUTS[max(0,i-1):i] if config['policy']=='sliding_1' else TRAIN_CUTS[:i]
            train = pd.concat([dataset(c,config['days']) for c in cuts],ignore_index=True)
            valid = dataset(TRAIN_CUTS[i],config['days'])
            assert train.match_date.max() < valid.match_date.min()
            assert not set(train.match_id).intersection(valid.match_id)
            metrics,*_ = fit_predict(config, train, valid)
            folds.append(dict(step=i,train_cuts=cuts,valid_cut=TRAIN_CUTS[i],**metrics))
        return dict(config=config,status='ok',folds=folds)
    except Exception as e:
        import traceback
        return dict(config=config,status='error',reason=str(e),traceback=traceback.format_exc())


def read_results():
    path = OUT/'results.jsonl'
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    return list({r['config']['id']:r for r in rows}.values())


def summarize():
    results=read_results()
    rows,folds=[],[]
    for r in results:
        c=r['config']
        row={**c,'params':json.dumps(c.get('params',{}),sort_keys=True),'status':r['status']}
        if r['status']=='ok':
            for metric in ('auc','accuracy','brier','logloss'):
                vals=[f[metric] for f in r['folds'] if f[metric] is not None]
                row[f'{metric}_mean']=float(np.mean(vals)) if vals else None
                row[f'{metric}_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            row['seconds']=sum(f['fit_predict_seconds'] for f in r['folds'])
            row['warning_count']=sum(len(f['warnings']) for f in r['folds'])
            for f in r['folds']: folds.append({**row,**f})
        else: row['reason']=r.get('reason','')
        rows.append(row)
    df=pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUT/'summary.csv',index=False)
        pd.DataFrame(folds).to_csv(OUT/'folds.csv',index=False)
    return df


def tuning_configs():
    # Se ajusta cada familia en su mejor combinación; también logística x bloques x C a 14d.
    df=summarize(); configs=[]
    grids={
        'dummy':{}, 'logistic':{'C':[.0001,.001,.01,.1,1.,10.]},
        'ridge':{'alpha':[1.,100.,10000.]}, 'sgd':{'alpha':[.001,.01,.1]},
        'linear_svm':{'C':[.001,.01,.1]}, 'gaussian_nb':{'var_smoothing':[1e-9,1e-5,.01]},
        'decision_tree':{'max_depth':[2,4,8],'min_samples_leaf':[50,200]},
        'random_forest':{'max_depth':[4,7,None],'min_samples_leaf':[20,100]},
        'extra_trees':{'max_depth':[4,7,None],'min_samples_leaf':[20,100]},
        'gradient_boosting':{'n_estimators':[60,120],'max_depth':[1,2]},
        'hist_gradient_boosting':{'max_leaf_nodes':[3,7,15],'l2_regularization':[1.,10.]},
        'adaboost':{'n_estimators':[30,100],'learning_rate':[.05,.3]},
        'mlp':{'hidden_layer_sizes':[(8,),(32,)],'alpha':[1.,10.]},
        'rbf_approx':{'C':[.01,.1,1.]}, 'knn':{'n_neighbors':[101,301,1001]},
        'lda':{'shrinkage':['auto',.1,.5]}, 'qda':{'reg_param':[.1,.5,.9]},
    }
    for model,grid in grids.items():
        best=df[(df.status=='ok')&(df.model==model)&(df.phase!='stress')].sort_values('auc_mean',ascending=False).iloc[0]
        keys=list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            c={k:best[k] for k in ('model','block','days','policy')}
            c.update(days=int(c['days']),phase='tune',alternative='tuning',params=dict(zip(keys,values)))
            c['id']=config_id(c);configs.append(c)
    for block,C in itertools.product(['A1','A2','composition','all'],[.0001,.001,.01,.1,1.,10.]):
        c=dict(model='logistic',block=block,days=14,policy='expanding',phase='tune',alternative='grid_blocks_C',params={'C':C})
        c['id']=config_id(c);configs.append(c)
    return list({c['id']:c for c in configs}.values())


def execute(configs,workers):
    done={r['config']['id'] for r in read_results() if r['status']!='error'}
    pending=[c for c in configs if c['id'] not in done]
    print(f'{len(pending)} configuraciones pendientes; workers={workers}',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, (OUT/'results.jsonl').open('a',encoding='utf-8') as log:
        futures={pool.submit(run_config,c):c for c in pending}
        for i,f in enumerate(as_completed(futures),1):
            result=f.result()
            log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush()
            c=result['config']; auc=np.mean([x['auc'] for x in result.get('folds',[])]) if result['status']=='ok' else float('nan')
            print(f'{i}/{len(pending)} {result["status"]} {c["id"]} AUC={auc:.5f}',flush=True)
            if result['status']=='error': print(result['traceback'],flush=True)
    summarize()


def select():
    df=summarize(); valid=df[(df.status=='ok')&(df.phase!='stress')].copy()
    reference=valid[(valid.model=='logistic')&(valid.block=='A1')&(valid.days==14)&(valid.policy=='expanding')&valid.params.eq('{}')].iloc[0]
    threshold=float(reference.auc_mean+reference.auc_std)
    # Prioridad declarada: A1, A2, B1, C, B2; siempre una logística simple si empata.
    surpass=valid[(valid.auc_mean>threshold)&(valid.model!='dummy')]
    if len(surpass): chosen=surpass.sort_values('auc_mean',ascending=False).iloc[0]
    else:
        simple=valid[(valid.model=='logistic')&(valid.block=='A1')&(valid.policy=='expanding')]
        tied=simple[simple.auc_mean>=simple.auc_mean.max()-1e-12]
        chosen=tied.sort_values(['brier_mean','days'],ascending=[True,True]).iloc[0]
    original=next(r['config'] for r in read_results() if r['config']['id']==chosen.id)
    decision=dict(config=original,reference_id=reference.id,reference_auc=float(reference.auc_mean),
                  reference_std=float(reference.auc_std),complexity_threshold=threshold,
                  candidates_above_threshold=len(surpass),selected_auc=float(chosen.auc_mean),
                  rule='A1 logística 14d como referencia. Complejidad solo si mejora > desvío muestral de A1; si empatan, logística A1, mejor ventana/C por CV. Empates de AUC (tolerancia 1e-12) dentro de A1 se resuelven por menor Brier de CV.',
                  selected_before_test=True,created_at=datetime.now(timezone.utc).isoformat())
    if (OUT/'selection.json').exists():
        previous=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
        if (OUT/'final_test.json').exists():
            assert previous['config']==decision['config'], 'No se permite cambiar selección después del test'
            return previous
    write_json(OUT/'selection.json',decision)
    return decision


def temporal_calibrator(config, cuts):
    """Platt postselección, exclusivamente con scores fuera de muestra del pasado."""
    scores,labels=[],[]
    for i in range(1,len(cuts)):
        previous=cuts[max(0,i-1):i] if config['policy']=='sliding_1' else cuts[:i]
        tr=pd.concat([dataset(c,config['days']) for c in previous],ignore_index=True)
        va=dataset(cuts[i],config['days'])
        _,score,*_=fit_predict(config,tr,va)
        scores.extend(score);labels.extend(va.winner.eq('A').astype(int))
    calibrator=LogisticRegression(C=1e6,max_iter=2000,random_state=SEED)
    calibrator.fit(np.asarray(scores).reshape(-1,1),labels)
    # La calibración no debe invertir el ranking elegido por AUC.
    assert calibrator.coef_[0,0]>0
    return calibrator,dict(method='sigmoid/Platt postselección',
                           train_cuts=cuts,validation_cuts=cuts[1:],n_oof=len(scores),
                           coefficient=float(calibrator.coef_[0,0]),
                           intercept=float(calibrator.intercept_[0]),
                           used_final_test_labels=False)


def attach_probability_metrics(metrics,proba,valid):
    y=valid.winner.eq('A').astype(int)
    metrics['brier']=float(brier_score_loss(y,proba))
    metrics['logloss']=float(log_loss(y,np.clip(proba,1e-15,1-1e-15)))
    metrics['probability_accuracy']=float(accuracy_score(y,proba>=.5))


def final():
    decision=select(); config=decision['config']
    output=OUT/'final_test.json'
    if output.exists():
        old=json.loads(output.read_text(encoding='utf-8'))
        assert old['selected_config']==config, 'La elección cambió después de abrir test'
        print('Test ya evaluado; se conserva.',flush=True)
        return
    # Desde aquí se permite leer 20/09; nunca se usa para volver a elegir.
    train=pd.concat([dataset(c,config['days']) for c in TRAIN_CUTS],ignore_index=True)
    if config['policy']=='sliding_1': train=train[train.corte==TRAIN_CUTS[-1]]
    test=dataset(TEST_CUT,config['days'])
    metrics,score,proba,predicted,artifact=fit_predict(config,train,test)
    dummy_config=dict(config,model='dummy',params={})
    dummy,*_=fit_predict(dummy_config,train,test)
    pd.DataFrame(dict(match_id=test.match_id,actual=test.winner.eq('A').astype(int),score=score,
                      probability=proba,predicted=predicted)).to_csv(OUT/'final_test_predictions.csv',index=False)
    calibration=[]
    if proba is not None:
        for low in np.arange(0,1,.1):
            mask=(proba>=low)&(proba<low+.1+ (1e-10 if low>.89 else 0))
            if mask.any(): calibration.append(dict(low=float(low),n=int(mask.sum()),predicted=float(proba[mask].mean()),observed=float(test.winner.eq('A').to_numpy()[mask].mean())))
    artifact.update(hero_ids=sorted(schema.HERO_IDS),k_shrinkage=100,
                    train_cuts=TRAIN_CUTS,source='experiments/training_strategy/selection.json')
    joblib.dump(artifact,OUT/'selected_model.joblib')
    write_json(output,dict(selected_config=config,metrics=metrics,dummy=dummy,calibration=calibration,
                          evaluated_at=datetime.now(timezone.utc).isoformat()))
    print('TEST',json.dumps(metrics),flush=True)

    # La combinación elegida por CV también se informa en octubre, sin reajuste.
    c=dict(config)
    tr_cuts=(TRAIN_CUTS+[TEST_CUT])[-1:] if c['policy']=='sliding_1' else TRAIN_CUTS+[TEST_CUT]
    tr=pd.concat([dataset(cut,c['days']) for cut in tr_cuts],ignore_index=True)
    m,*_=fit_predict(c,tr,dataset(STRESS_CUT,c['days']))
    write_json(OUT/'stress_selected.json',dict(config=c,metrics=m))


def posthoc_calibration():
    """Análisis adicional tras el primer test. No modifica selección ni evaluación original."""
    config=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['config']
    original=json.loads((OUT/'final_test.json').read_text(encoding='utf-8'))
    if hasattr(estimator(config['model'],config.get('params')),'predict_proba'):
        print('El elegido tiene probabilidades nativas; no se agrega calibración.',flush=True);return
    calibrator,info=temporal_calibrator(config,TRAIN_CUTS)
    saved=pd.read_csv(OUT/'final_test_predictions.csv')
    probability=calibrator.predict_proba(saved.score.to_numpy().reshape(-1,1))[:,1]
    test=dataset(TEST_CUT,config['days'])
    metrics=dict(original['metrics']);attach_probability_metrics(metrics,probability,test)
    bins=[]
    for low in np.arange(0,1,.1):
        mask=(probability>=low)&(probability<low+.1+(1e-10 if low>.89 else 0))
        if mask.any():bins.append(dict(low=float(low),n=int(mask.sum()),predicted=float(probability[mask].mean()),observed=float(test.winner.eq('A').to_numpy()[mask].mean())))
    saved['probability']=probability
    saved.to_csv(OUT/'final_test_predictions_calibrated_posthoc.csv',index=False)
    artifact=joblib.load(OUT/'selected_model.joblib');artifact['calibrator']=calibrator
    joblib.dump(artifact,OUT/'selected_model_calibrated_posthoc.joblib')
    # Para octubre, la calibración recibe únicamente scores temporales anteriores a octubre.
    calibrator,stress_info=temporal_calibrator(config,TRAIN_CUTS+[TEST_CUT])
    cuts=(TRAIN_CUTS+[TEST_CUT])[-1:] if config['policy']=='sliding_1' else TRAIN_CUTS+[TEST_CUT]
    train=pd.concat([dataset(c,config['days']) for c in cuts],ignore_index=True)
    valid=dataset(STRESS_CUT,config['days'])
    stress_metrics,score,*_=fit_predict(config,train,valid)
    attach_probability_metrics(stress_metrics,calibrator.predict_proba(score.reshape(-1,1))[:,1],valid)
    write_json(OUT/'calibration_posthoc.json',dict(config=config,
        performed_after_initial_test=True,created_at=datetime.now(timezone.utc).isoformat(),
        method_selected_without_comparing_test_calibrators=True,
        original_test_evaluated_at=original['evaluated_at'],test_fit=info,test_metrics=metrics,
        calibration_bins=bins,stress_fit=stress_info,stress_metrics=stress_metrics,
        original_test_sha256=sha256(OUT/'final_test.json')))
    print('Calibración post hoc guardada; evaluación original conservada.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['isolated','combinations','tune','final','calibrate','stress','summarize'],required=True)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='final': final();return
    if args.stage=='calibrate': posthoc_calibration();return
    if args.stage=='summarize': summarize();return
    if args.stage=='stress':
        assert (OUT/'final_test.json').exists(), 'F se ejecuta después de congelar selección y test final'
        configs=[dict(c,phase='stress',alternative='F',id='F|'+c['id']) for c in matrix()]
    elif args.stage=='tune': configs=tuning_configs()
    else: configs=[c for c in matrix() if c['phase']==args.stage]
    write_json(OUT/f'manifest_{args.stage}.json',configs)
    write_json(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),
               numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,seed=SEED,
               models={n:repr(estimator(n)) for n in MODELS},workers=args.workers,
               model_scope='17 clasificadores tabulares supervisados instalados; RBF aproximado con 128 bases, no SVC exacto; no enumeración infinita de modelos/hiperparámetros.'))
    execute(configs,args.workers)


if __name__=='__main__': main()
