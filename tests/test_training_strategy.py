"""Controles de fuga temporal, ponderación, caché y equivalencia de features."""
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'include'))
sys.path.insert(0,str(ROOT/'scripts'))
import run_training_experiments as runner
from deadlock import schema
from deadlock.experiment_data import vector_features, window_params
from deadlock.transform import build_match_row, parse_match


def test_stats_ends_strictly_before_match_window():
    p=window_params('2026-08-20',14)
    assert pd.Timestamp(p['max_unix_timestamp']+1,unit='s',tz='UTC')==pd.Timestamp('2026-08-20',tz='UTC')
    assert p['max_unix_timestamp']-p['min_unix_timestamp']+1==14*86400


def test_vector_features_matches_canonical_with_new_hero():
    ids=[i for i,_ in schema.HEROES]
    bases=[dict(match_id=1,match_date='2026-10-19',game_mode='Normal',match_mode='Ranked',
                average_badge=75.,average_badge_diff=None,winner='A',
                heroes_a=ids[:5]+[88],heroes_b=ids[5:11])]
    rng=np.random.default_rng(42)
    tables=dict(hero={h:rng.normal() for h in ids},
                synergy={frozenset((a,b)):rng.normal() for a in ids for b in ids if a!=b},
                counter={(a,b):{m:rng.normal() for m in schema.COUNTER_METRICS}
                         for a in ids for b in ids if a!=b})
    # Las dos implementaciones tienen caminos y agregaciones independientes.
    expected=build_match_row(bases[0],tables)
    actual=vector_features(bases,tables)
    np.testing.assert_allclose(actual[schema.FEATURE_COLUMNS].iloc[0],
                               [expected[c] for c in schema.FEATURE_COLUMNS],atol=1e-12)
    assert actual.hero_baba.iloc[0]==1


def test_repeated_heroes_rejected():
    raw=dict(match_id=1,start_time='2026-08-20 00:00:00',winning_team='Team0',
             players=[dict(team='Team0',hero_id=h) for h in [1,2,3,4,6,7]]+
                     [dict(team='Team1',hero_id=h) for h in [1,8,10,11,12,13]])
    assert parse_match(raw) is None


def test_decay_normalized_and_halves_by_cut():
    w=runner.weights_for(pd.Series(['a','a','b','b','c','c']),'decay_1')
    np.testing.assert_allclose(w.mean(),1.)
    np.testing.assert_allclose(w[0]/w[2],.5)
    np.testing.assert_allclose(w[2]/w[4],.5)
    assert w.sum()**2/(w@w)<len(w)


def test_walk_forward_never_uses_future_or_test(monkeypatch):
    observed=[]
    def fake_dataset(cut,days):
        assert cut not in (runner.TEST_CUT,runner.STRESS_CUT)
        return pd.DataFrame(dict(match_id=[cut],match_date=[cut],corte=[cut],winner=['A']))
    def fake_fit(config,train,valid):
        observed.append((list(train.corte),list(valid.corte)))
        return dict(auc=.5,accuracy=.5),None,None,None,None
    monkeypatch.setattr(runner,'dataset',fake_dataset)
    monkeypatch.setattr(runner,'fit_predict',fake_fit)
    c=dict(model='logistic',block='A1',days=14,policy='expanding',phase='isolated')
    r=runner.run_config(c)
    assert r['status']=='ok'
    assert observed==[(runner.TRAIN_CUTS[:i],[runner.TRAIN_CUTS[i]]) for i in range(1,4)]
    observed.clear()
    r=runner.run_config(dict(c,policy='sliding_1'))
    assert r['status']=='ok'
    assert observed==[([runner.TRAIN_CUTS[i-1]],[runner.TRAIN_CUTS[i]]) for i in range(1,4)]


def test_matrix_all_models_each_factor_and_no_duplicates():
    matrix=runner.matrix()
    assert len(matrix)==len(runner.MODELS)*len(runner.POLICIES)*len(runner.BLOCKS)*len(runner.WINDOWS)
    assert len({c['id'] for c in matrix})==len(matrix)
    for alt in ('A1','A2','B1','B2','C','D','E'):
        assert {c['model'] for c in matrix if c['alternative']==alt}==set(runner.MODELS)


def test_dag_cache_rejects_unknown_request_or_changed_bytes(tmp_path):
    # Función pura del DAG: permite comprobar el caché sin instalar Airflow en Windows.
    tree=ast.parse((ROOT/'dags/deadlock_ingest.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_bronce_stats_cacheado')
    namespace=dict(Path=Path,json=json,hashlib=hashlib)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'dag_cache','exec'),namespace)
    cache=namespace['_bronce_stats_cacheado']
    path=tmp_path/'hero_stats.json'; path.write_text('[{"hero_id":1}]')
    request={'params':{'days':14}}
    assert cache(path,request) is None
    meta=path.with_suffix('.request.json')
    meta.write_text(json.dumps(dict(request=request,sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    assert cache(path,request)==[{'hero_id':1}]
    assert cache(path,{'params':{'days':7}}) is None
    path.write_text('[{"hero_id":2}]')
    assert cache(path,request) is None
