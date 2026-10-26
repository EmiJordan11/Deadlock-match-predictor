"""Verifica trazabilidad y consistencia de una ejecución completa ya terminada."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_training_experiments as e
from deadlock.experiment_data import sha256,write_json


def main():
    out=e.OUT
    results=e.read_results()
    assert not any(r['status']=='error' for r in results)
    base=[r for r in results if r['config']['phase'] in ('isolated','combinations')]
    stress=[r for r in results if r['config']['phase']=='stress']
    assert len(base)==len(stress)==1224
    assert sum(r['status']=='ok' for r in base)==1116
    assert sum(r['status']=='unsupported' for r in base)==108
    for r in results:
        if r['status']!='ok':continue
        c=r['config'];is_stress=c['phase']=='stress'
        assert len(r['folds'])==(1 if is_stress else 3)
        for f in r['folds']:
            assert max(f['train_cuts'])<f['valid_cut']
            if not is_stress:assert e.TEST_CUT not in f['train_cuts']+[f['valid_cut']]
            else:assert f['valid_cut']==e.STRESS_CUT
            assert 0<=f['auc']<=1 and 0<=f['accuracy']<=1
            assert f['n_features']==len(e.columns(c['block']))
            assert f['effective_n']<=f['n_train']+1e-6
            if c['model']=='dummy':assert f['auc']==.5

    # Control negativo: cambiar estadísticas no altera la composición.
    index={(r['config']['model'],r['config']['block'],r['config']['days'],r['config']['policy']):r for r in base}
    for model in e.MODELS:
        for policy in e.POLICIES:
            reference=index[model,'composition',14,policy]
            if reference['status']!='ok':continue
            for days in (7,20):
                candidate=index[model,'composition',days,policy]
                np.testing.assert_allclose([f['auc'] for f in candidate['folds']],
                                           [f['auc'] for f in reference['folds']],atol=1e-12)

    # La predicción guardada coincide con el artefacto entregado.
    selected=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    test=json.loads((out/'final_test.json').read_text(encoding='utf-8'))
    assert selected['created_at']<test['evaluated_at']
    assert selected['config']==test['selected_config']
    artifact=joblib.load(out/'selected_model.joblib')
    data=e.dataset(e.TEST_CUT,selected['config']['days'])
    x=artifact['scaler'].transform(data[artifact['columns']].to_numpy())
    if artifact['kernel'] is not None:x=artifact['kernel'].transform(x)
    model=artifact['model']
    score=model.predict_proba(x)[:,1] if hasattr(model,'predict_proba') else model.decision_function(x)
    predictions=pd.read_csv(out/'final_test_predictions.csv')
    np.testing.assert_array_equal(data.match_id,predictions.match_id)
    np.testing.assert_allclose(score,predictions.score,atol=1e-12)
    assert len(predictions)==10000
    posthoc=json.loads((out/'calibration_posthoc.json').read_text(encoding='utf-8'))
    assert posthoc['performed_after_initial_test']
    assert posthoc['original_test_sha256']==sha256(out/'final_test.json')
    assert posthoc['test_fit']['train_cuts']==e.TRAIN_CUTS
    assert not posthoc['test_fit']['used_final_test_labels']
    assert max(posthoc['stress_fit']['validation_cuts'])<e.STRESS_CUT
    augmented=joblib.load(out/'selected_model_calibrated_posthoc.joblib')
    probabilities=augmented['calibrator'].predict_proba(score.reshape(-1,1))[:,1]
    calibrated=pd.read_csv(out/'final_test_predictions_calibrated_posthoc.csv')
    np.testing.assert_allclose(probabilities,calibrated.probability,atol=1e-12)

    checked=0
    for cut in e.TRAIN_CUTS+[e.TEST_CUT,e.STRESS_CUT]:
        for days in e.WINDOWS:
            folder=ROOT/'include/output/silver'/f'corte={cut}'/f'dias={days}'/'k=100'
            p=json.loads((folder/'parametros.json').read_text(encoding='utf-8'))
            assert p['features_sha256']==sha256(folder/'features.parquet')
            for source in p['stats_sources']:assert source['sha256']==sha256(ROOT/source['path'])
            for name in ('hero','counter','synergy'):
                source=ROOT/'include/output/bronze'/f'corte={cut}'/'stats'/f'dias={days}'/f'{name}_stats.json'
                meta=json.loads(source.with_suffix('.request.json').read_text(encoding='utf-8'))
                assert meta['sha256']==sha256(source)
                assert meta['request']['params']['match_mode']=='ranked'
            checked+=1
    report=dict(status='ok',cv_matrix=1224,cv_success=1116,cv_unsupported=108,
                stress_matrix=1224,validated_datasets=checked,
                final_predictions_match_artifact=True,selection_precedes_test=True,
                original_test_preserved_during_posthoc_calibration=True,
                composition_window_negative_control=True,no_future_in_training=True,
                source_hashes_match=True)
    write_json(out/'verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
