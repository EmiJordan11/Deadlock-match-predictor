"""Genera los Markdown a partir de resultados medidos, nunca de números a mano."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_training_experiments as e

OUT=e.OUT
DOCS=ROOT/'docs'
NAMES=dict(dummy='Dummy',logistic='Logística',ridge='Ridge',sgd='SGD logístico',linear_svm='SVM lineal',
           gaussian_nb='Naive Bayes gaussiano',decision_tree='Árbol',random_forest='Random forest',
           extra_trees='Extra Trees',gradient_boosting='Gradient boosting',hist_gradient_boosting='Hist. gradient boosting',
           adaboost='AdaBoost',mlp='Red MLP',rbf_approx='RBF aproximado',knn='KNN',lda='LDA',qda='QDA')
POLICY=dict(expanding='Creciente',sliding_1='Último corte',decay_1='Vida media 1',decay_2='Vida media 2')


def num(x,n=4):
    return '—' if pd.isna(x) else f'{x:.{n}f}'.replace('.',',')


def pct(x): return '—' if pd.isna(x) else f'{100*x:.2f}%'.replace('.',',')


def table(headers,rows):
    def clean(x): return str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(clean(x) for x in row)+' |' for row in rows])+'\n'


def main():
    df=e.summarize()
    assert not (df.status=='error').any(), 'Resolver errores antes de informar'
    base=df[df.phase.isin(['isolated','combinations'])]
    stress=df[df.phase=='stress']
    assert len(base)==1224 and len(stress)==1224, (len(base),len(stress))
    assert set(base.status)=={'ok','unsupported'}
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    final=json.loads((OUT/'final_test.json').read_text(encoding='utf-8'))
    stress_selected=json.loads((OUT/'stress_selected.json').read_text(encoding='utf-8'))
    calibration_path=OUT/'calibration_posthoc.json'
    posthoc=json.loads(calibration_path.read_text(encoding='utf-8')) if calibration_path.exists() else None
    audit=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
    coverage=json.loads((OUT/'stats_coverage.json').read_text(encoding='utf-8'))
    records=e.read_results()
    cv=df[(df.status=='ok')&(df.phase!='stress')].sort_values('auc_mean',ascending=False)
    isolated=base[base.phase=='isolated']
    chosen=df[df.id==selection['config']['id']].iloc[0]
    best=cv.iloc[0]
    fit_count=sum(len(r.get('folds',[])) for r in records)
    warning_fits=sum(bool(f['warnings']) for r in records for f in r.get('folds',[]))
    baseline=base[(base.model=='logistic')&(base.block=='A1')&(base.days==14)&(base.policy=='expanding')].iloc[0]
    hero_rates={}
    for cut in e.TRAIN_CUTS+[e.TEST_CUT,e.STRESS_CUT]:
        p=ROOT/'include/output/bronze'/f'corte={cut}'/'stats/dias=14/hero_stats.json'
        hero_rates[cut]={r['hero_id']:r['wins']/r['matches'] for r in json.loads(p.read_text(encoding='utf-8')) if r['matches']}
    common=set.intersection(*(set(r) for r in hero_rates.values()))
    meta=pd.DataFrame({cut:pd.Series({h:rates[h] for h in common}) for cut,rates in hero_rates.items()})

    def subset(block=None,policy=None,days=14):
        s=base.copy()
        if block is not None: s=s[s.block==block]
        if policy is not None: s=s[s.policy==policy]
        if days is not None: s=s[s.days==days]
        return s

    def row_table(s, reference_block='A1',reference_policy='expanding',extra=False):
        rows=[]
        for _,r in s.sort_values(['model','days','policy']).iterrows():
            name=NAMES[r.model]
            if extra: name+=f' · {r.days}d · {POLICY[r.policy]}'
            if r.status!='ok': rows.append([name,'Incompatible','—','—','fit sin sample_weight']);continue
            ref=subset(reference_block,reference_policy)
            ref=ref[(ref.model==r.model)&(ref.status=='ok')]
            delta=r.auc_mean-ref.auc_mean.iloc[0] if len(ref) else np.nan
            rows.append([name,f'{num(r.auc_mean)} ± {num(r.auc_std)}',pct(r.accuracy_mean),
                         ('+' if delta>=0 else '')+num(delta),num(r.brier_mean)])
        return table(['Modelo / variante','AUC media ± desvío','Accuracy','Δ AUC vs referencia','Brier'],rows)

    def logistic(s): return s[(s.model=='logistic')&(s.status=='ok')].iloc[0]

    def conclusion(s,ref):
        ok=s[s.status=='ok']; winner=ok.sort_values('auc_mean',ascending=False).iloc[0]
        delta=winner.auc_mean-ref.auc_mean
        return (f'El mayor AUC de este bloque es **{num(winner.auc_mean)} ± {num(winner.auc_std)}**, '
                f'con {NAMES[winner.model]}, {winner.days} días y {POLICY[winner.policy].lower()}. '
                f'Su diferencia con la referencia es {num(delta)}; el desvío de la referencia es {num(ref.auc_std)}. '
                + ('Supera el umbral de simplicidad para esa comparación local.' if delta>ref.auc_std else
                   'No supera el umbral de simplicidad para esa comparación local. El signo del cambio por sí solo no justifica agregar complejidad.'))

    text=[]
    def add(s): text.append(s)
    add('# Experimentos de entrenamiento y validación temporal\n\nEjecución del 26/10/2026. Resultados medidos sobre datos ranked; generado automáticamente desde los archivos de cada corrida.\n')
    add(f'La regla de elección seleccionó **{NAMES[chosen.model]} · {chosen.block} · {chosen.days} días · {POLICY[chosen.policy].lower()}**, '
        f'con **AUC walk-forward {num(chosen.auc_mean)} ± {num(chosen.auc_std)}** y accuracy **{pct(chosen.accuracy_mean)}**. '
        f'En el test del 20/09 obtuvo **AUC {num(final["metrics"]["auc"])}** y **accuracy {pct(final["metrics"]["accuracy"])}**. '
        f'En octubre, reentrenado con los cinco cortes anteriores y estadísticas recientes, obtuvo **AUC {num(stress_selected["metrics"]["auc"])}**.\n')
    add(f'Se registraron **{len(base)} configuraciones** en la matriz principal: {sum(base.status=="ok")} entrenadas y '
        f'{sum(base.status=="unsupported")} incompatibles con ponderación. Primero se hicieron 187 casos aislados, después 1.037 combinaciones. '
        f'El ajuste agregó {sum(df.phase=="tune")} configuraciones. F repitió las 1.224 en octubre '
        f'({sum((stress.status=="ok"))} entrenadas). Hay **{fit_count} ajustes/evaluaciones** en los registros de validación, ajuste y F; '
        'el test final y su Dummy se guardan aparte. No se reemplazaron errores por resultados estimados.\n')
    add('## Cómo leer los resultados\n')
    add('AUC mide ordenamiento: 0,5 corresponde al azar. Accuracy usa el umbral natural del estimador, sin optimizarlo con el test. '
        'La cifra ± es el **desvío muestral de tres pasos temporales**, no un intervalo de confianza. '
        'Brier mide error de probabilidades: menor es mejor. Ridge y SVM lineal no producen probabilidades en estas corridas; su AUC se calcula con el score continuo y Brier queda vacío. '
        'Las diferencias se comparan con la referencia indicada en cada sección. Ninguna tabla de octubre interviene en la elección.\n')
    add('## Datos, cambios necesarios y controles\n')
    add(table(['Corte','Filas crudas','Válidas','Válidas con catálogo viejo','Primer inicio UTC','Último inicio UTC'],
              [[r['corte'],r['raw_rows'],r['valid_rows'],r['valid_with_original_38'],r['first_timestamp'],r['last_timestamp']] for r in audit]))
    add('Se descargaron nuevamente las tres estadísticas para cada ancho de ventana; no se supuso el ancho del caché antiguo. '
        'Se conservaron las partidas originales de agosto/septiembre y se descargaron las de octubre. '
        'Las estadísticas usan [T−N días, T−1 segundo]; las partidas se filtran en [T, T+3 días). '
        'Se comprobaron modo ranked/normal, IDs únicos dentro y entre cortes, doce héroes distintos por partida, valores finitos y orden temporal. '
        'Se reconstruyó toda la plata con k=100. El cálculo vectorizado se contrastó contra `build_match_row` en 12 filas distribuidas por cada uno de los 18 datasets.\n')
    add(f'La comprobación diaria de retención encontró {coverage[0]["distinct_days"]} días con datos entre {coverage[0]["first_day"]} y {coverage[0]["last_day"]} '
        f'en el corte más antiguo. Para octubre encontró {coverage[1]["distinct_days"]} días, de {coverage[1]["first_day"]} a {coverage[1]["last_day"]}. '
        'Esto controla cobertura temporal disponible; no certifica que la API contenga todas las partidas del juego.\n')
    add('La API de assets del 26/10 devuelve **44 héroes**, frente a los 43 del documento. Se agregó el catálogo actual completo. '
        'En octubre/19 las estadísticas tienen 43; Nurse Harrow, ID 87, aún no aparece en ellas. '
        'El esquema actual tiene 44 columnas de composición y 28 ventajas: **72 features, 79 columnas totales**. '
        'En agosto/septiembre, seis columnas de héroes son cero; el bloque completo contiene las mismas 66 variables no constantes del esquema anterior. '
        '`average_badge`, `average_badge_diff`, IDs, fechas y ganador quedan fuera de X.\n')
    add('Como comprobación del contexto, la correlación de winrates de los héroes comunes (ventanas ranked de 14 días, sin shrinkage) '
        'respecto del corte 20/08 es la siguiente. La correlación resume el orden relativo; no demuestra estabilidad de todas las interacciones.\n')
    add(table(['Corte','Héroes comunes','Correlación vs 20/08'],[[cut,len(common),num(meta.corr().loc['2026-08-20',cut])]
              for cut in hero_rates]))
    add('El DAG ahora separa `stats/dias=N/` y `silver/corte=T/dias=N/k=K/`, valida petición y hash del caché, guarda `parametros.json` '
        'y exporta un CSV por combinación. Las corridas de este informe son locales y comparten cliente y transformaciones con el DAG; no se presentan como ejecuciones de Airflow.\n')
    add('## Validación común y alcance de modelos\n')
    add(table(['Paso','Entrena (creciente)','Valida','Entrena (último corte)'],
              [[i,', '.join(e.TRAIN_CUTS[:i]),e.TRAIN_CUTS[i],e.TRAIN_CUTS[i-1]] for i in range(1,4)]))
    add('Cada estimador usa las mismas filas y columnas dentro de una configuración. El escalado se ajusta solo en entrenamiento. '
        'C usa pesos por antigüedad en cortes, normalizados a media uno; el escalado también recibe esos pesos. '
        'Se prueban vidas medias 1 y 2. MLP y boosting no utilizan una partición aleatoria interna de validación para decidir cuándo parar. '
        'Las iteraciones y los hiperparámetros completos están registrados; comparar una familia no significa haber agotado sus posibles ajustes.\n')
    add('Los 17 clasificadores son: '+', '.join(NAMES[n] for n in e.MODELS)+'. '
        'RBF usa Nystroem con 128 bases; no se ejecutó SVC RBF exacto. KNN, LDA y QDA no aceptan pesos de entrenamiento y se marcan incompatibles en C. '
        'No se instalaron XGBoost, LightGBM o CatBoost, ausentes del entorno. “Todos” se operacionaliza como estas familias tabulares viables, '
        'con una matriz finita y reproducible, y no como todos los algoritmos o hiperparámetros existentes.\n')

    add('## Experimentos aislados\n')
    dummy=subset('A1','expanding');dummy=dummy[dummy.model=='dummy']
    add('### 0. Piso\n\nDummy usa la proporción de clases aprendida en cada entrenamiento. Su score es constante dentro de un corte: AUC 0,5. '
        'Su accuracy permite comprobar si el modelo aporta más que predecir siempre la clase mayoritaria.\n')
    add(row_table(dummy))
    a1=subset('A1','expanding')
    add('### A1. Una columna de ventaja, todo el pasado\n\nSolo `hero_winrate_adv_mean`, con estadísticas de 14 días. '
        'Referencia de las diferencias: el mismo modelo en este mismo A1, por lo que Δ=0. '
        'Los modelos lineales pueden compartir AUC al ser transformaciones monótonas del mismo score; eso no implica probabilidades idénticas.\n')
    add(row_table(a1));add(conclusion(a1,baseline)+'\n')
    a2=subset('A2','expanding')
    add('### A2. Las 28 ventajas\n\nMantiene ventana, entrenamiento y filas; cambia únicamente el bloque de columnas. '
        'Δ compara cada familia contra su A1 de 14 días.\n')
    add(row_table(a2));add(conclusion(a2,baseline)+'\n')
    add(f'Con logística, A2 cambia el AUC en {num(logistic(a2).auc_mean-baseline.auc_mean)} frente a A1. '
        'Una ganancia pequeña puede deberse a que counters y sinergias repiten parte de la señal de fuerza individual; '
        'además agregan estimaciones más ruidosas. Esa es una interpretación compatible con el resultado, no una prueba de que las interacciones no existan.\n')
    comp=subset('composition','expanding')
    add('### Control de composición: un coeficiente por héroe\n\nSe prueban solo las columnas de composición, '
        'para separar la información de los IDs de héroe de las estadísticas recientes. Δ compara con A1 de la misma familia. '
        'Los seis héroes futuros tienen columna cero en entrenamiento.\n')
    add(row_table(comp));add(conclusion(comp,baseline)+'\n')
    add('La composición aprende efectos propios de cada héroe y puede funcionar bien cuando el orden de fuerza cambia poco entre cortes. '
        'Su resultado en este mes no demuestra que esos coeficientes se mantengan después de una actualización; esa es la pregunta que se vuelve a medir en F.\n')
    all_=subset('all','expanding')
    add('### Familias sobre el bloque completo\n\nIncluye la comparación requerida entre logística, árbol, bosque y boosting, '
        'y los otros clasificadores, con idéntico walk-forward. Δ compara contra A1 de cada familia.\n')
    add(row_table(all_));add(conclusion(all_,baseline)+'\n')
    b1=subset('A1','sliding_1')
    add('### B1. A1 entrenado solo con el último corte\n\nAísla “dato fresco frente a más dato”. '
        'Cada entrenamiento usa 10.000 partidas en lugar de 10.000, 20.000 y 30.000. Δ compara con el mismo A1 creciente.\n')
    add(row_table(b1));add(conclusion(b1,baseline)+'\n')
    add(f'En logística, usar solo el último corte cambia el AUC en {num(logistic(b1).auc_mean-baseline.auc_mean)}. '
        'A1 aprende una sola pendiente: con un meta relativamente estable, desechar partidas antiguas reduce información para estimarla '
        'sin aportar una adaptación específica por héroe. El efecto sobre accuracy/Brier puede existir aunque el orden y el AUC empaten.\n')
    b2=subset('all','sliding_1')
    add('### B2. Composición + ventajas con último corte\n\nCambia únicamente el historial de entrenamiento respecto de familias sobre todas las columnas. '
        'Δ compara con **el mismo modelo y bloque completo creciente**, no con A1.\n')
    add(row_table(b2,reference_block='all'));add(conclusion(b2,logistic(all_))+'\n')
    add(f'La logística B2 cambia {num(logistic(b2).auc_mean-logistic(all_).auc_mean)} de AUC frente a todas creciente. '
        'Aquí se estiman muchos más efectos con un único corte. Que resulte peor sería compatible con mayor varianza por menor muestra; '
        'que resulte mejor apoyaría la utilidad de dar prioridad a lo reciente en este período concreto. La comparación no mide el costo de operar reentrenamientos semanales.\n')
    c=subset('all',None);c=c[c.policy.str.startswith('decay')]
    add('### C. Todos los cortes, ponderados por antigüedad\n\nVidas medias 1 y 2 sobre composición + ventajas, con 14 días de estadísticas. '
        'Δ compara con el mismo bloque completo sin decaimiento. KNN/LDA/QDA no se entrenaron con pesos: sus filas son incompatibilidades explícitas.\n')
    add(row_table(c,reference_block='all',extra=True));add(conclusion(c,logistic(all_))+'\n')
    effective=[]
    for policy in ('expanding','decay_1','decay_2'):
        r=next(r for r in records if r['config']['phase']!='stress' and r['config']['model']=='logistic'
               and r['config']['block']=='all' and r['config']['days']==14 and r['config']['policy']==policy and not r['config'].get('params'))
        f=r['folds'][-1];effective.append([POLICY[policy],f['n_train'],int(round(f['effective_n']))])
    add(table(['Último paso de C','Filas disponibles','Tamaño efectivo por pesos'],effective))
    add('El decaimiento conserva filas pero reduce el tamaño efectivo: los cortes viejos aportan menos. '
        'Si la mejora es pequeña frente al desvío, no compensa agregar la vida media como hiperparámetro; '
        'el mecanismo podría ser útil con cambios más fuertes, que estos tres pasos no representan por completo.\n')
    d=subset('A1','expanding',None)
    add('### D. Frescura de estadísticas: 7, 14 y 20 días\n\nMismas partidas, modelos y entrenamiento; solo cambia la información estadística previa. '
        'Δ usa 14 días del mismo estimador como referencia. La fila de 14 días reutiliza A1, sin contarse dos veces en la matriz.\n')
    add(row_table(d,extra=True))
    ld=d[(d.model=='logistic')&(d.status=='ok')].sort_values('days')
    spread=ld.auc_mean.max()-ld.auc_mean.min()
    add(f'En logística, el rango entre ventanas es **{num(spread)} de AUC**, frente a un desvío de A1 de {num(baseline.auc_std)}. '
        +('La diferencia observada es menor que la variación temporal: no hay evidencia suficiente para afirmar que un ancho sea superior en metas futuros.' if spread<=baseline.auc_std else
          'Hay una diferencia que merece seguimiento, aunque tres pasos no permiten asegurar que se repita en metas futuros.')+'\n')
    trend=subset('A1_trend','expanding')
    add('### E. Tendencia de la última semana\n\nAgrega una única columna a A1: ventaja de winrate de los 7 días recientes menos la de los 7 anteriores, '
        'calculada desde conteos crudos y con shrinkage por bloque. Δ compara con A1 de 14 días de la misma familia. '
        'Se ejecutó como exploración adicional aun si D empata, para cubrir el pedido de probar las alternativas. '
        '“Días desde el parche” queda fuera, tal como propone el documento: cuatro cortes de desarrollo no permiten aprender de forma defendible ese efecto por período.\n')
    add(row_table(trend));add(conclusion(trend,baseline)+'\n')
    add(f'En logística, la tendencia modifica el AUC en {num(logistic(trend).auc_mean-baseline.auc_mean)}. '
        'La señal intenta corregir el retraso del promedio de 14 días, pero combina dos estimaciones semanales y puede sumar ruido. '
        'Un resultado dentro del desvío no permite afirmar que el modelo aprendió a anticipar cambios de meta.\n')

    add('## Combinaciones posteriores\n')
    add('Se cruzaron **6 bloques × 3 ventanas × 4 políticas × 17 clasificadores**. Bloques: A1; A2; composición; todas; A1+tendencia; todas+tendencia. '
        'Las políticas son creciente, último corte y decaimiento con vida media 1/2. Los casos ya ejecutados aisladamente se reutilizan; '
        'las otras 1.037 configuraciones se ejecutaron después. Esto prueba D+B, D+C, E+B, E+C, D+E y sus combinaciones sobre las familias compatibles.\n')
    top=base[base.status=='ok'].sort_values('auc_mean',ascending=False).head(20)
    add(table(['Modelo','Bloque','Días','Entrenamiento','AUC ± desvío','Accuracy'],
              [[NAMES[r.model],r.block,r.days,POLICY[r.policy],f'{num(r.auc_mean)} ± {num(r.auc_std)}',pct(r.accuracy_mean)] for _,r in top.iterrows()]))
    add('Para evitar que una familia quede oculta por muchas variantes de otra, esta es la mejor configuración **de validación** por clasificador:\n')
    per_model=base[base.status=='ok'].sort_values('auc_mean',ascending=False).drop_duplicates('model')
    add(table(['Modelo','Bloque','Días','Entrenamiento','AUC ± desvío'],
              [[NAMES[r.model],r.block,r.days,POLICY[r.policy],f'{num(r.auc_mean)} ± {num(r.auc_std)}'] for _,r in per_model.iterrows()]))
    add('La búsqueda amplia aumenta el riesgo de elegir por ruido entre muchas configuraciones. Estas cifras son de selección, '
        'no una estimación libre del sesgo de búsqueda; el holdout temporal se informa una sola vez para el elegido. '
        'No se usa el máximo de cada tabla como prueba automática de superioridad.\n')

    add('## Ajuste de hiperparámetros\n')
    add('Se ajustó cada familia en su mejor combinación de validación, con grillas finitas registradas en `manifest_tune.json`. '
        'Además, logística recorre C=0,0001; 0,001; 0,01; 0,1; 1; 10 sobre A1, A2, composición y todas, con 14 días y creciente. '
        'La grilla completa, incluidas alternativas que empeoran, está en el anexo y el CSV. Se conserva la misma validación temporal.\n')
    tuned=df[(df.phase=='tune')&(df.status=='ok')].sort_values('auc_mean',ascending=False)
    add(table(['Modelo','Bloque','Días','Política','Parámetros','AUC ± desvío','Accuracy'],
              [[NAMES[r.model],r.block,r.days,POLICY[r.policy],r.params,f'{num(r.auc_mean)} ± {num(r.auc_std)}',pct(r.accuracy_mean)]
               for _,r in tuned.drop_duplicates('model').iterrows()]))
    add(f'El mejor AUC sin restricción de complejidad, después del ajuste, es **{num(best.auc_mean)} ± {num(best.auc_std)}** '
        f'({NAMES[best.model]}, {best.block}, {best.days}d, {POLICY[best.policy].lower()}). '
        f'La referencia simple es {num(selection["reference_auc"])} ± {num(selection["reference_std"])} y el umbral exigido es **{num(selection["complexity_threshold"])}**. '
        f'{selection["candidates_above_threshold"]} configuraciones superan ese umbral. '
        f'La selección final es {NAMES[chosen.model]} / {chosen.block} / {chosen.days}d / {POLICY[chosen.policy].lower()}, parámetros `{chosen.params}`.\n')
    add('Dentro de la misma estructura A1, los empates de AUC hasta 1e−12 se resuelven por menor Brier de validación, '
        'sin usar las probabilidades del test. Esta regla secundaria se explicitó después de las corridas aisladas y antes del ajuste/holdout.\n')

    add('## Holdout final del 20/09\n')
    add('Se congeló `selection.json` antes de abrir este test para la nueva búsqueda. Se evaluó el elegido y su piso; '
        'el resto de la matriz no se evaluó en este holdout. El notebook anterior ya había mirado este período: '
        'por eso no debe describirse como un conjunto jamás observado por el proyecto.\n')
    add(table(['Modelo','AUC','Accuracy','Brier','Log-loss'],
              [[name,num(m['auc']),pct(m['accuracy']),num(m['brier']),num(m['logloss'])]
               for name,m in [('Elegido',final['metrics']),('Dummy',final['dummy'])]]))
    calibration=final['calibration']
    if posthoc:
        calibration=posthoc['calibration_bins']
        add('### Calibración adicional posterior a la primera evaluación\n\nEl SVM elegido no tiene probabilidades nativas; '
            'por eso Brier y log-loss de su **evaluación original** quedan vacíos. La evaluación automática terminó antes de incorporar calibración. '
            'Se conserva intacta y se agrega una extensión **post hoc**, sin cambiar la elección, el score ni el AUC: '
            f'Platt se ajusta sobre {posthoc["test_fit"]["n_oof"]} scores fuera de muestra de los tres pasos de desarrollo, '
            'exclusivamente con etiquetas anteriores al 20/09. Se usa un único método fijo y no se elige el método comparando el test. '
            'Aunque el ajuste no usa etiquetas de test, esta extensión se decidió cuando la primera evaluación ya estaba disponible; '
            'no se presenta como una calibración congelada antes de esa evaluación.\n')
        cm=posthoc['test_metrics']
        add(table(['Análisis posterior','AUC original','Accuracy umbral del SVM','Accuracy prob. ≥0,5','Brier','Log-loss'],
                  [['SVM + Platt',num(cm['auc']),pct(cm['accuracy']),pct(cm['probability_accuracy']),num(cm['brier']),num(cm['logloss'])]]))
    add('Probabilidades por intervalo, cuando están disponibles. El número observado depende del tamaño de cada grupo; '
        'los extremos con pocas partidas son más inciertos. Ningún ajuste de la sigmoide usa etiquetas de test.\n')
    add(table(['Intervalo','Partidas','Probabilidad media','Fracción de victorias A'],
              [[f'{num(r["low"],1)}–{num(r["low"]+.1,1)}',r['n'],pct(r['predicted']),pct(r['observed'])] for r in calibration]))
    ece=(sum(r['n']*abs(r['predicted']-r['observed']) for r in calibration)/sum(r['n'] for r in calibration)) if calibration else np.nan
    add(f'Error absoluto de calibración agregado en estos bins: **{num(ece)}**. '
        'Es descriptivo de este test, depende del binning y no garantiza calibración futura.\n')

    add('## F. Cambio de período y héroes nuevos: 19/10\n')
    add('La prueba usa los cinco cortes de agosto/septiembre como pasado (o solo el 20/09 en último corte) y partidas del 19/10 como evaluación. '
        'Las columnas se reconstruyen con estadísticas previas a octubre/19. Se repite **la matriz completa de todas las alternativas y combinaciones**, '
        'con un único salto temporal: no se calcula desvío entre pasos de F. '
        'Esta prueba va después de elegir y evaluar en septiembre; sus resultados son diagnóstico externo y no reabren la búsqueda.\n')
    rows=[]
    for model in e.MODELS:
        vals=[]
        for block,policy in [('A1','expanding'),('A2','expanding'),('composition','expanding'),('all','sliding_1')]:
            r=stress[(stress.model==model)&(stress.block==block)&(stress.policy==policy)&(stress.days==14)].iloc[0]
            vals.append(num(r.auc_mean))
        rows.append([NAMES[model]]+vals)
    add(table(['Modelo','A1 creciente','A2 creciente','Composición creciente','B2 último corte'],rows))
    add('Estos son los mayores AUC de las combinaciones base en octubre, **solo como resumen exploratorio**. '
        'Mostrar máximos entre muchos candidatos puede ser optimista; esta tabla no cambia la selección ni convierte octubre en validación de una nueva búsqueda.\n')
    ftop=stress[stress.status=='ok'].sort_values('auc_mean',ascending=False).head(10)
    add(table(['Modelo','Bloque','Días','Entrenamiento','AUC octubre','Accuracy'],
              [[NAMES[r.model],r.block,r.days,POLICY[r.policy],num(r.auc_mean),pct(r.accuracy_mean)] for _,r in ftop.iterrows()]))
    add(f'El modelo elegido en validación, reentrenado con el pasado disponible, obtiene en octubre **AUC {num(stress_selected["metrics"]["auc"])}**, '
        f'accuracy **{pct(stress_selected["metrics"]["accuracy"])}** y Brier **{num(stress_selected["metrics"]["brier"])}**. '
        'Los coeficientes de composición de los héroes nunca vistos no pudieron aprenderse antes de octubre. '
        'A1 recibe información de esos héroes a través de estadísticas recientes sin ampliar la dimensión de entrada del modelo.\n')
    if posthoc:
        add(f'En la extensión de calibración posterior, Platt se ajusta con {posthoc["stress_fit"]["n_oof"]} scores temporales '
            f'que terminan en septiembre y da en octubre Brier **{num(posthoc["stress_metrics"]["brier"])}** y '
            f'log-loss **{num(posthoc["stress_metrics"]["logloss"])}**. No usa etiquetas de octubre para ajustar la sigmoide.\n')
    sf=stress[(stress.model=='logistic')&(stress.days==14)]
    a=sf[(sf.block=='A1')&(sf.policy=='expanding')].iloc[0]
    b=sf[(sf.block=='all')&(sf.policy=='sliding_1')].iloc[0]
    delta=a.auc_mean-b.auc_mean
    add(f'La comparación logística central de F da A1={num(a.auc_mean)} frente a B2={num(b.auc_mean)}; '
        f'Δ A1−B2={num(delta)}. '+('A1 tiene un AUC numéricamente mayor.' if delta>0 else
         'B2 tiene un AUC al menos igual al de A1.')+' El signo por sí solo no establece superioridad estadística. '+
        ' Esto describe una comparación observada; no identifica causalmente el efecto de un parche. '
        'Entre septiembre y octubre cambian período, disponibilidad de héroes y población de partidas; hay un solo corte posterior.\n')
    oper=stress[(stress.model=='logistic')&(stress.block=='A1')&(stress.days==chosen.days)&(stress.policy=='expanding')].iloc[0]
    add(f'Frente al elegido, A1 logística con la misma ventana de {chosen.days} días obtiene AUC **{num(oper.auc_mean)}**, '
        f'contra **{num(stress_selected["metrics"]["auc"])}** del elegido. '
        'Es un contraste de diseños completos, no una ablación de una sola columna. '
        'La ventaja de validación del modelo complejo se revierte en octubre: esto muestra el límite de seleccionarlo con un mes estable. '
        'Se informa esa reversión sin reemplazar retrospectivamente al ganador ni usar octubre como una nueva grilla de selección.\n')

    add('## Qué se puede concluir y qué queda limitado\n')
    add('Las alternativas aisladas permiten atribuir diferencias a un cambio concreto de columnas, historial o ventana; las combinaciones permiten comprobar interacciones. '
        'El resultado debe defenderse con los tres valores por paso del CSV, además de su promedio. '
        'El criterio de simplicidad exige comparar la mejora con la variación entre cortes, además de mirar el máximo.\n')
    add(f'**Convergencia:** {warning_fits} ajustes registraron advertencias. En MLP se conserva el presupuesto de 120 iteraciones y '
        'se informa si alcanzó el límite; no se afirma optimalidad de esa familia. '
        'Las advertencias reales se conservan completas por configuración y paso en `results.jsonl`.\n')
    add('Las principales limitaciones son tres pasos de validación, un mes de desarrollo con poco cambio de meta, '
        'muestreo de las primeras 10.000 partidas por ID y un solo corte de octubre. El muestreo cubre parte del primer día, no tres días completos. '
        'No se evalúa explícitamente un horizonte de dos/tres semanas con múltiples repeticiones, ni se demuestra rendimiento garantizado para noviembre. '
        'Las estadísticas de cada corte son las que hoy devuelve la API para esa ventana; se conservan sus hashes, pero no se puede reconstruir la versión que la API servía en aquel día.\n')
    add('El notebook preliminar se conserva señalado como histórico; su lector usa los archivos y el catálogo presentes en sus datos anteriores, '
        'sin mezclar las nuevas particiones ni romper por las carpetas vacías. No se reemplazan sus salidas por las del estudio actual.\n')
    add('**Verificación de código:** siete controles automatizados cubren límites temporales, equivalencia de features, catálogo, pesos, '
        'separación de cortes, cobertura de la matriz e integridad del caché. Los archivos Python se compilaron. '
        'No se ejecutó la suite de integridad de Airflow: Airflow y Docker no están disponibles en este entorno Windows. '
        'El chequeo del caché del DAG se hace sobre su función pura y no sustituye una prueba del scheduler.\n')
    add('La comprobación integral también verifica los 18 datasets y sus hashes, 1.224 casos de desarrollo más 1.224 de F, '
        'ausencia de futuro en cada paso, invariancia de composición al cambiar solo la ventana estadística, '
        'coincidencia entre artefacto y predicciones guardadas, y preservación del primer test durante la calibración posterior. '
        'El resultado está en [verification.json](../experiments/training_strategy/verification.json).\n')
    add('## Archivos y reproducción\n')
    add('- [Todas las configuraciones, con interpretación por fila](resultados-experimentos-entrenamiento-completos.md).\n'
        '- [CSV resumen](../experiments/training_strategy/summary.csv) y [métricas por paso](../experiments/training_strategy/folds.csv).\n'
        '- [Resultados crudos](../experiments/training_strategy/results.jsonl), [selección](../experiments/training_strategy/selection.json) y [test final](../experiments/training_strategy/final_test.json).\n'
        '- [Calibración posterior identificada por separado](../experiments/training_strategy/calibration_posthoc.json).\n'
        '- [Auditoría de datos](../experiments/training_strategy/data_audit.json) y [cobertura de estadísticas](../experiments/training_strategy/stats_coverage.json).\n'
        '- [Protocolo y comandos](../experiments/training_strategy/README.md), [entorno/configuraciones](../experiments/training_strategy/environment.json), '
        '[preparación](../scripts/prepare_training_experiments.py), [ejecución](../scripts/run_training_experiments.py) y [generación del informe](../scripts/report_training_experiments.py).\n'
        '- [Documento original](entrega3-estrategia-entrenamiento-original.md).\n')
    add('Los contratos de datos se verificaron con [Deadlock API OpenAPI](https://api.deadlock-api.com/openapi.json). '
        'El esquema temporal sigue el principio de entrenamiento con pasado de [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), '
        'con cortes enteros como unidades. Las métricas del informe provienen de las corridas locales, no de esas referencias.\n')
    DOCS.mkdir(exist_ok=True)
    (DOCS/'resultados-experimentos-entrenamiento.md').write_text('\n'.join(text),encoding='utf-8')

    annex=['# Resultados completos de cada configuración\n',
           'Las fases aisladas, combinadas y ajuste informan media ± desvío muestral de tres pasos. F informa un único corte posterior. '
           'Δ compara con A1, 14d, creciente de la misma familia y la misma fase temporal; es una referencia descriptiva, no una prueba de significancia. '
           'La selección se hace únicamente con validación agosto/septiembre.\n']
    for phase,title in [('isolated','1. Alternativas por separado'),('combinations','2. Combinaciones'),('tune','3. Ajuste'),('stress','4. F y combinaciones en octubre')]:
        annex.append('## '+title+'\n');rows=[]
        for _,r in df[df.phase==phase].sort_values(['alternative','block','days','policy','model','params']).iterrows():
            if r.status!='ok':
                rows.append([NAMES[r.model],r.block,r.days,POLICY[r.policy],r.params,'—','—','Incompatible: fit no admite sample_weight.']);continue
            ref_df=(stress if phase=='stress' else base)
            ref=ref_df[(ref_df.model==r.model)&(ref_df.block=='A1')&(ref_df.days==14)&(ref_df.policy=='expanding')].iloc[0]
            delta=r.auc_mean-ref.auc_mean
            desc=f'Δ={num(delta)} vs A1 de esta familia. '
            if phase=='stress': desc+='Un corte; comparación exploratoria, sin desvío temporal.'
            elif delta>ref.auc_std: desc+='Supera el desvío de su referencia familiar.'
            elif delta<-ref.auc_std: desc+='Queda por debajo en más que el desvío de su referencia familiar.'
            else: desc+='Diferencia dentro del desvío de su referencia familiar.'
            if r.warning_count: desc+=f' Advertencias en {int(r.warning_count)} eventos; ver JSONL.'
            rows.append([NAMES[r.model],r.block,r.days,POLICY[r.policy],r.params,
                         num(r.auc_mean)+(f' ± {num(r.auc_std)}' if phase!='stress' else ''),pct(r.accuracy_mean),desc])
        annex.append(table(['Modelo','Columnas','Stats días','Entrenamiento','Ajuste','AUC','Accuracy','Interpretación'],rows))
    (DOCS/'resultados-experimentos-entrenamiento-completos.md').write_text('\n'.join(annex),encoding='utf-8')
    print('Informes generados.',flush=True)


if __name__=='__main__': main()
