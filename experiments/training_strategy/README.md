# Experimentos de entrenamiento

La ejecución se divide en alternativas aisladas, combinaciones, ajuste, selección/test final y estrés temporal en octubre. Los resultados observados se escriben en `results.jsonl`; los manifiestos registran las configuraciones completas. No se reutilizan las métricas del notebook preliminar.

## Protocolo

- Entrenamiento/validación: 20/08 → 27/08; 20/08+27/08 → 03/09; 20/08+27/08+03/09 → 10/09. En ventana deslizante se conserva solo el último corte anterior.
- Test final reservado: 20/09, exclusivamente elegido + Dummy. El historial del notebook ya contenía evaluaciones de este período; es un holdout de esta nueva búsqueda, no un conjunto nunca observado por el proyecto.
- F: entrenamiento anterior a octubre, incorporando 20/09, y evaluación en 19/10. Comparación exploratoria de la matriz completa; octubre no cambia la selección.
- Datos ranked/normal, 10.000 partidas crudas por corte, tres días como ventana permitida, `min_matches_stats=1`, `k=100`, ventanas de estadísticas de 7/14/20 días. Se informa la cobertura efectiva del muestreo ascendente por ID; no se afirma que cubra íntegramente tres días.
- Estadísticas terminan en T−1 segundo; partidas comienzan en T. Se verifica que no haya IDs repetidos ni filas fuera de la ventana.
- `StandardScaler` se ajusta exclusivamente en entrenamiento. Con decaimiento también recibe los pesos de entrenamiento. Los pesos se normalizan a media 1 para conservar la escala de regularización; se registra tamaño efectivo.
- Métrica principal: media de AUC entre los tres pasos, ponderados por igual. Desvío muestral, `ddof=1`. Accuracy usa `predict`, sin ajustar umbral con validación ni test. Se conservan Brier y log-loss cuando hay `predict_proba`; Ridge y SVM lineal se evalúan con `decision_function` para AUC.
- Comparador de simplicidad fijado: logística A1, 14 días, creciente, C=0,01 y escala estándar. Una alternativa más compleja se elige solo si su AUC medio supera la media de este comparador más su desvío. Si ninguna lo logra, se elige A1 logística y su ventana/C por CV. Empates de AUC dentro de la misma estructura A1 (tolerancia 1e-12) se resuelven por menor Brier de CV: un reescalamiento monótono puede mantener el orden y cambiar la calidad de las probabilidades. Esta regla secundaria se explicitó tras los casos aislados y antes del ajuste y de abrir el test. Esto es una regla de decisión, no una prueba de significancia.
- Las configuraciones de cada familia están en `environment.json`. Semilla 42. No se pretende enumerar todos los algoritmos o hiperparámetros existentes. Se cubren 17 clasificadores tabulares instalados, las cuatro familias requeridas y varios controles adicionales. El kernel RBF es aproximado mediante Nystroem, con 128 bases.
- KNN, LDA y QDA no admiten `sample_weight`: C y sus combinaciones se registran como incompatibles para esas familias. No se simula ponderación con otro experimento de remuestreo.
- E se ejecuta aunque D empate, por el pedido de probar las alternativas: análisis exploratorio adicional. Tendencia = ventaja reciente de 7 días menos ventaja de los 7 días anteriores. El segundo bloque se reconstruye restando **conteos crudos** de las ventanas de 14 y 7 días y aplicando luego shrinkage. No es `ventaja_7 - ventaja_14`.
- Días desde el parche no se incluye: el documento lo excluye y solo hay cuatro cortes de desarrollo para estimar un efecto por corte.
- F utiliza un catálogo uniforme de 44 héroes, verificado en la API el 26/10/2026. El corte 19/10 tiene estadísticas para 43; Nurse Harrow (87) queda neutral si no tiene historia. Los seis héroes ausentes en agosto/septiembre tienen composición cero en esos cortes; por eso las 72 features actuales equivalen a las 66 históricas más seis columnas constantes.
- Primero se ejecutan los 187 casos aislados (incluidas incompatibilidades), luego las 1.037 combinaciones restantes. En total: 1.224 configuraciones de la matriz, antes del ajuste. F repite esa matriz con un único salto a octubre.
- No se agregan datos sintéticos, no se mezclan partidas unranked, no se incluye rango/badge, fecha, identificador ni resultado posterior como predictor.
- El ganador SVM no produce probabilidades nativas. La evaluación automática del test terminó antes de incorporar Platt y se conserva sin cambios. Se agrega por separado una extensión **post hoc**: sigmoide sobre scores fuera de muestra de los pasos temporales de desarrollo. No se ajusta con etiquetas de test ni cambia AUC/ranking o selección; tampoco se comparan métodos de calibración usando test. El Brier/log-loss posterior se distingue de la primera evaluación; accuracy conserva el umbral del SVM y se informa aparte el de probabilidad 0,5. Para octubre, la sigmoide usa pasos que terminan antes de octubre. La extensión se decidió con la primera evaluación ya disponible: no se presenta como preespecificada ni congelada antes de ese primer test.

## Reproducción

Desde la raíz, con Python y `experiments/training_strategy/requirements.txt` instalados (versiones efectivamente usadas):

```powershell
python scripts/prepare_training_experiments.py
python scripts/run_training_experiments.py --stage isolated --workers 4
python scripts/run_training_experiments.py --stage combinations --workers 4
python scripts/run_training_experiments.py --stage tune --workers 4
python scripts/run_training_experiments.py --stage final
python scripts/run_training_experiments.py --stage calibrate
python scripts/run_training_experiments.py --stage stress --workers 4
python scripts/check_training_experiments.py
python scripts/report_training_experiments.py
python -m pytest tests/test_training_strategy.py -q
```

Las descargas usan el mismo cliente y las mismas transformaciones canónicas que el DAG. Esta ejecución local no se presenta como una corrida del scheduler de Airflow. El DAG también se actualizó para particionar por ventana/k y verificar la procedencia del caché. Los datos y sus hashes se conservan para evitar depender de la retención de la API.

## Referencias

Documento de partida: [estrategia entregada](../../docs/entrega3-estrategia-entrenamiento-original.md). Contratos de endpoints: [Deadlock API OpenAPI](https://api.deadlock-api.com/openapi.json). Referencia de partición temporal: [TimeSeriesSplit de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Aquí los bloques se generan por corte completo, en lugar de cortar por cantidad de filas.
