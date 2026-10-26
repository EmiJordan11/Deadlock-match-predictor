# Experimentos de entrenamiento y validación temporal

Ejecución del 26/10/2026. Resultados medidos sobre datos ranked; generado automáticamente desde los archivos de cada corrida.

La regla de elección seleccionó **SVM lineal · all_trend · 20 días · creciente**, con **AUC walk-forward 0,6162 ± 0,0034** y accuracy **58,36%**. En el test del 20/09 obtuvo **AUC 0,6104** y **accuracy 57,81%**. En octubre, reentrenado con los cinco cortes anteriores y estadísticas recientes, obtuvo **AUC 0,6091**.

Se registraron **1224 configuraciones** en la matriz principal: 1116 entrenadas y 108 incompatibles con ponderación. Primero se hicieron 187 casos aislados, después 1.037 combinaciones. El ajuste agregó 90 configuraciones. F repitió las 1.224 en octubre (1116 entrenadas). Hay **4734 ajustes/evaluaciones** en los registros de validación, ajuste y F; el test final y su Dummy se guardan aparte. No se reemplazaron errores por resultados estimados.

## Cómo leer los resultados

AUC mide ordenamiento: 0,5 corresponde al azar. Accuracy usa el umbral natural del estimador, sin optimizarlo con el test. La cifra ± es el **desvío muestral de tres pasos temporales**, no un intervalo de confianza. Brier mide error de probabilidades: menor es mejor. Ridge y SVM lineal no producen probabilidades en estas corridas; su AUC se calcula con el score continuo y Brier queda vacío. Las diferencias se comparan con la referencia indicada en cada sección. Ninguna tabla de octubre interviene en la elección.

## Datos, cambios necesarios y controles

| Corte | Filas crudas | Válidas | Válidas con catálogo viejo | Primer inicio UTC | Último inicio UTC |
| --- | --- | --- | --- | --- | --- |
| 2026-08-20 | 10000 | 10000 | 10000 | 2026-08-20 00:00:02 | 2026-08-20 16:38:16 |
| 2026-08-27 | 10000 | 10000 | 10000 | 2026-08-27 00:00:13 | 2026-08-27 16:31:27 |
| 2026-09-03 | 10000 | 10000 | 10000 | 2026-09-03 00:00:03 | 2026-09-03 17:14:12 |
| 2026-09-10 | 10000 | 10000 | 10000 | 2026-09-10 00:00:00 | 2026-09-10 18:07:02 |
| 2026-09-20 | 10000 | 10000 | 10000 | 2026-09-20 00:00:09 | 2026-09-20 16:56:19 |
| 2026-10-19 | 10000 | 9998 | 60 | 2026-10-19 00:00:03 | 2026-10-19 12:53:20 |

Se descargaron nuevamente las tres estadísticas para cada ancho de ventana; no se supuso el ancho del caché antiguo. Se conservaron las partidas originales de agosto/septiembre y se descargaron las de octubre. Las estadísticas usan [T−N días, T−1 segundo]; las partidas se filtran en [T, T+3 días). Se comprobaron modo ranked/normal, IDs únicos dentro y entre cortes, doce héroes distintos por partida, valores finitos y orden temporal. Se reconstruyó toda la plata con k=100. El cálculo vectorizado se contrastó contra `build_match_row` en 12 filas distribuidas por cada uno de los 18 datasets.

La comprobación diaria de retención encontró 20 días con datos entre 2026-07-31 y 2026-08-19 en el corte más antiguo. Para octubre encontró 20 días, de 2026-09-29 a 2026-10-18. Esto controla cobertura temporal disponible; no certifica que la API contenga todas las partidas del juego.

La API de assets del 26/10 devuelve **44 héroes**, frente a los 43 del documento. Se agregó el catálogo actual completo. En octubre/19 las estadísticas tienen 43; Nurse Harrow, ID 87, aún no aparece en ellas. El esquema actual tiene 44 columnas de composición y 28 ventajas: **72 features, 79 columnas totales**. En agosto/septiembre, seis columnas de héroes son cero; el bloque completo contiene las mismas 66 variables no constantes del esquema anterior. `average_badge`, `average_badge_diff`, IDs, fechas y ganador quedan fuera de X.

Como comprobación del contexto, la correlación de winrates de los héroes comunes (ventanas ranked de 14 días, sin shrinkage) respecto del corte 20/08 es la siguiente. La correlación resume el orden relativo; no demuestra estabilidad de todas las interacciones.

| Corte | Héroes comunes | Correlación vs 20/08 |
| --- | --- | --- |
| 2026-08-20 | 38 | 1,0000 |
| 2026-08-27 | 38 | 0,9795 |
| 2026-09-03 | 38 | 0,9769 |
| 2026-09-10 | 38 | 0,9756 |
| 2026-09-20 | 38 | 0,9707 |
| 2026-10-19 | 38 | 0,7813 |

El DAG ahora separa `stats/dias=N/` y `silver/corte=T/dias=N/k=K/`, valida petición y hash del caché, guarda `parametros.json` y exporta un CSV por combinación. Las corridas de este informe son locales y comparten cliente y transformaciones con el DAG; no se presentan como ejecuciones de Airflow.

## Validación común y alcance de modelos

| Paso | Entrena (creciente) | Valida | Entrena (último corte) |
| --- | --- | --- | --- |
| 1 | 2026-08-20 | 2026-08-27 | 2026-08-20 |
| 2 | 2026-08-20, 2026-08-27 | 2026-09-03 | 2026-08-27 |
| 3 | 2026-08-20, 2026-08-27, 2026-09-03 | 2026-09-10 | 2026-09-03 |

Cada estimador usa las mismas filas y columnas dentro de una configuración. El escalado se ajusta solo en entrenamiento. C usa pesos por antigüedad en cortes, normalizados a media uno; el escalado también recibe esos pesos. Se prueban vidas medias 1 y 2. MLP y boosting no utilizan una partición aleatoria interna de validación para decidir cuándo parar. Las iteraciones y los hiperparámetros completos están registrados; comparar una familia no significa haber agotado sus posibles ajustes.

Los 17 clasificadores son: Dummy, Logística, Ridge, SGD logístico, SVM lineal, Naive Bayes gaussiano, Árbol, Random forest, Extra Trees, Gradient boosting, Hist. gradient boosting, AdaBoost, Red MLP, RBF aproximado, KNN, LDA, QDA. RBF usa Nystroem con 128 bases; no se ejecutó SVC RBF exacto. KNN, LDA y QDA no aceptan pesos de entrenamiento y se marcan incompatibles en C. No se instalaron XGBoost, LightGBM o CatBoost, ausentes del entorno. “Todos” se operacionaliza como estas familias tabulares viables, con una matriz finita y reproducible, y no como todos los algoritmos o hiperparámetros existentes.

## Experimentos aislados

### 0. Piso

Dummy usa la proporción de clases aprendida en cada entrenamiento. Su score es constante dentro de un corte: AUC 0,5. Su accuracy permite comprobar si el modelo aporta más que predecir siempre la clase mayoritaria.

| Modelo / variante | AUC media ± desvío | Accuracy | Δ AUC vs referencia | Brier |
| --- | --- | --- | --- | --- |
| Dummy | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |

### A1. Una columna de ventaja, todo el pasado

Solo `hero_winrate_adv_mean`, con estadísticas de 14 días. Referencia de las diferencias: el mismo modelo en este mismo A1, por lo que Δ=0. Los modelos lineales pueden compartir AUC al ser transformaciones monótonas del mismo score; eso no implica probabilidades idénticas.

| Modelo / variante | AUC media ± desvío | Accuracy | Δ AUC vs referencia | Brier |
| --- | --- | --- | --- | --- |
| AdaBoost | 0,6111 ± 0,0026 | 57,80% | +0,0000 | 0,2655 |
| Árbol | 0,6086 ± 0,0024 | 57,70% | +0,0000 | 0,2408 |
| Dummy | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Extra Trees | 0,6129 ± 0,0025 | 57,79% | +0,0000 | 0,2420 |
| Naive Bayes gaussiano | 0,6136 ± 0,0023 | 57,92% | +0,0000 | 0,2400 |
| Gradient boosting | 0,6122 ± 0,0027 | 57,79% | +0,0000 | 0,2403 |
| Hist. gradient boosting | 0,6098 ± 0,0037 | 57,72% | +0,0000 | 0,2404 |
| KNN | 0,5559 ± 0,0036 | 54,29% | +0,0000 | 0,2713 |
| LDA | 0,6136 ± 0,0023 | 57,99% | +0,0000 | 0,2400 |
| SVM lineal | 0,6136 ± 0,0023 | 57,99% | +0,0000 | — |
| Logística | 0,6136 ± 0,0023 | 57,99% | +0,0000 | 0,2400 |
| Red MLP | 0,6136 ± 0,0023 | 58,06% | +0,0000 | 0,2401 |
| QDA | 0,6136 ± 0,0023 | 57,96% | +0,0000 | 0,2400 |
| Random forest | 0,6069 ± 0,0055 | 57,44% | +0,0000 | 0,2411 |
| RBF aproximado | 0,6134 ± 0,0023 | 58,04% | +0,0000 | 0,2401 |
| Ridge | 0,6136 ± 0,0023 | 57,98% | +0,0000 | — |
| SGD logístico | 0,6136 ± 0,0023 | 57,98% | +0,0000 | 0,2400 |

El mayor AUC de este bloque es **0,6136 ± 0,0023**, con Logística, 14 días y creciente. Su diferencia con la referencia es 0,0000; el desvío de la referencia es 0,0023. No supera el umbral de simplicidad para esa comparación local. El signo del cambio por sí solo no justifica agregar complejidad.

### A2. Las 28 ventajas

Mantiene ventana, entrenamiento y filas; cambia únicamente el bloque de columnas. Δ compara cada familia contra su A1 de 14 días.

| Modelo / variante | AUC media ± desvío | Accuracy | Δ AUC vs referencia | Brier |
| --- | --- | --- | --- | --- |
| AdaBoost | 0,6124 ± 0,0019 | 57,71% | +0,0013 | 0,2642 |
| Árbol | 0,6080 ± 0,0025 | 57,42% | -0,0005 | 0,2410 |
| Dummy | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Extra Trees | 0,6125 ± 0,0031 | 57,93% | -0,0003 | 0,2406 |
| Naive Bayes gaussiano | 0,6108 ± 0,0040 | 57,90% | -0,0027 | 0,3128 |
| Gradient boosting | 0,6128 ± 0,0033 | 57,74% | +0,0005 | 0,2402 |
| Hist. gradient boosting | 0,6104 ± 0,0042 | 57,69% | +0,0005 | 0,2405 |
| KNN | 0,6042 ± 0,0015 | 57,33% | +0,0483 | 0,2417 |
| LDA | 0,6123 ± 0,0056 | 58,10% | -0,0013 | 0,2402 |
| SVM lineal | 0,6120 ± 0,0056 | 57,93% | -0,0015 | — |
| Logística | 0,6118 ± 0,0044 | 58,03% | -0,0018 | 0,2403 |
| Red MLP | 0,6067 ± 0,0077 | 57,37% | -0,0069 | 0,2412 |
| QDA | 0,6100 ± 0,0041 | 57,87% | -0,0035 | 0,2468 |
| Random forest | 0,6112 ± 0,0032 | 57,82% | +0,0043 | 0,2404 |
| RBF aproximado | 0,6081 ± 0,0038 | 57,81% | -0,0054 | 0,2412 |
| Ridge | 0,6118 ± 0,0053 | 57,91% | -0,0018 | — |
| SGD logístico | 0,6117 ± 0,0044 | 58,08% | -0,0018 | 0,2403 |

El mayor AUC de este bloque es **0,6128 ± 0,0033**, con Gradient boosting, 14 días y creciente. Su diferencia con la referencia es -0,0008; el desvío de la referencia es 0,0023. No supera el umbral de simplicidad para esa comparación local. El signo del cambio por sí solo no justifica agregar complejidad.

Con logística, A2 cambia el AUC en -0,0018 frente a A1. Una ganancia pequeña puede deberse a que counters y sinergias repiten parte de la señal de fuerza individual; además agregan estimaciones más ruidosas. Esa es una interpretación compatible con el resultado, no una prueba de que las interacciones no existan.

### Control de composición: un coeficiente por héroe

Se prueban solo las columnas de composición, para separar la información de los IDs de héroe de las estadísticas recientes. Δ compara con A1 de la misma familia. Los seis héroes futuros tienen columna cero en entrenamiento.

| Modelo / variante | AUC media ± desvío | Accuracy | Δ AUC vs referencia | Brier |
| --- | --- | --- | --- | --- |
| AdaBoost | 0,5557 ± 0,0010 | 53,60% | -0,0554 | 0,2602 |
| Árbol | 0,5373 ± 0,0144 | 52,97% | -0,0713 | 0,2493 |
| Dummy | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Extra Trees | 0,5928 ± 0,0025 | 56,76% | -0,0201 | 0,2461 |
| Naive Bayes gaussiano | 0,6043 ± 0,0080 | 57,44% | -0,0092 | 0,2418 |
| Gradient boosting | 0,5770 ± 0,0017 | 55,70% | -0,0353 | 0,2467 |
| Hist. gradient boosting | 0,5954 ± 0,0028 | 56,84% | -0,0145 | 0,2436 |
| KNN | 0,5885 ± 0,0049 | 56,37% | +0,0325 | 0,2449 |
| LDA | 0,6153 ± 0,0035 | 58,14% | +0,0017 | 0,2396 |
| SVM lineal | 0,6153 ± 0,0035 | 58,10% | +0,0017 | — |
| Logística | 0,6153 ± 0,0034 | 58,12% | +0,0017 | 0,2396 |
| Red MLP | 0,6007 ± 0,0085 | 57,19% | -0,0129 | 0,2430 |
| QDA | 0,6009 ± 0,0095 | 57,13% | -0,0126 | 0,2427 |
| Random forest | 0,5918 ± 0,0022 | 56,48% | -0,0151 | 0,2458 |
| RBF aproximado | 0,6110 ± 0,0039 | 57,63% | -0,0024 | 0,2414 |
| Ridge | 0,6153 ± 0,0035 | 58,11% | +0,0017 | — |
| SGD logístico | 0,6152 ± 0,0035 | 58,17% | +0,0017 | 0,2396 |

El mayor AUC de este bloque es **0,6153 ± 0,0035**, con SVM lineal, 14 días y creciente. Su diferencia con la referencia es 0,0017; el desvío de la referencia es 0,0023. No supera el umbral de simplicidad para esa comparación local. El signo del cambio por sí solo no justifica agregar complejidad.

La composición aprende efectos propios de cada héroe y puede funcionar bien cuando el orden de fuerza cambia poco entre cortes. Su resultado en este mes no demuestra que esos coeficientes se mantengan después de una actualización; esa es la pregunta que se vuelve a medir en F.

### Familias sobre el bloque completo

Incluye la comparación requerida entre logística, árbol, bosque y boosting, y los otros clasificadores, con idéntico walk-forward. Δ compara contra A1 de cada familia.

| Modelo / variante | AUC media ± desvío | Accuracy | Δ AUC vs referencia | Brier |
| --- | --- | --- | --- | --- |
| AdaBoost | 0,6124 ± 0,0019 | 57,71% | +0,0013 | 0,2642 |
| Árbol | 0,6089 ± 0,0032 | 57,52% | +0,0004 | 0,2409 |
| Dummy | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Extra Trees | 0,6115 ± 0,0032 | 57,85% | -0,0014 | 0,2414 |
| Naive Bayes gaussiano | 0,6114 ± 0,0033 | 57,93% | -0,0021 | 0,3217 |
| Gradient boosting | 0,6133 ± 0,0028 | 57,95% | +0,0011 | 0,2401 |
| Hist. gradient boosting | 0,6108 ± 0,0032 | 57,89% | +0,0010 | 0,2404 |
| KNN | 0,6031 ± 0,0015 | 57,28% | +0,0472 | 0,2420 |
| LDA | 0,6144 ± 0,0064 | 58,01% | +0,0009 | 0,2398 |
| SVM lineal | 0,6145 ± 0,0066 | 58,00% | +0,0009 | — |
| Logística | 0,6144 ± 0,0053 | 58,15% | +0,0008 | 0,2398 |
| Red MLP | 0,5946 ± 0,0056 | 56,85% | -0,0190 | 0,2453 |
| QDA | 0,6052 ± 0,0074 | 57,38% | -0,0083 | 0,2531 |
| Random forest | 0,6116 ± 0,0031 | 57,80% | +0,0047 | 0,2404 |
| RBF aproximado | 0,6115 ± 0,0037 | 58,14% | -0,0019 | 0,2406 |
| Ridge | 0,6144 ± 0,0063 | 58,03% | +0,0009 | — |
| SGD logístico | 0,6144 ± 0,0051 | 58,13% | +0,0008 | 0,2398 |

El mayor AUC de este bloque es **0,6145 ± 0,0066**, con SVM lineal, 14 días y creciente. Su diferencia con la referencia es 0,0009; el desvío de la referencia es 0,0023. No supera el umbral de simplicidad para esa comparación local. El signo del cambio por sí solo no justifica agregar complejidad.

### B1. A1 entrenado solo con el último corte

Aísla “dato fresco frente a más dato”. Cada entrenamiento usa 10.000 partidas en lugar de 10.000, 20.000 y 30.000. Δ compara con el mismo A1 creciente.

| Modelo / variante | AUC media ± desvío | Accuracy | Δ AUC vs referencia | Brier |
| --- | --- | --- | --- | --- |
| AdaBoost | 0,6115 ± 0,0016 | 58,02% | +0,0004 | 0,2648 |
| Árbol | 0,6056 ± 0,0009 | 57,32% | -0,0029 | 0,2415 |
| Dummy | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Extra Trees | 0,6128 ± 0,0023 | 58,11% | -0,0001 | 0,2423 |
| Naive Bayes gaussiano | 0,6136 ± 0,0023 | 57,96% | +0,0000 | 0,2400 |
| Gradient boosting | 0,6115 ± 0,0014 | 57,81% | -0,0007 | 0,2403 |
| Hist. gradient boosting | 0,6094 ± 0,0023 | 57,54% | -0,0004 | 0,2406 |
| KNN | 0,5569 ± 0,0033 | 54,61% | +0,0010 | 0,2714 |
| LDA | 0,6136 ± 0,0023 | 58,00% | +0,0000 | 0,2399 |
| SVM lineal | 0,6136 ± 0,0023 | 58,00% | +0,0000 | — |
| Logística | 0,6136 ± 0,0023 | 58,01% | +0,0000 | 0,2400 |
| Red MLP | 0,6136 ± 0,0023 | 57,93% | +0,0000 | 0,2400 |
| QDA | 0,6136 ± 0,0023 | 58,00% | +0,0000 | 0,2400 |
| Random forest | 0,6021 ± 0,0016 | 57,00% | -0,0047 | 0,2422 |
| RBF aproximado | 0,6133 ± 0,0024 | 57,91% | -0,0001 | 0,2401 |
| Ridge | 0,6136 ± 0,0023 | 58,01% | +0,0000 | — |
| SGD logístico | 0,6136 ± 0,0023 | 58,01% | +0,0000 | 0,2400 |

El mayor AUC de este bloque es **0,6136 ± 0,0023**, con Logística, 14 días y último corte. Su diferencia con la referencia es 0,0000; el desvío de la referencia es 0,0023. No supera el umbral de simplicidad para esa comparación local. El signo del cambio por sí solo no justifica agregar complejidad.

En logística, usar solo el último corte cambia el AUC en 0,0000. A1 aprende una sola pendiente: con un meta relativamente estable, desechar partidas antiguas reduce información para estimarla sin aportar una adaptación específica por héroe. El efecto sobre accuracy/Brier puede existir aunque el orden y el AUC empaten.

### B2. Composición + ventajas con último corte

Cambia únicamente el historial de entrenamiento respecto de familias sobre todas las columnas. Δ compara con **el mismo modelo y bloque completo creciente**, no con A1.

| Modelo / variante | AUC media ± desvío | Accuracy | Δ AUC vs referencia | Brier |
| --- | --- | --- | --- | --- |
| AdaBoost | 0,6132 ± 0,0016 | 57,92% | +0,0007 | 0,2632 |
| Árbol | 0,6020 ± 0,0039 | 57,34% | -0,0069 | 0,2424 |
| Dummy | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Extra Trees | 0,6109 ± 0,0038 | 57,83% | -0,0006 | 0,2415 |
| Naive Bayes gaussiano | 0,6113 ± 0,0044 | 57,86% | -0,0001 | 0,3228 |
| Gradient boosting | 0,6124 ± 0,0016 | 58,19% | -0,0009 | 0,2403 |
| Hist. gradient boosting | 0,6082 ± 0,0020 | 57,84% | -0,0026 | 0,2409 |
| KNN | 0,6039 ± 0,0027 | 57,60% | +0,0007 | 0,2420 |
| LDA | 0,6103 ± 0,0038 | 57,80% | -0,0041 | 0,2406 |
| SVM lineal | 0,6101 ± 0,0038 | 57,82% | -0,0044 | — |
| Logística | 0,6113 ± 0,0038 | 57,89% | -0,0031 | 0,2404 |
| Red MLP | 0,5842 ± 0,0053 | 55,87% | -0,0103 | 0,2497 |
| QDA | 0,6006 ± 0,0035 | 56,93% | -0,0046 | 0,2569 |
| Random forest | 0,6100 ± 0,0026 | 57,97% | -0,0015 | 0,2407 |
| RBF aproximado | 0,6108 ± 0,0052 | 58,16% | -0,0007 | 0,2408 |
| Ridge | 0,6105 ± 0,0038 | 57,85% | -0,0040 | — |
| SGD logístico | 0,6114 ± 0,0037 | 57,90% | -0,0029 | 0,2404 |

El mayor AUC de este bloque es **0,6132 ± 0,0016**, con AdaBoost, 14 días y último corte. Su diferencia con la referencia es -0,0012; el desvío de la referencia es 0,0053. No supera el umbral de simplicidad para esa comparación local. El signo del cambio por sí solo no justifica agregar complejidad.

La logística B2 cambia -0,0031 de AUC frente a todas creciente. Aquí se estiman muchos más efectos con un único corte. Que resulte peor sería compatible con mayor varianza por menor muestra; que resulte mejor apoyaría la utilidad de dar prioridad a lo reciente en este período concreto. La comparación no mide el costo de operar reentrenamientos semanales.

### C. Todos los cortes, ponderados por antigüedad

Vidas medias 1 y 2 sobre composición + ventajas, con 14 días de estadísticas. Δ compara con el mismo bloque completo sin decaimiento. KNN/LDA/QDA no se entrenaron con pesos: sus filas son incompatibilidades explícitas.

| Modelo / variante | AUC media ± desvío | Accuracy | Δ AUC vs referencia | Brier |
| --- | --- | --- | --- | --- |
| AdaBoost · 14d · Vida media 1 | 0,6129 ± 0,0017 | 57,77% | +0,0005 | 0,2651 |
| AdaBoost · 14d · Vida media 2 | 0,6132 ± 0,0022 | 57,76% | +0,0008 | 0,2645 |
| Árbol · 14d · Vida media 1 | 0,6082 ± 0,0027 | 57,68% | -0,0007 | 0,2412 |
| Árbol · 14d · Vida media 2 | 0,6097 ± 0,0039 | 57,63% | +0,0008 | 0,2409 |
| Dummy · 14d · Vida media 1 | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Dummy · 14d · Vida media 2 | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Extra Trees · 14d · Vida media 1 | 0,6108 ± 0,0030 | 57,81% | -0,0007 | 0,2414 |
| Extra Trees · 14d · Vida media 2 | 0,6111 ± 0,0033 | 57,90% | -0,0004 | 0,2414 |
| Naive Bayes gaussiano · 14d · Vida media 1 | 0,6114 ± 0,0037 | 57,91% | -0,0000 | 0,3222 |
| Naive Bayes gaussiano · 14d · Vida media 2 | 0,6114 ± 0,0035 | 57,90% | -0,0000 | 0,3220 |
| Gradient boosting · 14d · Vida media 1 | 0,6131 ± 0,0021 | 58,02% | -0,0002 | 0,2402 |
| Gradient boosting · 14d · Vida media 2 | 0,6132 ± 0,0026 | 58,01% | -0,0001 | 0,2401 |
| Hist. gradient boosting · 14d · Vida media 1 | 0,6107 ± 0,0024 | 57,97% | -0,0001 | 0,2405 |
| Hist. gradient boosting · 14d · Vida media 2 | 0,6110 ± 0,0027 | 57,92% | +0,0002 | 0,2404 |
| KNN · 14d · Vida media 1 | Incompatible | — | — | fit sin sample_weight |
| KNN · 14d · Vida media 2 | Incompatible | — | — | fit sin sample_weight |
| LDA · 14d · Vida media 1 | Incompatible | — | — | fit sin sample_weight |
| LDA · 14d · Vida media 2 | Incompatible | — | — | fit sin sample_weight |
| SVM lineal · 14d · Vida media 1 | 0,6139 ± 0,0062 | 58,01% | -0,0005 | — |
| SVM lineal · 14d · Vida media 2 | 0,6143 ± 0,0065 | 58,07% | -0,0002 | — |
| Logística · 14d · Vida media 1 | 0,6139 ± 0,0050 | 58,25% | -0,0005 | 0,2399 |
| Logística · 14d · Vida media 2 | 0,6142 ± 0,0052 | 58,20% | -0,0002 | 0,2398 |
| Red MLP · 14d · Vida media 1 | 0,5911 ± 0,0031 | 56,49% | -0,0035 | 0,2462 |
| Red MLP · 14d · Vida media 2 | 0,5932 ± 0,0043 | 56,63% | -0,0013 | 0,2456 |
| QDA · 14d · Vida media 1 | Incompatible | — | — | fit sin sample_weight |
| QDA · 14d · Vida media 2 | Incompatible | — | — | fit sin sample_weight |
| Random forest · 14d · Vida media 1 | 0,6114 ± 0,0028 | 58,11% | -0,0002 | 0,2404 |
| Random forest · 14d · Vida media 2 | 0,6121 ± 0,0030 | 58,13% | +0,0005 | 0,2403 |
| RBF aproximado · 14d · Vida media 1 | 0,6116 ± 0,0045 | 58,22% | +0,0001 | 0,2406 |
| RBF aproximado · 14d · Vida media 2 | 0,6115 ± 0,0043 | 58,22% | +0,0001 | 0,2406 |
| Ridge · 14d · Vida media 1 | 0,6139 ± 0,0058 | 58,10% | -0,0006 | — |
| Ridge · 14d · Vida media 2 | 0,6142 ± 0,0061 | 58,11% | -0,0002 | — |
| SGD logístico · 14d · Vida media 1 | 0,6140 ± 0,0050 | 58,16% | -0,0004 | 0,2399 |
| SGD logístico · 14d · Vida media 2 | 0,6142 ± 0,0051 | 58,25% | -0,0001 | 0,2398 |

El mayor AUC de este bloque es **0,6143 ± 0,0065**, con SVM lineal, 14 días y vida media 2. Su diferencia con la referencia es -0,0001; el desvío de la referencia es 0,0053. No supera el umbral de simplicidad para esa comparación local. El signo del cambio por sí solo no justifica agregar complejidad.

| Último paso de C | Filas disponibles | Tamaño efectivo por pesos |
| --- | --- | --- |
| Creciente | 30000 | 30000 |
| Vida media 1 | 30000 | 23333 |
| Vida media 2 | 30000 | 27836 |

El decaimiento conserva filas pero reduce el tamaño efectivo: los cortes viejos aportan menos. Si la mejora es pequeña frente al desvío, no compensa agregar la vida media como hiperparámetro; el mecanismo podría ser útil con cambios más fuertes, que estos tres pasos no representan por completo.

### D. Frescura de estadísticas: 7, 14 y 20 días

Mismas partidas, modelos y entrenamiento; solo cambia la información estadística previa. Δ usa 14 días del mismo estimador como referencia. La fila de 14 días reutiliza A1, sin contarse dos veces en la matriz.

| Modelo / variante | AUC media ± desvío | Accuracy | Δ AUC vs referencia | Brier |
| --- | --- | --- | --- | --- |
| AdaBoost · 7d · Creciente | 0,6115 ± 0,0024 | 57,88% | +0,0004 | 0,2641 |
| AdaBoost · 14d · Creciente | 0,6111 ± 0,0026 | 57,80% | +0,0000 | 0,2655 |
| AdaBoost · 20d · Creciente | 0,6117 ± 0,0021 | 57,55% | +0,0007 | 0,2670 |
| Árbol · 7d · Creciente | 0,6059 ± 0,0068 | 57,43% | -0,0027 | 0,2411 |
| Árbol · 14d · Creciente | 0,6086 ± 0,0024 | 57,70% | +0,0000 | 0,2408 |
| Árbol · 20d · Creciente | 0,6086 ± 0,0025 | 57,32% | +0,0000 | 0,2408 |
| Dummy · 7d · Creciente | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Dummy · 14d · Creciente | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Dummy · 20d · Creciente | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Extra Trees · 7d · Creciente | 0,6134 ± 0,0030 | 57,94% | +0,0005 | 0,2422 |
| Extra Trees · 14d · Creciente | 0,6129 ± 0,0025 | 57,79% | +0,0000 | 0,2420 |
| Extra Trees · 20d · Creciente | 0,6133 ± 0,0007 | 57,80% | +0,0004 | 0,2419 |
| Naive Bayes gaussiano · 7d · Creciente | 0,6138 ± 0,0022 | 58,05% | +0,0003 | 0,2400 |
| Naive Bayes gaussiano · 14d · Creciente | 0,6136 ± 0,0023 | 57,92% | +0,0000 | 0,2400 |
| Naive Bayes gaussiano · 20d · Creciente | 0,6137 ± 0,0012 | 57,80% | +0,0001 | 0,2400 |
| Gradient boosting · 7d · Creciente | 0,6117 ± 0,0036 | 57,97% | -0,0005 | 0,2403 |
| Gradient boosting · 14d · Creciente | 0,6122 ± 0,0027 | 57,79% | +0,0000 | 0,2403 |
| Gradient boosting · 20d · Creciente | 0,6125 ± 0,0018 | 57,77% | +0,0003 | 0,2402 |
| Hist. gradient boosting · 7d · Creciente | 0,6109 ± 0,0040 | 57,87% | +0,0010 | 0,2405 |
| Hist. gradient boosting · 14d · Creciente | 0,6098 ± 0,0037 | 57,72% | +0,0000 | 0,2404 |
| Hist. gradient boosting · 20d · Creciente | 0,6110 ± 0,0016 | 57,57% | +0,0012 | 0,2404 |
| KNN · 7d · Creciente | 0,5582 ± 0,0046 | 54,41% | +0,0023 | 0,2707 |
| KNN · 14d · Creciente | 0,5559 ± 0,0036 | 54,29% | +0,0000 | 0,2713 |
| KNN · 20d · Creciente | 0,5584 ± 0,0010 | 54,62% | +0,0025 | 0,2711 |
| LDA · 7d · Creciente | 0,6138 ± 0,0022 | 58,07% | +0,0003 | 0,2399 |
| LDA · 14d · Creciente | 0,6136 ± 0,0023 | 57,99% | +0,0000 | 0,2400 |
| LDA · 20d · Creciente | 0,6137 ± 0,0012 | 57,87% | +0,0001 | 0,2399 |
| SVM lineal · 7d · Creciente | 0,6138 ± 0,0022 | 58,07% | +0,0003 | — |
| SVM lineal · 14d · Creciente | 0,6136 ± 0,0023 | 57,99% | +0,0000 | — |
| SVM lineal · 20d · Creciente | 0,6137 ± 0,0012 | 57,87% | +0,0001 | — |
| Logística · 7d · Creciente | 0,6138 ± 0,0022 | 58,08% | +0,0003 | 0,2400 |
| Logística · 14d · Creciente | 0,6136 ± 0,0023 | 57,99% | +0,0000 | 0,2400 |
| Logística · 20d · Creciente | 0,6137 ± 0,0012 | 57,86% | +0,0001 | 0,2400 |
| Red MLP · 7d · Creciente | 0,6138 ± 0,0022 | 58,01% | +0,0003 | 0,2400 |
| Red MLP · 14d · Creciente | 0,6136 ± 0,0023 | 58,06% | +0,0000 | 0,2401 |
| Red MLP · 20d · Creciente | 0,6137 ± 0,0012 | 57,86% | +0,0001 | 0,2400 |
| QDA · 7d · Creciente | 0,6138 ± 0,0022 | 58,13% | +0,0003 | 0,2400 |
| QDA · 14d · Creciente | 0,6136 ± 0,0023 | 57,96% | +0,0000 | 0,2400 |
| QDA · 20d · Creciente | 0,6137 ± 0,0012 | 57,82% | +0,0001 | 0,2400 |
| Random forest · 7d · Creciente | 0,6074 ± 0,0076 | 57,44% | +0,0006 | 0,2412 |
| Random forest · 14d · Creciente | 0,6069 ± 0,0055 | 57,44% | +0,0000 | 0,2411 |
| Random forest · 20d · Creciente | 0,6067 ± 0,0043 | 57,36% | -0,0002 | 0,2413 |
| RBF aproximado · 7d · Creciente | 0,6137 ± 0,0023 | 57,96% | +0,0003 | 0,2400 |
| RBF aproximado · 14d · Creciente | 0,6134 ± 0,0023 | 58,04% | +0,0000 | 0,2401 |
| RBF aproximado · 20d · Creciente | 0,6135 ± 0,0013 | 57,80% | +0,0001 | 0,2401 |
| Ridge · 7d · Creciente | 0,6138 ± 0,0022 | 58,07% | +0,0003 | — |
| Ridge · 14d · Creciente | 0,6136 ± 0,0023 | 57,98% | +0,0000 | — |
| Ridge · 20d · Creciente | 0,6137 ± 0,0012 | 57,87% | +0,0001 | — |
| SGD logístico · 7d · Creciente | 0,6138 ± 0,0022 | 58,07% | +0,0003 | 0,2400 |
| SGD logístico · 14d · Creciente | 0,6136 ± 0,0023 | 57,98% | +0,0000 | 0,2400 |
| SGD logístico · 20d · Creciente | 0,6137 ± 0,0012 | 57,86% | +0,0001 | 0,2400 |

En logística, el rango entre ventanas es **0,0003 de AUC**, frente a un desvío de A1 de 0,0023. La diferencia observada es menor que la variación temporal: no hay evidencia suficiente para afirmar que un ancho sea superior en metas futuros.

### E. Tendencia de la última semana

Agrega una única columna a A1: ventaja de winrate de los 7 días recientes menos la de los 7 anteriores, calculada desde conteos crudos y con shrinkage por bloque. Δ compara con A1 de 14 días de la misma familia. Se ejecutó como exploración adicional aun si D empata, para cubrir el pedido de probar las alternativas. “Días desde el parche” queda fuera, tal como propone el documento: cuatro cortes de desarrollo no permiten aprender de forma defendible ese efecto por período.

| Modelo / variante | AUC media ± desvío | Accuracy | Δ AUC vs referencia | Brier |
| --- | --- | --- | --- | --- |
| AdaBoost | 0,6111 ± 0,0026 | 57,80% | +0,0000 | 0,2655 |
| Árbol | 0,6087 ± 0,0030 | 57,59% | +0,0002 | 0,2407 |
| Dummy | 0,5000 ± 0,0000 | 50,12% | +0,0000 | 0,2500 |
| Extra Trees | 0,6139 ± 0,0020 | 58,00% | +0,0010 | 0,2449 |
| Naive Bayes gaussiano | 0,6139 ± 0,0024 | 57,98% | +0,0004 | 0,2399 |
| Gradient boosting | 0,6127 ± 0,0028 | 57,78% | +0,0005 | 0,2403 |
| Hist. gradient boosting | 0,6121 ± 0,0040 | 57,83% | +0,0023 | 0,2403 |
| KNN | 0,6045 ± 0,0040 | 57,37% | +0,0486 | 0,2417 |
| LDA | 0,6138 ± 0,0023 | 57,99% | +0,0003 | 0,2399 |
| SVM lineal | 0,6138 ± 0,0023 | 58,00% | +0,0003 | — |
| Logística | 0,6138 ± 0,0023 | 57,99% | +0,0003 | 0,2400 |
| Red MLP | 0,6138 ± 0,0024 | 57,93% | +0,0003 | 0,2400 |
| QDA | 0,6139 ± 0,0023 | 58,00% | +0,0003 | 0,2400 |
| Random forest | 0,6111 ± 0,0045 | 57,95% | +0,0042 | 0,2405 |
| RBF aproximado | 0,6135 ± 0,0027 | 58,10% | +0,0001 | 0,2400 |
| Ridge | 0,6138 ± 0,0023 | 58,00% | +0,0003 | — |
| SGD logístico | 0,6138 ± 0,0023 | 57,99% | +0,0003 | 0,2400 |

El mayor AUC de este bloque es **0,6139 ± 0,0024**, con Naive Bayes gaussiano, 14 días y creciente. Su diferencia con la referencia es 0,0004; el desvío de la referencia es 0,0023. No supera el umbral de simplicidad para esa comparación local. El signo del cambio por sí solo no justifica agregar complejidad.

En logística, la tendencia modifica el AUC en 0,0003. La señal intenta corregir el retraso del promedio de 14 días, pero combina dos estimaciones semanales y puede sumar ruido. Un resultado dentro del desvío no permite afirmar que el modelo aprendió a anticipar cambios de meta.

## Combinaciones posteriores

Se cruzaron **6 bloques × 3 ventanas × 4 políticas × 17 clasificadores**. Bloques: A1; A2; composición; todas; A1+tendencia; todas+tendencia. Las políticas son creciente, último corte y decaimiento con vida media 1/2. Los casos ya ejecutados aisladamente se reutilizan; las otras 1.037 configuraciones se ejecutaron después. Esto prueba D+B, D+C, E+B, E+C, D+E y sus combinaciones sobre las familias compatibles.

| Modelo | Bloque | Días | Entrenamiento | AUC ± desvío | Accuracy |
| --- | --- | --- | --- | --- | --- |
| SVM lineal | all_trend | 20 | Creciente | 0,6162 ± 0,0040 | 58,27% |
| LDA | all_trend | 20 | Creciente | 0,6160 ± 0,0041 | 58,23% |
| SVM lineal | all_trend | 20 | Vida media 2 | 0,6159 ± 0,0040 | 58,29% |
| Ridge | all_trend | 20 | Creciente | 0,6159 ± 0,0041 | 58,16% |
| Ridge | all_trend | 20 | Vida media 2 | 0,6157 ± 0,0040 | 58,24% |
| Logística | all_trend | 20 | Creciente | 0,6156 ± 0,0038 | 58,23% |
| SVM lineal | all | 20 | Creciente | 0,6156 ± 0,0048 | 58,35% |
| SGD logístico | all_trend | 20 | Creciente | 0,6155 ± 0,0036 | 58,19% |
| SVM lineal | all_trend | 20 | Vida media 1 | 0,6155 ± 0,0037 | 58,34% |
| SVM lineal | all | 7 | Creciente | 0,6155 ± 0,0051 | 58,22% |
| LDA | all | 20 | Creciente | 0,6155 ± 0,0047 | 58,28% |
| SGD logístico | all_trend | 20 | Vida media 2 | 0,6155 ± 0,0037 | 58,15% |
| Logística | all_trend | 20 | Vida media 2 | 0,6155 ± 0,0038 | 58,22% |
| SVM lineal | all | 20 | Vida media 2 | 0,6155 ± 0,0047 | 58,26% |
| LDA | all | 7 | Creciente | 0,6154 ± 0,0050 | 58,22% |
| Ridge | all | 20 | Creciente | 0,6154 ± 0,0046 | 58,25% |
| Ridge | all_trend | 20 | Vida media 1 | 0,6154 ± 0,0038 | 58,20% |
| Logística | composition | 20 | Vida media 2 | 0,6153 ± 0,0036 | 58,19% |
| Logística | composition | 14 | Vida media 2 | 0,6153 ± 0,0036 | 58,19% |
| Logística | composition | 7 | Vida media 2 | 0,6153 ± 0,0036 | 58,19% |

Para evitar que una familia quede oculta por muchas variantes de otra, esta es la mejor configuración **de validación** por clasificador:

| Modelo | Bloque | Días | Entrenamiento | AUC ± desvío |
| --- | --- | --- | --- | --- |
| SVM lineal | all_trend | 20 | Creciente | 0,6162 ± 0,0040 |
| LDA | all_trend | 20 | Creciente | 0,6160 ± 0,0041 |
| Ridge | all_trend | 20 | Creciente | 0,6159 ± 0,0041 |
| Logística | all_trend | 20 | Creciente | 0,6156 ± 0,0038 |
| SGD logístico | all_trend | 20 | Creciente | 0,6155 ± 0,0036 |
| Naive Bayes gaussiano | A1_trend | 20 | Vida media 2 | 0,6141 ± 0,0013 |
| QDA | A1_trend | 20 | Creciente | 0,6141 ± 0,0013 |
| Red MLP | A1_trend | 20 | Creciente | 0,6140 ± 0,0013 |
| Extra Trees | A1_trend | 20 | Vida media 2 | 0,6140 ± 0,0014 |
| RBF aproximado | A1_trend | 20 | Creciente | 0,6138 ± 0,0018 |
| Gradient boosting | all | 20 | Creciente | 0,6138 ± 0,0024 |
| AdaBoost | A2 | 14 | Vida media 2 | 0,6132 ± 0,0022 |
| Random forest | all | 20 | Vida media 1 | 0,6128 ± 0,0027 |
| Hist. gradient boosting | A1_trend | 20 | Vida media 2 | 0,6127 ± 0,0021 |
| Árbol | A1_trend | 20 | Creciente | 0,6102 ± 0,0024 |
| KNN | A1_trend | 14 | Último corte | 0,6057 ± 0,0038 |
| Dummy | A1 | 14 | Vida media 1 | 0,5000 ± 0,0000 |

La búsqueda amplia aumenta el riesgo de elegir por ruido entre muchas configuraciones. Estas cifras son de selección, no una estimación libre del sesgo de búsqueda; el holdout temporal se informa una sola vez para el elegido. No se usa el máximo de cada tabla como prueba automática de superioridad.

## Ajuste de hiperparámetros

Se ajustó cada familia en su mejor combinación de validación, con grillas finitas registradas en `manifest_tune.json`. Además, logística recorre C=0,0001; 0,001; 0,01; 0,1; 1; 10 sobre A1, A2, composición y todas, con 14 días y creciente. La grilla completa, incluidas alternativas que empeoran, está en el anexo y el CSV. Se conserva la misma validación temporal.

| Modelo | Bloque | Días | Política | Parámetros | AUC ± desvío | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| SVM lineal | all_trend | 20 | Creciente | {"C": 0.1} | 0,6162 ± 0,0034 | 58,36% |
| Logística | all_trend | 20 | Creciente | {"C": 0.1} | 0,6162 ± 0,0040 | 58,26% |
| SGD logístico | all_trend | 20 | Creciente | {"alpha": 0.001} | 0,6161 ± 0,0037 | 58,25% |
| LDA | all_trend | 20 | Creciente | {"shrinkage": "auto"} | 0,6160 ± 0,0041 | 58,23% |
| Ridge | all_trend | 20 | Creciente | {"alpha": 100.0} | 0,6159 ± 0,0041 | 58,16% |
| Naive Bayes gaussiano | A1_trend | 20 | Vida media 2 | {"var_smoothing": 1e-05} | 0,6141 ± 0,0013 | 57,96% |
| QDA | A1_trend | 20 | Creciente | {"reg_param": 0.9} | 0,6141 ± 0,0012 | 57,93% |
| Red MLP | A1_trend | 20 | Creciente | {"alpha": 10.0, "hidden_layer_sizes": [32]} | 0,6141 ± 0,0012 | 58,01% |
| Extra Trees | A1_trend | 20 | Vida media 2 | {"max_depth": 4, "min_samples_leaf": 100} | 0,6140 ± 0,0014 | 58,03% |
| Gradient boosting | all | 20 | Creciente | {"max_depth": 1, "n_estimators": 120} | 0,6140 ± 0,0020 | 57,93% |
| RBF aproximado | A1_trend | 20 | Creciente | {"C": 0.1} | 0,6138 ± 0,0018 | 58,07% |
| Hist. gradient boosting | A1_trend | 20 | Vida media 2 | {"l2_regularization": 10.0, "max_leaf_nodes": 3} | 0,6134 ± 0,0010 | 57,83% |
| AdaBoost | A2 | 14 | Vida media 2 | {"learning_rate": 0.05, "n_estimators": 100} | 0,6134 ± 0,0023 | 57,75% |
| Random forest | all | 20 | Vida media 1 | {"max_depth": 4, "min_samples_leaf": 100} | 0,6132 ± 0,0021 | 57,87% |
| KNN | A1_trend | 14 | Último corte | {"n_neighbors": 1001} | 0,6111 ± 0,0025 | 57,85% |
| Árbol | A1_trend | 20 | Creciente | {"max_depth": 4, "min_samples_leaf": 200} | 0,6107 ± 0,0019 | 57,45% |

El mejor AUC sin restricción de complejidad, después del ajuste, es **0,6162 ± 0,0034** (SVM lineal, all_trend, 20d, creciente). La referencia simple es 0,6136 ± 0,0023 y el umbral exigido es **0,6159**. 11 configuraciones superan ese umbral. La selección final es SVM lineal / all_trend / 20d / creciente, parámetros `{"C": 0.1}`.

Dentro de la misma estructura A1, los empates de AUC hasta 1e−12 se resuelven por menor Brier de validación, sin usar las probabilidades del test. Esta regla secundaria se explicitó después de las corridas aisladas y antes del ajuste/holdout.

## Holdout final del 20/09

Se congeló `selection.json` antes de abrir este test para la nueva búsqueda. Se evaluó el elegido y su piso; el resto de la matriz no se evaluó en este holdout. El notebook anterior ya había mirado este período: por eso no debe describirse como un conjunto jamás observado por el proyecto.

| Modelo | AUC | Accuracy | Brier | Log-loss |
| --- | --- | --- | --- | --- |
| Elegido | 0,6104 | 57,81% | — | — |
| Dummy | 0,5000 | 50,21% | 0,2500 | 0,6932 |

### Calibración adicional posterior a la primera evaluación

El SVM elegido no tiene probabilidades nativas; por eso Brier y log-loss de su **evaluación original** quedan vacíos. La evaluación automática terminó antes de incorporar calibración. Se conserva intacta y se agrega una extensión **post hoc**, sin cambiar la elección, el score ni el AUC: Platt se ajusta sobre 30000 scores fuera de muestra de los tres pasos de desarrollo, exclusivamente con etiquetas anteriores al 20/09. Se usa un único método fijo y no se elige el método comparando el test. Aunque el ajuste no usa etiquetas de test, esta extensión se decidió cuando la primera evaluación ya estaba disponible; no se presenta como una calibración congelada antes de esa evaluación.

| Análisis posterior | AUC original | Accuracy umbral del SVM | Accuracy prob. ≥0,5 | Brier | Log-loss |
| --- | --- | --- | --- | --- | --- |
| SVM + Platt | 0,6104 | 57,81% | 57,75% | 0,2406 | 0,6739 |

Probabilidades por intervalo, cuando están disponibles. El número observado depende del tamaño de cada grupo; los extremos con pocas partidas son más inciertos. Ningún ajuste de la sigmoide usa etiquetas de test.

| Intervalo | Partidas | Probabilidad media | Fracción de victorias A |
| --- | --- | --- | --- |
| 0,1–0,2 | 1 | 19,45% | 0,00% |
| 0,2–0,3 | 168 | 27,27% | 28,57% |
| 0,3–0,4 | 1215 | 36,03% | 36,13% |
| 0,4–0,5 | 3038 | 45,42% | 44,34% |
| 0,5–0,6 | 3442 | 54,78% | 52,96% |
| 0,6–0,7 | 1838 | 64,05% | 62,24% |
| 0,7–0,8 | 290 | 72,82% | 73,45% |
| 0,8–0,9 | 8 | 81,58% | 87,50% |

Error absoluto de calibración agregado en estos bins: **0,0134**. Es descriptivo de este test, depende del binning y no garantiza calibración futura.

## F. Cambio de período y héroes nuevos: 19/10

La prueba usa los cinco cortes de agosto/septiembre como pasado (o solo el 20/09 en último corte) y partidas del 19/10 como evaluación. Las columnas se reconstruyen con estadísticas previas a octubre/19. Se repite **la matriz completa de todas las alternativas y combinaciones**, con un único salto temporal: no se calcula desvío entre pasos de F. Esta prueba va después de elegir y evaluar en septiembre; sus resultados son diagnóstico externo y no reabren la búsqueda.

| Modelo | A1 creciente | A2 creciente | Composición creciente | B2 último corte |
| --- | --- | --- | --- | --- |
| Dummy | 0,5000 | 0,5000 | 0,5000 | 0,5000 |
| Logística | 0,6233 | 0,6266 | 0,5912 | 0,6231 |
| Ridge | 0,6233 | 0,6280 | 0,5911 | 0,6171 |
| SGD logístico | 0,6233 | 0,6251 | 0,5916 | 0,6231 |
| SVM lineal | 0,6233 | 0,6285 | 0,5911 | 0,6104 |
| Naive Bayes gaussiano | 0,6233 | 0,6161 | 0,5903 | 0,6201 |
| Árbol | 0,6209 | 0,6189 | 0,5283 | 0,6193 |
| Random forest | 0,6209 | 0,6223 | 0,5759 | 0,6179 |
| Extra Trees | 0,6235 | 0,6211 | 0,5760 | 0,6210 |
| Gradient boosting | 0,6230 | 0,6239 | 0,5648 | 0,6202 |
| Hist. gradient boosting | 0,6236 | 0,6217 | 0,5838 | 0,6175 |
| AdaBoost | 0,6212 | 0,6233 | 0,5407 | 0,6222 |
| Red MLP | 0,6233 | 0,6205 | 0,5882 | 0,6041 |
| RBF aproximado | 0,6221 | 0,6144 | 0,5912 | 0,6166 |
| KNN | 0,5708 | 0,6112 | 0,5787 | 0,6134 |
| LDA | 0,6233 | 0,6286 | 0,5912 | 0,6144 |
| QDA | 0,6233 | 0,5949 | 0,5913 | 0,5765 |

Estos son los mayores AUC de las combinaciones base en octubre, **solo como resumen exploratorio**. Mostrar máximos entre muchos candidatos puede ser optimista; esta tabla no cambia la selección ni convierte octubre en validación de una nueva búsqueda.

| Modelo | Bloque | Días | Entrenamiento | AUC octubre | Accuracy |
| --- | --- | --- | --- | --- | --- |
| SVM lineal | A2 | 20 | Vida media 1 | 0,6309 | 59,64% |
| SGD logístico | all | 7 | Creciente | 0,6309 | 59,25% |
| SVM lineal | A2 | 20 | Vida media 2 | 0,6307 | 59,52% |
| SGD logístico | all_trend | 7 | Creciente | 0,6306 | 59,33% |
| SGD logístico | all | 7 | Vida media 2 | 0,6305 | 59,45% |
| SGD logístico | all | 20 | Creciente | 0,6304 | 59,34% |
| Ridge | A2 | 20 | Vida media 1 | 0,6303 | 59,47% |
| SGD logístico | all | 20 | Vida media 2 | 0,6302 | 59,69% |
| LDA | A2 | 20 | Creciente | 0,6302 | 59,55% |
| Ridge | A2 | 20 | Vida media 2 | 0,6301 | 59,58% |

El modelo elegido en validación, reentrenado con el pasado disponible, obtiene en octubre **AUC 0,6091**, accuracy **57,91%** y Brier **—**. Los coeficientes de composición de los héroes nunca vistos no pudieron aprenderse antes de octubre. A1 recibe información de esos héroes a través de estadísticas recientes sin ampliar la dimensión de entrada del modelo.

En la extensión de calibración posterior, Platt se ajusta con 40000 scores temporales que terminan en septiembre y da en octubre Brier **0,2438** y log-loss **0,6818**. No usa etiquetas de octubre para ajustar la sigmoide.

La comparación logística central de F da A1=0,6233 frente a B2=0,6231; Δ A1−B2=0,0002. A1 tiene un AUC numéricamente mayor. El signo por sí solo no establece superioridad estadística.  Esto describe una comparación observada; no identifica causalmente el efecto de un parche. Entre septiembre y octubre cambian período, disponibilidad de héroes y población de partidas; hay un solo corte posterior.

Frente al elegido, A1 logística con la misma ventana de 20 días obtiene AUC **0,6241**, contra **0,6091** del elegido. Es un contraste de diseños completos, no una ablación de una sola columna. La ventaja de validación del modelo complejo se revierte en octubre: esto muestra el límite de seleccionarlo con un mes estable. Se informa esa reversión sin reemplazar retrospectivamente al ganador ni usar octubre como una nueva grilla de selección.

## Qué se puede concluir y qué queda limitado

Las alternativas aisladas permiten atribuir diferencias a un cambio concreto de columnas, historial o ventana; las combinaciones permiten comprobar interacciones. El resultado debe defenderse con los tres valores por paso del CSV, además de su promedio. El criterio de simplicidad exige comparar la mejora con la variación entre cortes, además de mirar el máximo.

**Convergencia:** 232 ajustes registraron advertencias. En MLP se conserva el presupuesto de 120 iteraciones y se informa si alcanzó el límite; no se afirma optimalidad de esa familia. Las advertencias reales se conservan completas por configuración y paso en `results.jsonl`.

Las principales limitaciones son tres pasos de validación, un mes de desarrollo con poco cambio de meta, muestreo de las primeras 10.000 partidas por ID y un solo corte de octubre. El muestreo cubre parte del primer día, no tres días completos. No se evalúa explícitamente un horizonte de dos/tres semanas con múltiples repeticiones, ni se demuestra rendimiento garantizado para noviembre. Las estadísticas de cada corte son las que hoy devuelve la API para esa ventana; se conservan sus hashes, pero no se puede reconstruir la versión que la API servía en aquel día.

El notebook preliminar se conserva señalado como histórico; su lector usa los archivos y el catálogo presentes en sus datos anteriores, sin mezclar las nuevas particiones ni romper por las carpetas vacías. No se reemplazan sus salidas por las del estudio actual.

**Verificación de código:** siete controles automatizados cubren límites temporales, equivalencia de features, catálogo, pesos, separación de cortes, cobertura de la matriz e integridad del caché. Los archivos Python se compilaron. No se ejecutó la suite de integridad de Airflow: Airflow y Docker no están disponibles en este entorno Windows. El chequeo del caché del DAG se hace sobre su función pura y no sustituye una prueba del scheduler.

La comprobación integral también verifica los 18 datasets y sus hashes, 1.224 casos de desarrollo más 1.224 de F, ausencia de futuro en cada paso, invariancia de composición al cambiar solo la ventana estadística, coincidencia entre artefacto y predicciones guardadas, y preservación del primer test durante la calibración posterior. El resultado está en [verification.json](../experiments/training_strategy/verification.json).

## Archivos y reproducción

- [Todas las configuraciones, con interpretación por fila](resultados-experimentos-entrenamiento-completos.md).
- [CSV resumen](../experiments/training_strategy/summary.csv) y [métricas por paso](../experiments/training_strategy/folds.csv).
- [Resultados crudos](../experiments/training_strategy/results.jsonl), [selección](../experiments/training_strategy/selection.json) y [test final](../experiments/training_strategy/final_test.json).
- [Calibración posterior identificada por separado](../experiments/training_strategy/calibration_posthoc.json).
- [Auditoría de datos](../experiments/training_strategy/data_audit.json) y [cobertura de estadísticas](../experiments/training_strategy/stats_coverage.json).
- [Protocolo y comandos](../experiments/training_strategy/README.md), [entorno/configuraciones](../experiments/training_strategy/environment.json), [preparación](../scripts/prepare_training_experiments.py), [ejecución](../scripts/run_training_experiments.py) y [generación del informe](../scripts/report_training_experiments.py).
- [Documento original](entrega3-estrategia-entrenamiento-original.md).

Los contratos de datos se verificaron con [Deadlock API OpenAPI](https://api.deadlock-api.com/openapi.json). El esquema temporal sigue el principio de entrenamiento con pasado de [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), con cortes enteros como unidades. Las métricas del informe provienen de las corridas locales, no de esas referencias.
