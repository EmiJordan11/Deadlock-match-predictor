# Estrategia de entrenamiento y validación frente al cambio de meta

**Proyecto Integrador de Ciencia de Datos · UTN FRM · 2026 · Entrega 3 (Modelado)**
**Pregunta del proyecto:** ¿Podré saber quién ganará la partida sabiendo la composición de héroes de los equipos?

Este documento explica, con el detalle necesario para defenderlo en la Entrega 3:

1. qué significa que "el meta cambia" y por qué eso afecta a cómo se entrena el modelo;
2. qué formas hay de validar un modelo cuyos datos están en el tiempo, y en qué se diferencia el **walk-forward** de las demás;
3. las alternativas de entrenamiento (A a F), cada una con su mecanismo, sus ventajas, sus desventajas y lo que esperamos que dé con nuestros datos;
4. cuáles vamos a probar, en qué orden y con qué regla vamos a elegir.

Todos los números citados se midieron el 23/10/2026 contra la Deadlock API o sobre el bronce del DAG. Cuando un número viene de un script de exploración y no del pipeline, se aclara.

---

## 0. Correcciones respecto de lo que se discutió antes

Al armar este documento se revisaron los datos con más cuidado, y varias afirmaciones previas no se sostienen tal como estaban dichas:

| Lo que se había dicho | Lo que se comprobó |
|---|---|
| "El meta cambió mucho de julio a septiembre: correlación 0,69, Vyper +6,2, Mirage −4,8" | Esa comparación mezclaba **estadísticas unranked** (cortes 01/07 y 01/08, bajados con el filtro viejo) con **ranked** (corte 20/09). Comparando solo ranked, con la misma ventana de 14 días, entre el 20/08 y el 20/09 la correlación del winrate de los héroes entre cortes es **0,97 a 1,00**. En ese mes el meta casi no se movió (sección 2.2). |
| "El peso de la ventaja se mantuvo estable de julio a septiembre" | De los cuatro cortes de esa tabla, dos eran unranked. Con ranked solamente hay dos puntos: 11,1 (20/08) y 12,2 (05/09). Siguen siendo consistentes, pero la evidencia es más chica. |
| "Las probabilidades salen bien calibradas" | Se midió en el notebook preliminar, cuyo test (20/09) tenía la plata construida con `k = 10.000` y el entrenamiento con `k = 100`. Esa medición no vale: hay que repetirla. |
| "Con los cortes semanales más los de julio y agosto quedan unos 5 pasos de walk-forward" | Los cortes de julio y agosto son unranked y quedan afuera. Con 20/08, 27/08, 03/09 y 10/09 quedan **3 pasos**, más el test del 20/09. |
| "Las fechas de parche hay que sacarlas de otra fuente" | La API las tiene: `/v1/patches` (notas de parche) y `/v1/patches/big-days`. |
| "Validar dentro de un corte infla a la logística con 66 columnas: 0,624 contra 0,570" | Esos números salieron de los cortes **unranked**. Con cortes ranked, el modelo con un peso por héroe rinde igual dentro del corte que en otro corte (sección 2.4). Lo único que se cae es la logística de 66 columnas **sin regularizar**, y eso es sobreajuste por cantidad de columnas, no cambio de meta. |

La consecuencia de fondo, que atraviesa todo el documento: **en el único período que podemos usar (20/08 a 20/09, ranked, con los 38 héroes del catálogo), el meta estuvo casi quieto.** Es muy probable que las alternativas A, B y C empaten en ese período. La única forma de ponerlas a prueba frente a un cambio de meta real es la alternativa F (sección 5.6).

---

## 1. Conceptos base

### 1.1 Qué es un corte

Un corte es una corrida del DAG con una fecha `T`. Cada corte tiene **dos ventanas de tiempo que no se pisan**:

```
            ventana de estadísticas (N días)                 ventana de partidas (3 días)
  ─────────[=========================================] T [=====================]────────►
            de acá salen las ventajas                      de acá salen las FILAS
            (winrate de cada héroe, de cada                (cada partida, con su
             cruce y de cada pareja)                        composición y su winner)
```

- **Las filas** (lo que el modelo tiene que predecir) son partidas jugadas **después** de `T`.
- **Las columnas de ventaja** de cada fila se calculan con estadísticas de partidas jugadas **antes** de `T`. Por ejemplo, `hero_winrate_adv_mean` es el winrate promedio de los 6 héroes de A en esos N días, menos el de los 6 de B (centrado en 0 y con shrinkage).

Que las dos ventanas no se pisen es lo que evita la fuga: ninguna fila lleva adentro su propio resultado.

Los parámetros del DAG que definen un corte son `fecha_corte` (T), `dias_ventana_stats` (N), `dias_ventana_matches`, `objetivo_filas` y `k_shrinkage`. **Para que dos cortes sean comparables tienen que compartir todos los parámetros salvo la fecha.** Por eso la plata del 20/09, construida con `k = 10.000` y unos 20 días, hay que rehacerla con los mismos parámetros que los cortes nuevos.

### 1.2 Qué es "el meta"

Es la jerga de los juegos competitivos para *el estado actual del juego*: qué héroes son fuertes, cuáles se eligen y qué combinaciones funcionan. Cambia por dos motivos:

- **Parches de balance.** Valve sube o baja números de héroes e ítems. En nuestro período hubo parches el **12/08** (cambios a Apollo), el **22/08** (Celeste, ítems de curación), el **16/09** (menor) y el **29/09** ("City Never Sleeps", una actualización grande). Los títulos y fechas salen de `/v1/patches`.
- **Los jugadores.** Descubren una estrategia, la copian, y otra pasa de moda, aunque ningún número del juego haya cambiado.

### 1.3 Qué aprende el modelo, y por qué importa para el meta

Hay dos lugares donde el modelo puede "guardar" quién es fuerte:

- **En sus pesos.** Una logística con las 38 columnas de héroes aprende un número por héroe: "McGinnis suma tanto". Ese número queda congelado en el momento del entrenamiento.
- **En las columnas.** La regla de una columna recibe `hero_winrate_adv_mean` ya calculada con las estadísticas del momento. Lo único que aprende es **cuánto pesa estar mejor parado**: un único número β en `P(gana A) = σ(β · ventaja + b)`.

Si el meta cambia, lo primero queda viejo y lo segundo se actualiza solo. Esa es la diferencia que separa a la alternativa A de la B.

### 1.4 Qué es validar, y qué no puede prometer

Validar es **estimar cómo va a andar el modelo en el uso**, con datos que el modelo no vio. Ninguna validación garantiza el futuro: todas suponen que el futuro se parece en algo a los datos con que se validó. Si mañana sale un parche que cambia todo, ningún número de hoy lo anticipa.

Lo que sí se puede hacer, y es lo que propone este documento:

1. **Que la validación imite el uso lo más posible**: entrenar con el pasado y medir en lo que viene después (sección 3).
2. **Medir sobre la mayor cantidad posible de "pasos" hacia adelante**, para ver cuánto varía el resultado de un período al siguiente, y no quedarse con un número solo.
3. **Preferir diseños robustos por construcción**, que no dependan de que el futuro se parezca al pasado en el detalle, como la alternativa A.
4. **Decir explícitamente qué no se pudo medir**: en nuestro caso, un cambio de meta grande.

---

## 2. Lo que sabemos hoy, medido

### 2.1 Qué cortes sirven

| Corte | Población | Estado | ¿Entra al estudio? |
|---|---|---|---|
| 01/07 | Unranked | Bronce y plata de septiembre | **No**: otra población, y la API ya no tiene esos datos para rehacerlo |
| 01/08 | Unranked | Bronce y plata de septiembre | **No**: ídem |
| 20/08, 27/08, 03/09, 10/09 | Ranked | A generar con el DAG | **Sí**: entrenamiento y validación |
| 20/09 | Ranked | Bronce bien; plata con `k = 10.000` | **Sí**, como test, después de rehacer la plata con los parámetros comunes |
| 19/10 | Ranked | Solo bronce: 99,4 % de partidas descartadas por héroes fuera del catálogo | Solo con la alternativa F |

Hay dos límites duros:
- **Retención de la API:** guarda unos 3 meses. Una ventana de un día que termina el 20/07 ya viene vacía; el 31/07 todavía tiene datos (364.631 partidas-héroe). Cada día se pierde un día más. Un corte del 20/08 con 20 días de estadísticas empieza el 31/07: **está justo en el borde y hay que correrlo primero.**
- **Héroes nuevos:** después del 21/09 aparecieron 5 héroes (ids 78, 84, 85, 86 y 88). Con el catálogo de 38, los cortes posteriores no sirven.

### 2.2 Cuánto se movió el meta en ranked

Winrate de cada héroe en la ventana de 14 días previa a cada fecha, con datos ranked (consulta directa a `/v1/analytics/hero-stats`, fuera del pipeline):

| Correlación entre cortes | 20/08 | 27/08 | 03/09 | 10/09 | 20/09 |
|---|---|---|---|---|---|
| **20/08** | 1 | 0,98 | 0,98 | 0,98 | 0,97 |
| **27/08** | | 1 | 1,00 | 0,99 | 0,98 |
| **03/09** | | | 1 | 1,00 | 0,98 |
| **10/09** | | | | 1 | 0,99 |

Los cambios más grandes entre el 20/08 y el 20/09: McGinnis 52,8 % → 49,8 % (−2,9 puntos, casi todo entre el 20/08 y el 27/08, alrededor del parche del 22/08), Warden +1,8 y Haze +1,4. El resto se movió menos de un punto.

**Lectura:** en este mes el orden de fuerza de los héroes es prácticamente el mismo. Hubo parches, pero chicos.

### 2.3 Cuánto pesa la ventaja

Logística de una columna (`hero_winrate_adv_mean`), ajustada por separado en cada corte ranked de exploración (14 días, `k = 100`, 10.000 partidas cada uno):

| Corte | Peso β | Probabilidad de que gane A con ventaja +0,05 |
|---|---|---|
| 20/08 | 11,1 | 63,3 % |
| 05/09 | 12,2 | 65,6 % |

Con 10.000 partidas, el error estándar de β ronda ±0,6, así que la diferencia está dentro de lo que da el azar. Hay que confirmarlo con los cortes oficiales.

### 2.4 ¿Un peso por héroe envejece entre cortes ranked?

ROC AUC entrenando en un corte ranked de exploración. "Dentro" es validación cruzada al azar dentro del mismo corte; "en el otro" es medido en el otro corte, dos semanas antes o después:

| Modelo | 20/08: dentro / en 05/09 | 05/09: dentro / en 20/08 |
|---|---|---|
| Regla (1 columna) | 0,605 / 0,618 | 0,618 / 0,605 |
| Composición (38 héroes) | 0,605 / 0,619 | 0,623 / 0,605 |
| Todas (66), C = 1 | 0,602 / 0,616 | 0,622 / **0,592** |
| Todas (66), C = 0,001 | 0,605 / 0,618 | 0,621 / 0,605 |

- Con el meta quieto, **los pesos por héroe no envejecen**: rinden lo mismo que la regla en el otro corte.
- La caída de la logística de 66 columnas sin regularizar (0,622 → 0,592) se arregla con regularización: es sobreajuste por tener muchas columnas, no cambio de meta.
- Cada corte "acierta" distinto: el 05/09 es más fácil de predecir que el 20/08 con cualquier modelo. Esa variación entre cortes es justamente lo que el walk-forward pone a la vista.

---

## 3. Formas de validar, y por qué walk-forward

Para todos los esquemas se usa la misma notación: los cortes ordenados en el tiempo son 20/08, 27/08, 03/09 y 10/09 (entrenamiento) y 20/09 (test). `E` = entrena, `V` = valida, `·` = no se usa.

### 3.1 Partición al azar (holdout o KFold estratificado sobre filas)

Se mezclan todas las partidas de todos los cortes y se reparten al azar.

```
filas mezcladas:   E V E E V E E E V E E V E ...   (partidas de todos los cortes, de los dos lados)
```

- **Qué mide:** cómo anda el modelo con partidas "parecidas" a las de entrenamiento, del mismo momento.
- **Problema:** cada partida de validación tiene hermanas del mismo corte en entrenamiento: mismo meta, mismas estadísticas. Si el meta cambia, el número sale más alto que en el uso. El instructivo lo llama "el error más caro de esta entrega": no da ningún aviso, el número sale lindo.
- **Cuándo sirve:** cuando las filas son independientes y no hay tiempo de por medio. No es nuestro caso.

### 3.2 GroupKFold por corte (lo que tiene hoy el notebook preliminar)

Se deja afuera un corte entero por vez y se entrena con todos los demás, **sean anteriores o posteriores**.

```
              20/08  27/08  03/09  10/09
turno 1:        V      E      E      E
turno 2:        E      V      E      E     ← entrena con 03/09 y 10/09 para "predecir" 27/08
turno 3:        E      E      V      E
turno 4:        E      E      E      V
```

- **Qué mide:** cómo anda el modelo en un corte que no vio, *en cualquier dirección del tiempo*.
- **Problema 1, conceptual:** usa el futuro para predecir el pasado. El uso real nunca es así.
- **Problema 2, una fuga chica:** la ventana de estadísticas de un corte de entrenamiento *posterior* contiene las partidas del corte que se está validando. Por ejemplo, la ventana de 14 días del 03/09 va del 20/08 al 03/09 e incluye las partidas de validación del 27/08 al 30/08. Las columnas de entrenamiento quedan calculadas, en una parte mínima (unas 10.000 de ~240.000 partidas), con resultados de validación. El efecto numérico es despreciable, pero conceptualmente es una fuga, y es fácil de evitar.

### 3.3 Walk-forward con ventana creciente (expanding window)

Se entrena con **todo lo anterior** al corte k y se valida en el corte k. Se repite avanzando un corte por vez.

```
              20/08  27/08  03/09  10/09  │ 20/09
paso 1:         E      V      ·      ·    │
paso 2:         E      E      V      ·    │
paso 3:         E      E      E      V    │
final:          E      E      E      E    │  TEST (una sola vez)
```

- **Qué mide:** cómo anda el modelo **en el período siguiente**, habiendo visto solo el pasado. Es exactamente el uso.
- **Cumple el instructivo:** "si tus filas están en el tiempo y vas a predecir lo que viene: por fecha, se entrena con lo anterior y se valida con lo posterior (`TimeSeriesSplit`)".
- **No tiene la fuga de 3.2:** las columnas del corte validado se calculan con estadísticas que incluyen las partidas de entrenamiento, pero eso es información del pasado, disponible al momento de predecir. Ningún resultado de validación entra al entrenamiento.
- **Cada paso es un "salto hacia adelante" observado.** La media de los pasos estima el rendimiento esperado; el desvío dice cuánto varía de un período al siguiente.
- **Desventaja:** los primeros pasos entrenan con pocos datos (el paso 1, con un solo corte). Y con 4 cortes hay 3 pasos: el desvío se calcula sobre 3 números.

### 3.4 Walk-forward con ventana deslizante (sliding window)

Igual que 3.3, pero se entrena solo con **los últimos m cortes**, no con todo lo anterior. Con m = 1:

```
              20/08  27/08  03/09  10/09  │ 20/09
paso 1:         E      V      ·      ·    │
paso 2:         ·      E      V      ·    │
paso 3:         ·      ·      E      V    │
final:          ·      ·      ·      E    │  TEST
```

Es la forma de evaluar la alternativa B, "entrenar solo con lo más reciente". Comparar 3.3 contra 3.4 con el mismo modelo responde a la pregunta "¿conviene más dato o dato más fresco?".

### 3.5 Holdout temporal final

En todos los casos, el corte 20/09 queda afuera de todo y se mira **una vez**, al final, con el modelo ya elegido. La validación de 3.3 o 3.4 se usa para elegir; el test se usa para informar.

### 3.6 Tabla comparativa

| Esquema | ¿Usa el futuro para predecir el pasado? | ¿El mismo meta queda de los dos lados? | Cuántos números da | Qué responde |
|---|---|---|---|---|
| Al azar sobre filas | sí | **sí** | 5 (KFold) | "¿anda con partidas como estas?" |
| GroupKFold por corte | **sí** | no | uno por corte | "¿anda en un corte que no vio?" |
| Walk-forward creciente | no | no | uno por corte menos 1 | "¿anda en el próximo período, entrenado con todo el pasado?" |
| Walk-forward deslizante | no | no | uno por corte menos 1 | "¿anda en el próximo período, entrenado solo con lo reciente?" |
| Holdout temporal (test) | no | no | 1 | el número que se informa |

### 3.7 `TimeSeriesSplit` de scikit-learn y nuestra versión por corte

`TimeSeriesSplit` toma las filas **ordenadas** y las corta en bloques consecutivos del mismo tamaño: entrena con los bloques anteriores y valida con el siguiente. Tiene tres parámetros que importan:

- `max_train_size`: si se fija, la ventana es deslizante (3.4); si no, creciente (3.3).
- `gap`: filas que se saltean entre entrenamiento y validación, para dejar un margen.
- `test_size`: tamaño de cada bloque de validación.

Nuestros bloques naturales no son "n filas" sino **cortes**, de tamaños distintos. Por eso conviene generar los pasos a mano y pasárselos a scikit-learn, que acepta cualquier lista de pares (índices de entrenamiento, índices de validación) como `cv`:

```python
def walk_forward(corte_de_cada_fila, ventana=None):
    """Un paso por corte: entrena con los anteriores (todos, o los últimos `ventana`) y valida en ese."""
    cortes = sorted(corte_de_cada_fila.unique())
    for i in range(1, len(cortes)):
        anteriores = cortes[:i] if ventana is None else cortes[max(0, i - ventana):i]
        entrena = np.flatnonzero(corte_de_cada_fila.isin(anteriores))
        valida = np.flatnonzero(corte_de_cada_fila.eq(cortes[i]))
        yield entrena, valida

cv_creciente = list(walk_forward(g_train))              # alternativas A y C
cv_deslizante = list(walk_forward(g_train, ventana=1))  # alternativa B
cross_validate(pipeline, X_train, y_train, cv=cv_creciente, scoring={"auc": "roc_auc", "acc": "accuracy"})
```

El `gap` no hace falta: las partidas de un corte terminan a los 3 días de su `T`, y el corte siguiente empieza una semana después.

### 3.8 El horizonte

En el walk-forward con cortes semanales, cada paso mide "una semana hacia adelante". Se puede medir también "dos semanas hacia adelante" (entrenar hasta k − 2 y validar en k) para ver si el rendimiento se degrada con la distancia. Es la pregunta práctica para la aplicación: el modelo que se entrega el 28/10 se va a usar en la exposición del 18/11.

- Con la alternativa A, el horizonte afecta solo a β, porque las columnas se recalculan al predecir.
- Con la alternativa B, afecta a todos los pesos por héroe.

---

## 4. Qué se compara en cada paso

En cada paso del walk-forward se entrena el pipeline y se mide **ROC AUC** (métrica principal) y **accuracy** (al lado) sobre el corte de validación. Se informan la media y el desvío de los pasos.

Además del piso (`DummyClassifier`) y de las alternativas de las secciones 5.1 a 5.6, el instructivo pide comparar **al menos tres familias de modelos con la misma validación**: logística, árbol de decisión, random forest y gradient boosting, sobre las mismas columnas. Esa comparación se hace con el walk-forward creciente.

---

## 5. Las alternativas en detalle

### 5.1 Alternativa A — Que el meta viva en las columnas, no en el modelo

**Idea.** La fuerza de cada héroe no se aprende de las filas de entrenamiento: se **recalcula en cada momento** con las estadísticas de los N días previos, y llega al modelo ya calculada como ventaja de A sobre B. El modelo solo aprende la relación entre esa ventaja y la probabilidad de ganar.

**Cómo funciona, con números.** Con β ≈ 12 (sección 2.3):

| Ventaja de A (`hero_winrate_adv_mean`) | Lo que significa | Probabilidad de que gane A |
|---|---|---|
| 0 | composiciones parejas | 50 % |
| +0,02 | los héroes de A ganan, en promedio, 1 punto más que los de B | ≈ 56 % |
| +0,05 | 2,5 puntos más | ≈ 65 % |
| −0,05 | 2,5 puntos menos | ≈ 35 % |

(La ventaja está centrada: `2 · winrate − 1`. Una diferencia de 0,05 equivale a 2,5 puntos de winrate.)

Cuando un parche vuelve fuerte a un héroe, sus estadísticas suben en los días siguientes, la ventaja de los equipos que lo eligen sube, y el modelo le da más probabilidad **sin reentrenar**.

**Variantes que entran en A:**
- **A1, la regla de una columna** (`hero_winrate_adv_mean`).
- **A2, las 28 columnas de ventaja**: suman la ventaja en cruces, parejas, kills, souls, mínimo, máximo y desvío. Todas se recalculan por corte, así que siguen siendo "meta en las columnas". Hasta ahora no le ganaron a A1 por más que el desvío.

**Ventajas:**
- **Robusta por diseño:** no hay ningún peso atado a un héroe ni a un momento.
- **Pocos parámetros:** uno en A1. No sobreajusta y se explica en una oración.
- **Sirve para héroes nuevos:** si se actualiza el catálogo y la API trae sus estadísticas, el héroe nuevo entra al promedio sin tocar el modelo.
- **Simple para la aplicación:** al predecir, se piden las estadísticas de los últimos N días (una llamada a `hero-stats` para A1) y se aplica el modelo.

**Desventajas:**
- **Llega tarde a los cambios.** Ejemplo concreto: el parche del **16/09** cae 4 días antes del corte de test (**20/09**). Con 14 días de estadísticas, 10 de los 14 son de antes del parche; con 20 días, 16 de 20. Las ventajas del test describen, en su mayoría, el juego anterior al parche.
- **Solo sabe lo que dicen las estadísticas.** Si una combinación funciona por algo que el winrate promedio no captura, A no lo ve.
- **Pendiente:** la calibración (si "60 %" significa 60 de cada 100) hay que volver a medirla con datos consistentes.

**Qué esperamos:** AUC alrededor de 0,61 y accuracy alrededor de 58 %, estable entre pasos.

**¿Se prueba?** Sí. Es el diseño principal.

### 5.2 Alternativa B — Ventana móvil: entrenar solo con lo más reciente y reentrenar seguido

**Idea.** El modelo puede tener pesos por héroe (las 38 columnas de composición), pero se entrena **solo con el último corte** y se reentrena cada vez que hay uno nuevo. Así los pesos están siempre al día. Es la propuesta de "entrenar con un corte y testear con el siguiente", repetida en el tiempo.

**Cómo funciona.** Cada semana el DAG genera un corte nuevo, se reentrena con él y ese modelo se usa hasta el reentrenamiento siguiente. Se evalúa con el walk-forward deslizante (sección 3.4).

**Variantes:**
- **B1:** el pipeline de A1 entrenado con ventana deslizante. Aísla el efecto "dato fresco contra más dato" sin cambiar de modelo.
- **B2:** composición (38) más ventajas, con regularización, entrenado con ventana deslizante. Es la versión "pesos por héroe al día".

**Ventajas:**
- **Pesos al día:** lo que el modelo aprende es del meta más reciente.
- **Puede captar cosas que las estadísticas no captan**, porque aprende de los resultados directamente: por ejemplo, un héroe que gana más en partidas de cierto tipo.

**Desventajas:**
- **También llega tarde:** entre un reentrenamiento y el siguiente los pesos quedan fijos, y un parche a mitad de semana no se ve hasta el próximo corte.
- **Menos datos por entrenamiento:** 10.000 partidas para estimar 38 pesos por héroe o más. Con el meta quieto, eso es solo ruido sin beneficio.
- **Exige infraestructura en la aplicación:** reentrenar periódicamente, por ejemplo agendando el DAG con `@weekly`, que hoy no tiene schedule.
- **No sirve para héroes nuevos** hasta tener un corte con ellos.

**Qué esperamos con nuestros datos:** con el meta quieto (sección 2.4), B1 y B2 deberían empatar con A, o quedar un poco abajo por entrenar con menos datos.

**¿Se prueba?** Sí, las dos variantes. Es la comparación que responde "¿un corte o varios?" con un número, y no con un argumento.

### 5.3 Alternativa C — Todos los cortes, con más peso a los recientes

**Idea.** Punto medio entre A (todo el pasado pesa igual) y B (solo el último corte). Se entrena con todos los cortes anteriores, pero cada partida pesa según su antigüedad:

```
peso de una partida = 0,5 ^ (antigüedad en cortes / vida media)
```

Con vida media 1, el corte anterior pesa 1, el de hace dos cortes 0,5, el de hace tres 0,25. Con vida media 2: 1, 0,71, 0,5, y así.

**Cómo se implementa.** Los estimadores de scikit-learn aceptan `sample_weight` en `fit`. En un pipeline se pasa como `pipeline.fit(X, y, modelo__sample_weight=pesos)`. Como el peso depende de qué corte se está validando, conviene recorrer los pasos del walk-forward con un loop propio que calcule los pesos en cada paso.

**Ventajas:**
- Usa todos los datos, pero deja que lo reciente mande.
- Si el meta está quieto, se comporta como A; si se mueve, se acerca a B.

**Desventajas:**
- **La vida media es otro hiperparámetro**, y con 3 pasos de walk-forward es difícil elegirla con confianza: la diferencia entre vida media 1 y 2 va a ser menor que el desvío.
- Para A1 (un solo peso β, estable entre cortes) ponderar casi no cambia nada. Tiene sentido sobre todo para modelos con pesos por héroe (B2).

**¿Se prueba?** Sí, con dos valores (vida media 1 y 2 cortes) sobre el pipeline de B2. Es barato. Si empata, se informa como "probado y no anduvo".

### 5.4 Alternativa D — Qué tan fresca es la información del meta (`dias_ventana_stats` como hiperparámetro)

**Idea.** En A, toda la adaptación al meta pasa por **cuántos días de estadísticas** se usan para calcular las ventajas. Ese número es un hiperparámetro como cualquier otro, y se elige por walk-forward.

**El compromiso:**
- **Ventana corta (7 días):** sigue el meta rápido. Después del parche del 16/09, una ventana de 7 días al 20/09 tiene 4 días post-parche de 7, no 4 de 14 ni 4 de 20. Cuesta más ruido.
- **Ventana larga (20 o 30 días):** estadísticas más estables, pero describen un juego que puede ya no existir.

**¿Cuánto ruido agrega acortar?** El error estándar de un winrate es `√(p(1 − p) / n)`. Con los volúmenes medidos:

| Ventana | Partidas por héroe (aprox.) | Error del winrate de un héroe |
|---|---|---|
| 7 días | 20.000 a 75.000 | ±0,2 a 0,35 puntos |
| 14 días | 40.000 a 150.000 | ±0,13 a 0,25 puntos |
| 20 días | 60.000 a 220.000 | ±0,1 a 0,2 puntos |

Los héroes difieren entre sí en varios puntos de winrate (de ~45 % a ~53 %), así que incluso 7 días los ordena bien. El ruido de las parejas (counters y sinergias) es mayor, pero A1 no las usa.

**Cómo se implementa.** Para cada corte hay que tener las estadísticas de cada ancho de ventana: 3 pedidos a la API por corte y por ancho (o 1 para A1, que usa solo `hero-stats`). El DAG hoy guarda las estadísticas en `bronze/corte=T/stats/` **sin el ancho de ventana en la ruta**, así que dos corridas con distinto `dias_ventana_stats` se pisan, o la segunda reusa el caché de la primera. Hay dos formas:
1. **Cambiar el DAG** para que guarde en `stats/dias=N/` y construya la plata por cada N. Es lo más prolijo: todo sigue saliendo del pipeline.
2. Bajar las estadísticas extra desde el notebook. Es más rápido de escribir, pero queda una fuente de datos por fuera del pipeline.

Se recomienda la primera.

**Límite por retención:** una ventana de 30 días para el corte del 20/08 empezaría el 21/07, que la API ya no tiene. Para los cortes de agosto, el ancho máximo posible es de unos 20 días, y solo si se corren ya.

**Ventajas:**
- Ataca la debilidad principal de A (llegar tarde) sin agregar parámetros al modelo.
- Es barato en datos y en cómputo.
- La respuesta sirve directo para la aplicación: cuántos días de estadísticas pedir al predecir.

**Desventajas:**
- Requiere un cambio en el DAG.
- La retención limita los anchos posibles.
- Con el meta quieto, las diferencias entre anchos van a ser chicas. Igual es un resultado: "en un período sin parches grandes, el ancho de ventana no importa entre 7 y 20 días".

**¿Se prueba?** Sí, con 7, 14 y 20 días sobre A1. Es el ajuste que más sentido tiene en este problema.

### 5.5 Alternativa E — Columnas que avisen que el meta se está moviendo

**Idea.** Darle al modelo información sobre **qué tan viejas o qué tan inestables** son las estadísticas, para que aprenda a desconfiar de ellas cuando corresponde. Dos candidatas:

- **Tendencia:** para cada héroe, el winrate de los últimos 7 días menos el de los 7 anteriores, agregado como ventaja de A sobre B. Si un héroe viene subiendo, su winrate de 14 días lo subestima.
- **Días desde el último parche:** sale de `/v1/patches`. Si el parche es reciente, las estadísticas mezclan dos juegos distintos.

**Ventajas:**
- Es la respuesta directa a "llega tarde": no achica la ventana, le avisa al modelo.
- La tendencia sale casi gratis si se hace D, porque usa las mismas ventanas de 7 días.

**Desventajas:**
- **Pocos casos para aprender:** "días desde el parche" toma un valor por corte. Con 4 o 5 cortes y 2 parches en el período, el modelo no tiene de dónde aprender esa relación.
- La tendencia agrega columnas ruidosas en un período donde el meta casi no se movió.
- Más complejidad para explicar, por una ganancia probablemente nula.

**¿Se prueba?** Solo la tendencia, y solo si D muestra que el ancho de ventana importa. "Días desde el parche" queda afuera, con el motivo escrito.

### 5.6 Alternativa F — Actualizar el catálogo a 43 héroes y testear con un corte de octubre

Esta alternativa no estaba en la lista original. Aparece al ver que en agosto y septiembre el meta estuvo quieto (sección 2.2).

**Idea.** El único cambio de meta grande al alcance es la actualización del **29/09** ("City Never Sleeps"). Después de ella aparecen 5 héroes nuevos: probablemente llegaron con esa actualización, aunque el resumen de la API no alcanza para confirmarlo. Para usar ese período, el pipeline tiene que conocer a los 43 héroes. Con eso se podría:
- entrenar con los cortes de agosto y septiembre;
- testear (o agregar un paso de walk-forward) con un corte de octubre, por ejemplo el 19/10, que ya está en bronce.

Ese paso sí mide lo que nos importa: **cómo anda un modelo entrenado antes de un cambio grande, aplicado después.**

**Qué esperaríamos:**
- **A** debería aguantar: las ventajas de octubre se calculan con estadísticas de octubre, que incluyen a los héroes nuevos.
- **B2** tiene pesos por héroe aprendidos antes del parche, y **ningún peso** para los 5 héroes nuevos: es la prueba más clara de la diferencia entre las dos filosofías.

**Ventajas:**
- Es el único experimento que responde la objeción de fondo ("con estos cortes no sé si funciona en metas futuros") con un cambio de meta real.
- **La aplicación de la Entrega 4 lo necesita igual:** en noviembre el juego tiene 43 héroes, y un pipeline con 38 descarta casi todas las partidas.

**Desventajas:**
- **Cambio en el pipeline:** `schema.HEROES` pasa de 38 a 43; el CSV pasa de 73 a 78 columnas, y los cortes viejos tienen ceros en las 5 columnas nuevas. Hay que re-correr el DAG en todos los cortes para que el esquema sea uniforme.
- **Más trabajo antes del 28/10.**
- Los héroes nuevos tienen poca historia en la ventana de estadísticas de su primer corte. El shrinkage los acerca a la neutralidad, que es justamente lo que tiene que hacer.

**¿Se prueba?** Es la decisión que más cambia el alcance, y la tiene que tomar el grupo. La recomendación es hacerla si hay tiempo antes del 28/10, porque además adelanta trabajo de la Entrega 4. Si no hay tiempo, se presenta como **limitación explícita**: "en el período medido el meta estuvo estable; no pudimos medir el modelo frente a un cambio grande".

---

## 6. Qué vamos a probar, en qué orden y cómo se elige

### 6.1 La matriz de experimentos

Todos con walk-forward sobre los cortes 20/08, 27/08, 03/09 y 10/09 (3 pasos). Todas las corridas comparten `match_mode = ranked`, el mismo `k_shrinkage` (100) y el mismo `dias_ventana_stats`, salvo en D, donde varía.

| # | Qué | Modelo y columnas | Ventana de entrenamiento | Para qué |
|---|---|---|---|---|
| 0 | Piso | `DummyClassifier` | creciente | contra qué se compara todo |
| 1 | A1 | logística, 1 columna | creciente | el diseño principal |
| 2 | A2 | logística, 28 ventajas | creciente | ¿suman las otras ventajas? |
| 3 | Familias | logística, árbol, bosque y boosting sobre las 66 | creciente | la comparación que pide el instructivo |
| 4 | B1 | logística, 1 columna | deslizante (último corte) | ¿más dato o dato fresco?, sin cambiar de modelo |
| 5 | B2 | logística regularizada, 38 + 28 | deslizante | pesos por héroe al día |
| 6 | C | igual que B2 | creciente con pesos, vida media 1 y 2 | punto medio |
| 7 | D | A1 con 7, 14 y 20 días de estadísticas | creciente | cuántos días siguen mejor al meta |
| 8 | Ajuste | grilla del elegido (`C`, bloques de columnas) | creciente | el ajuste que pide el instructivo, con la tabla completa |
| 9 | E (condicional) | A1 + tendencia | creciente | solo si D muestra diferencias |
| 10 | F (si se decide) | A1 y B2 entrenados en ago–sep | test en octubre | cambio de meta real |

### 6.2 La regla de elección, escrita antes de ver los resultados

1. **Se elige por ROC AUC medio en el walk-forward**, con el desvío al lado.
2. **Un modelo más complejo solo se elige si le gana al más simple por más que el desvío del simple.** Esa es la regla del instructivo.
3. **Entre empatados, el más simple y el más robusto por diseño.** El orden de preferencia es A1, A2, B1, C, B2. Si todas las alternativas empatan, que es lo más probable con el meta quieto, gana A1, y el argumento es de diseño: no tiene pesos atados al meta, sirve para héroes nuevos y no requiere reentrenar.
4. **El test (20/09) se mira una vez**, con el modelo elegido, y se informa junto con el piso.

### 6.3 Qué se presenta si todo empata

Un empate también es un resultado, y hay que poder contarlo:

> "Comparamos entrenar con todos los cortes anteriores, solo con el último y con pesos por antigüedad, y la ventana de estadísticas de 7, 14 y 20 días. En el período agosto–septiembre todas dan lo mismo dentro del desvío, porque el meta casi no se movió (correlación del winrate de los héroes de 0,97 a 1,00 entre cortes). Elegimos la regla de una columna porque, a igual rendimiento, es la única que no depende de que el meta se quede quieto."

---

## 7. Cómo responder las preguntas del docente sobre la validación

**"¿Por qué los pedazos de la validación se arman así?"**
Porque el modelo se usa para predecir partidas futuras con estadísticas del pasado. Cada paso del walk-forward entrena con cortes anteriores y valida en el siguiente, que es exactamente esa situación. El test es el corte más reciente y se mira una sola vez.

**"¿Qué le pasaría al número si los armaran al azar?"**
Mezclaría partidas del mismo corte de los dos lados: el modelo vería el meta de validación durante el entrenamiento. En nuestros datos ranked de agosto y septiembre la diferencia es chica porque el meta estuvo estable. En los cortes unranked de julio y agosto, donde el meta sí cambió, la logística de 66 columnas pasaba de 0,570 a 0,624 de AUC.

**"¿Y por qué no dejar un corte afuera por vez (GroupKFold)?"**
Porque en algunos turnos entrena con el futuro para predecir el pasado, y porque las estadísticas de un corte posterior incluyen las partidas del corte que se valida: es una fuga chica, pero evitable.

**"¿Cómo saben que va a andar en metas futuros?"**
No lo sabemos, y lo decimos. Medimos un mes sin cambios grandes. Elegimos el diseño que no depende de que el meta se quede quieto, y (si se hace F) lo probamos contra la actualización del 29/09.

---

## 8. Próximos pasos

1. **Fijar los parámetros comunes del DAG:** `match_mode = ranked` (ya está en el código), `k_shrinkage = 100` y `dias_ventana_stats` (14 o 20; ver el punto 2).
2. **Correr primero el corte del 20/08**, porque está en el borde de la retención. Con 20 días, su ventana empieza el 31/07, que hoy todavía existe en la API.
3. **Correr el resto de los cortes:** 27/08, 03/09 y 10/09.
4. **Rehacer la plata del 20/09** con los parámetros comunes. Si el ancho elegido es 20, alcanza con re-disparar sin `force` (reusa el bronce). Si es 14, hay que bajar de nuevo las estadísticas.
5. **Cambiar el DAG para D:** estadísticas en `stats/dias=N/` y una plata por cada ancho. Además, un `parametros.json` en cada plata, para que el notebook verifique que todos los cortes sean comparables.
6. **Decidir F:** catálogo de 43 héroes y corte de octubre.
7. **Notebook:** reemplazar `GroupKFold` por el walk-forward de la sección 3.7, agregar la matriz de la sección 6.1 y recalcular dentro del notebook lo que hoy está en la tabla de exploración (carril, tramo, jugadores), o sacarlo.

---

## 9. Glosario

| Término | Significado en este proyecto |
|---|---|
| **Corte** | Una corrida del DAG con fecha `T`: estadísticas de antes de `T` y partidas de después de `T`. |
| **Meta** | El estado del juego en un momento: qué héroes son fuertes y cuáles se eligen. |
| **Parche** | Actualización de Valve que cambia números de héroes o ítems. Fechas en `/v1/patches`. |
| **Ventaja** | Diferencia A − B de una estadística previa, por ejemplo el winrate promedio de los héroes. Centrada en 0. |
| **Shrinkage (`k`)** | Ajuste que acerca a la neutralidad las tasas con pocas partidas. `k` son "partidas imaginarias de 50/50". |
| **Walk-forward** | Validación que entrena con el pasado y valida en el período siguiente, avanzando de a un período. |
| **Ventana creciente / deslizante** | En walk-forward: entrenar con todo el pasado, o solo con los últimos `m` períodos. |
| **Horizonte** | Distancia en el tiempo entre el fin del entrenamiento y lo que se predice. |
| **ROC AUC** | Probabilidad de que, tomando una partida que ganó A y una que perdió, el modelo le dé más probabilidad a la que ganó. 0,5 es azar. |
| **Calibración** | Que una probabilidad de 60 % signifique 60 victorias de cada 100. |
| **Retención** | Cuánto tiempo guarda la API sus datos. Medida: unos 3 meses. |
