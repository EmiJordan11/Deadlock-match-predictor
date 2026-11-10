# airflow-deadlock

*¿Se puede saber quién gana una partida de Deadlock sabiendo solo qué héroes
eligió cada equipo?* Proyecto Integrador · Ciencia de Datos · UTN FRM 2026.

## La aplicación (Entrega 4)

**URL pública:** <https://emijordan11.github.io/Deadlock-match-predictor/>

Una página que cuenta el proyecto en tres pasos (los datos, lo que
encontramos, el modelo) y después deja explorar los 38 héroes y armar un
draft: los equipos eligen por turnos y el modelo recomienda héroes y muestra
cómo se mueve la probabilidad de cada equipo pick a pick.

### La cadena completa

```
DAG deadlock_ingest      → include/output/bronze/ y silver/   (Entrega 1)
notebooks/entrega2_...   → hipótesis                          (Entrega 2)
notebooks/entrega3_modelado_v3.ipynb → modelo_entrega3_composicion.joblib  (Entrega 3)
app/build_app.py         → docs/index.html                    (Entrega 4)
```

### Regenerar la aplicación

```bash
pip install -r notebooks/requirements.txt
python app/build_app.py
```

`build_app.py` lee los cortes de `include/output/silver/` y el bronce de
partidas, entrena el modelo elegido en la Entrega 3 (logística sobre las 38
columnas de composición, sin escalar, l2, `C = 0,1`, entrenada con el último
corte de entrenamiento) con la semilla 42, vuelve a medir con el mismo
walk-forward la tabla de comparación que muestra la app, y escribe
`docs/index.html`. Al final imprime los números y verifica que las
predicciones coincidan con el `.joblib` que guardó el notebook. Si los datos
cambian, correrlo de nuevo actualiza la aplicación. Tarda unos cinco minutos.

| Archivo | Qué es |
|---|---|
| `app/build_app.py` | El script que genera la aplicación. |
| `app/plantilla.html` | El HTML, el CSS y el JavaScript de la página; los números entran por marcadores `{{...}}`. |
| `docs/index.html` | La aplicación generada: los gráficos (especificaciones de Altair) y los datos viajan adentro. |
| `docs/heroes/` | Los retratos de los 38 héroes, bajados de los assets de deadlock-api.com. |
| `docs/vendor/` | Vega, Vega-Lite y vega-embed, para que la página no dependa de un CDN. |

En el navegador no corre Python: el modelo es una logística, así que la
página calcula la probabilidad con los 38 pesos y el intercepto que escribe
`build_app.py`, y da lo mismo que el notebook.

### Publicarla

GitHub Pages sirve la carpeta `docs/` de la rama `main`: en el repositorio,
*Settings → Pages → Build and deployment → Deploy from a branch*, rama `main`,
carpeta `/docs`. Cada `git push` con un `docs/index.html` nuevo actualiza la URL.
Para verla localmente alcanza con abrir `docs/index.html` en el navegador.

## Héroes nuevos: el catálogo de cada corte

Valve agrega héroes cada pocas semanas. Con el catálogo fijo de 38, el filtro
de calidad descartaba toda partida con un héroe nuevo: el 19/10 quedaban 60
de 10.000. Ahora el DAG arma el catálogo de cada corte (`build_catalogo`,
módulo `include/deadlock/catalogo.py`) con tres fuentes:

1. `/v1/assets/heroes`: los héroes jugables. No se filtra por
   `in_development`: Violet y Baba tienen esa marca y se juegan en ranked.
2. Las estadísticas de la ventana: los héroes jugados antes del corte.
3. Las partidas del corte: un héroe recién lanzado puede aparecer acá antes
   que en las otras dos.

Cada héroe es una columna de composición (+1/−1/0); uno que no se jugó en un
corte tiene su columna en cero, así que juntar cortes con catálogos
distintos es completar con 0. El registro
(`include/output/silver/catalogo_dinamico/registro_heroes.json`) le fija a
cada héroe un nombre de columna la primera vez que aparece y no lo cambia
nunca. Si aparece un héroe que el registro no tenía, el log de
`build_catalogo` lo avisa con un `HEROES NUEVOS`. Un héroe que se juega sin
tener estadísticas todavía aporta ventaja 0 (neutra) a las 28 ventajas.

La salida va a `include/output/silver/catalogo_dinamico/corte=T/dias=N/k=K/`:
`features.parquet` y `dataset.csv`, `catalogo.json` (qué héroes tiene el
corte, cuántas partidas jugó cada uno, si tiene estadísticas) y
`filtro_calidad.json` (cuántas partidas se descartaron y por qué). Los cortes
de la Entrega 3 (`silver/corte=T/`, 38 héroes) no se tocan: el notebook de
modelado, la aplicación y los experimentos siguen dando los mismos números.
Correr el DAG sobre una fecha de la Entrega 3 da las mismas 73 columnas de
siempre más las de los héroes nuevos, todas en cero.

## Experimentos de la Entrega 3

La comparación temporal completa se encuentra en
[resultados de entrenamiento](docs/resultados-experimentos-entrenamiento.md), con
[todas las configuraciones](docs/resultados-experimentos-entrenamiento-completos.md)
y [protocolo/comandos de reproducción](experiments/training_strategy/README.md).
Las alternativas se ejecutan primero por separado y luego combinadas, sobre
17 clasificadores. La validación usa pasado → futuro y reserva el 20/09 para
el elegido. Octubre se evalúa como estrés posterior.

Los cortes de la Entrega 3 usan el catálogo de 38 héroes: 73 columnas
totales y 66 predictoras. Se eligieron en fechas en las que los héroes
posteriores todavía no existían. El DAG ya no depende de ese catálogo (ver
«Héroes nuevos: el catálogo de cada corte»). Los datasets nuevos están
en `include/output/silver/corte=T/dias=N/k=K/features.parquet`, junto a
`parametros.json` con configuración y hashes. El DAG exporta `dataset.csv`
en esa misma partición. Las estadísticas crudas se separan por `stats/dias=N/`
y solo se reutilizan si la petición y el hash coinciden.

## Pipeline

Pipeline de la Entrega 1 (Ingeniería de datos) del Proyecto Integrador -
Ciencia de Datos, UTN FRM 2026. Construye el dataset de partidas de
Deadlock que responde la pregunta del proyecto: *¿se puede saber quién gana
una partida sabiendo solo la composición de héroes de los dos equipos?*

El diseño de las 73 columnas, los 6 problemas detectados en el diseño
original y sus soluciones están documentados en
`../decisiones-diseno-dataset.md` y `../estructura-dataset-final.md`. Este
proyecto es la implementación de ese diseño, con la misma estructura que
`airflow-fifa/` (Astro CLI + Airflow 3, TaskFlow API) pero adaptada a una
fuente distinta: una API REST JSON estable, no un sitio para scrapear.

## Arrancar

```bash
astro dev start
```

UI en `localhost:8080`. Disparar `deadlock_ingest` manualmente (no tiene
schedule -- ver "Por qué no está agendado" más abajo).

## Los params de la corrida

| Param | Default | Qué controla |
|---|---|---|
| `fecha_corte` | `2026-08-20` | La fecha `T` que separa las dos ventanas (Problema 3). Fija a propósito: nunca calculada con `datetime.now()`. |
| `dias_ventana_stats` | 14 | Ancho de la ventana de estadísticas, hacia atrás desde T. |
| `dias_ventana_matches` | 3 | Ancho de la ventana de partidas, hacia adelante desde T. |
| `objetivo_filas` | 10000 | Partidas a juntar. |
| `min_matches_stats` | 1 | `min_matches` pedido a los 3 endpoints de analytics, antes del shrinkage. |
| `k_shrinkage` | 100 | Constante `k` del shrinkage bayesiano (Problema 4). |
| `force` | `false` | Ignora el bronce en cache para este `fecha_corte` y vuelve a pedirle todo a la API, aunque ya existan archivos. |

Toda corrida junta `objetivo_filas` partidas — no hay un modo "de prueba"
más chico. Para iterar rápido durante el desarrollo, bajá `objetivo_filas`
a mano en el trigger; para la corrida oficial, dejalo en 10000 (o el valor
que el grupo documente) junto con el `fecha_corte` elegido.

## El grafo

```
resolve_window
      │
      ├──► extract_hero_stats ─────┐
      ├──► extract_counter_stats ──┤
      ├──► extract_synergy_stats ──┼──► validate_bronze ──► build_stat_tables ──┐
      └──► extract_matches ────────┘                                            ├──► build_features ──► quality_gate ──► export_csv
                                    └────────────────────► build_matches_table ─┘
```

- **Bronce** (`extract_*`): las respuestas crudas de la API, particionadas
  por `corte=<T>` y estadísticas por `dias=N`. El caché se reutiliza si
  coinciden la petición y el contenido verificado; las distintas ventanas
  no comparten archivos.
- **Plata** (`build_stat_tables`, `build_matches_table`, `build_features`):
  las tablas de consulta con *shrinkage* ya aplicado, la base de partidas
  parseada, y finalmente las features del catálogo actual.
- **`quality_gate`**: los 6 chequeos medibles de la consigna. Si alguno
  falla, el DAG falla -- no se publica un CSV que no cumple.

## Por qué no está agendado

Con `fecha_corte` fijo por Param, agendar (`@daily`, etc.) no aporta nada:
"automatizado" acá significa que ningún dato se toca a mano entre la API y
el CSV final, no que el DAG corra solo todos los días. Se dispara
manualmente para cada corte que el grupo quiera producir.

## Por qué es más simple que `fifa_ingest.py`

`fifa_ingest.py` tiene sensor, branch, short-circuit, `.expand()` dinámico
y 3 niveles de degradación porque scrapea un sitio detrás de Cloudflare que
puede bloquear, cambiar de estructura o caerse. La Deadlock API es un REST
JSON estable, verificado en vivo antes de escribir este DAG: no requiere
API key y no bloquea. Su único modo de falla esperable es rate limiting
transitorio (429), que resuelven los `retries` de cada tarea de red -- no
hace falta una máquina de estados completa para eso. El razonamiento
completo está en el docstring de `dags/deadlock_ingest.py`.

## Verificar el dataset producido

```python
import pandas as pd
df = pd.read_parquet("include/output/silver/corte=2026-08-20/dias=14/k=100/features.parquet")

df["match_id"].is_unique                       # clave sin duplicados
len(df)                                         # volumen
df.shape                                        # ancho
df.dtypes.value_counts()                        # mezcla de tipos
df.isna().mean().sort_values(ascending=False)   # nulos conocidos (solo average_badge/average_badge_diff deberían tener)
df.columns[df.isna().all()]                     # sin columnas 100% nulas
```

## Testing

```bash
astro dev pytest
```
