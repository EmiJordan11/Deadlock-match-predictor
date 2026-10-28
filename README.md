# airflow-deadlock

## Experimentos de la Entrega 3

La comparación temporal completa se encuentra en
[resultados de entrenamiento](docs/resultados-experimentos-entrenamiento.md), con
[todas las configuraciones](docs/resultados-experimentos-entrenamiento-completos.md)
y [protocolo/comandos de reproducción](experiments/training_strategy/README.md).
Las alternativas se ejecutan primero por separado y luego combinadas, sobre
17 clasificadores. La validación usa pasado → futuro y reserva el 20/09 para
el elegido. Octubre se evalúa como estrés posterior.

El esquema usa el catálogo de 38 héroes: 73 columnas totales y 66
predictoras. Los cortes de la Entrega 3 se eligieron en fechas en las que
los héroes posteriores todavía no existían. Los datasets nuevos están
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
