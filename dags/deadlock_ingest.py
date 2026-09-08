"""
### Dataset canonico Deadlock -- Ciencia de Datos, UTN FRM 2026

Construye el dataset de partidas de Deadlock (deadlock-api.com) para
responder la pregunta del proyecto: ¿se puede saber quien gana una partida
sabiendo solo la composicion de heroes de los dos equipos?

El diseño completo de las 73 columnas, y por que cada una es como es, esta
documentado en docs/decisiones-diseno-dataset.md y
docs/estructura-dataset-final.md. Este DAG materializa esas decisiones.

**Dos ventanas de tiempo que no se solapan** (Problema 3 -- fuga de datos):
una fecha de corte `T` separa las estadisticas historicas de heroes (todo lo
anterior a T) de las partidas que van a ser filas del dataset (todo lo
posterior a T). Si se solaparan, cada fila contendria un fragmento del
resultado que se le pide predecir al modelo.

**Por que este DAG es mas simple que fifa_ingest.py**, que tiene sensor,
branch, short-circuit, `.expand()` dinamico y 3 niveles de degradacion:
todo eso existe en fifa_ingest porque sofifa.com esta detras de Cloudflare y
puede bloquear, cambiar de estructura o caerse. La Deadlock API es un REST
JSON estable, sin bloqueo, verificado en vivo antes de escribir este DAG.
Su unico modo de falla esperable es rate limiting transitorio (429), que
resuelven los `retries` de cada tarea de red -- no hace falta una maquina de
estados completa para eso.

**Bronce y plata**, igual que en fifa_ingest:

* **Bronce** (`extract_*`) -- las 3 respuestas de analytics y las paginas de
  `/v1/matches/metadata`, tal como las devolvio la API. Particionado por
  `corte=<T>`: correr el DAG dos veces con el mismo T no genera trafico
  extra, porque cada tarea de extraccion revisa primero si su archivo ya
  existe.
* **Plata** (`build_stat_tables`, `build_matches_table`, `build_features`) --
  las tablas de consulta con shrinkage ya aplicado, la base de partidas
  parseada, y finalmente las 66 columnas de features. Separar estos tres
  pasos permite recalcular las features (por ejemplo si cambia
  `k_shrinkage`) sin volver a pedirle nada a la API ni reparsear el bronce.

No hay capa de respaldo congelado: la Entrega 1 no se demuestra con internet
en vivo (se muestra una corrida ya hecha en la UI), asi que la resiliencia
que le importa a este DAG es la reproducibilidad del `T` fijo, no sobrevivir
un corte de red durante la clase.
"""
from __future__ import annotations

import gzip
import json
import logging
import pickle
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, task

from deadlock import client, schema
from deadlock.transform import build_match_row, parse_match
from deadlock.transform import build_stat_tables as _build_stat_tables

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/usr/local/airflow/include/output")
BRONCE_DIR = OUTPUT_DIR / "bronze"
PLATA_DIR = OUTPUT_DIR / "silver"


def _stats_bronze_path(corte: str, nombre: str) -> Path:
    return BRONCE_DIR / f"corte={corte}" / "stats" / f"{nombre}.json"


def _matches_bronze_dir(corte: str) -> Path:
    return BRONCE_DIR / f"corte={corte}" / "matches"


def _silver_dir(corte: str) -> Path:
    return PLATA_DIR / f"corte={corte}"


@dag(
    dag_id="deadlock_ingest",
    # Con T fijo por Param, agendar no aporta nada: "automatizado" acá
    # significa que ningun dato se toca a mano, no que corra solo. Se
    # dispara manualmente para cada corte que se quiera producir.
    schedule=None,
    start_date=pendulum.datetime(2026, 9, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    tags=["ciencia-de-datos", "unidad-1", "dataset-canonico", "deadlock"],
    doc_md=__doc__,
    params={
        "fecha_corte": Param(
            "2026-08-20", type="string", format="date",
            title="Fecha de corte (T)",
            description=("Separa la ventana de estadisticas (todo lo anterior) de la "
                         "ventana de partidas (todo lo posterior). Fija, nunca "
                         "'hoy': es la base de la reproducibilidad (Problema 3)."),
        ),
        "dias_ventana_stats": Param(
            14, type="integer", minimum=1,
            title="Ancho de la ventana de estadisticas (dias, hacia atras desde T)",
        ),
        "dias_ventana_matches": Param(
            3, type="integer", minimum=1,
            title="Ancho de la ventana de partidas (dias, hacia adelante desde T)",
        ),
        "objetivo_filas": Param(
            10000, type="integer", minimum=1,
            title="Partidas a juntar en modo full",
        ),
        "min_matches_stats": Param(
            1, type="integer", minimum=1,
            title="min_matches pedido a los endpoints de analytics",
            description="Bajo a proposito: no perder pares raros antes del shrinkage (Problema 4).",
        ),
        "k_shrinkage": Param(
            100, type="integer", minimum=1,
            title="Constante k del shrinkage bayesiano (Problema 4)",
            description="Ver docs/decisiones-diseno-dataset.md por el razonamiento estadistico.",
        ),
        "modo": Param(
            "subset", enum=["subset", "full"],
            title="Modo de corrida",
            description="subset: ~500 partidas, para iterar rapido. full: la corrida de la Entrega 1.",
        ),
    },
)
def deadlock_ingest():

    @task
    def resolve_window(**context) -> dict:
        """Calcula y loguea las 4 fechas de las dos ventanas. No toca la red.

        Es la tarea que responde "¿con que ventana corriste?" mirando el
        log, sin tener que leer el codigo del DAG.
        """
        params = context["params"]
        T = pendulum.parse(params["fecha_corte"])
        stats_inicio = T.subtract(days=params["dias_ventana_stats"])
        matches_fin = T.add(days=params["dias_ventana_matches"])

        ventana = {
            "fecha_corte": T.to_date_string(),
            "stats_min_ts": int(stats_inicio.timestamp()),
            "stats_max_ts": int(T.timestamp()),
            "matches_min_ts": int(T.timestamp()),
            "matches_max_ts": int(matches_fin.timestamp()),
        }
        log.info("Ventana de estadisticas: %s -> %s (T)",
                 stats_inicio.to_date_string(), ventana["fecha_corte"])
        log.info("Ventana de partidas: %s (T) -> %s",
                 ventana["fecha_corte"], matches_fin.to_date_string())
        return ventana

    @task(retries=3, retry_delay=pendulum.duration(seconds=30))
    def extract_hero_stats(ventana: dict, **context) -> str:
        """Bronce: /v1/analytics/hero-stats. Una sola llamada, los 38 heroes."""
        destino = _stats_bronze_path(ventana["fecha_corte"], "hero_stats")
        if destino.exists():
            log.info("hero_stats ya en bronce (corte=%s), no se vuelve a pedir.",
                      ventana["fecha_corte"])
            return str(destino)

        datos = client.get_hero_stats(ventana["stats_min_ts"], ventana["stats_max_ts"],
                                        context["params"]["min_matches_stats"])
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(datos), encoding="utf-8")
        log.info("hero_stats: %s heroes -> %s", len(datos), destino)
        return str(destino)

    @task(retries=3, retry_delay=pendulum.duration(seconds=30))
    def extract_counter_stats(ventana: dict, **context) -> str:
        """Bronce: /v1/analytics/hero-counter-stats. Matriz completa heroe-vs-heroe."""
        destino = _stats_bronze_path(ventana["fecha_corte"], "counter_stats")
        if destino.exists():
            log.info("counter_stats ya en bronce (corte=%s), no se vuelve a pedir.",
                      ventana["fecha_corte"])
            return str(destino)

        datos = client.get_counter_stats(ventana["stats_min_ts"], ventana["stats_max_ts"],
                                           context["params"]["min_matches_stats"])
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(datos), encoding="utf-8")
        log.info("counter_stats: %s pares -> %s", len(datos), destino)
        return str(destino)

    @task(retries=3, retry_delay=pendulum.duration(seconds=30))
    def extract_synergy_stats(ventana: dict, **context) -> str:
        """Bronce: /v1/analytics/hero-synergy-stats. Matriz completa de parejas del mismo equipo."""
        destino = _stats_bronze_path(ventana["fecha_corte"], "synergy_stats")
        if destino.exists():
            log.info("synergy_stats ya en bronce (corte=%s), no se vuelve a pedir.",
                      ventana["fecha_corte"])
            return str(destino)

        datos = client.get_synergy_stats(ventana["stats_min_ts"], ventana["stats_max_ts"],
                                           context["params"]["min_matches_stats"])
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(datos), encoding="utf-8")
        log.info("synergy_stats: %s pares -> %s", len(datos), destino)
        return str(destino)

    @task(retries=3, retry_delay=pendulum.duration(seconds=30))
    def extract_matches(ventana: dict, **context) -> list[str]:
        """Bronce: /v1/matches/metadata, paginado por match_id.

        Dado el volumen medido de la fuente (decenas de miles de partidas
        por dia), en la practica suele alcanzar con 1-2 paginas -- pero el
        cliente (deadlock.client.iter_matches) esta preparado para paginar
        de verdad si la ventana elegida es mas angosta.

        El cache por corte NO es todo-o-nada: si ya hay paginas de una
        corrida anterior (por ejemplo un `modo=subset` de prueba) pero
        juntan menos partidas que el objetivo actual, se completa la
        diferencia paginando desde el ultimo match_id ya bajado, en vez de
        reusarlas tal cual (lo que dejaria el dataset corto en silencio) o
        de volver a pedir todo desde el principio.
        """
        corte = ventana["fecha_corte"]
        destino_dir = _matches_bronze_dir(corte)
        params = context["params"]
        objetivo = 500 if params["modo"] == "subset" else params["objetivo_filas"]

        rutas = sorted(destino_dir.glob("pagina_*.json.gz")) if destino_dir.exists() else []
        juntadas, ultimo_match_id = 0, None
        for ruta in rutas:
            with gzip.open(ruta, "rt", encoding="utf-8") as f:
                lote = json.load(f)
            juntadas += len(lote)
            if lote:
                ultimo_match_id = max(ultimo_match_id or 0, max(m["match_id"] for m in lote))

        if rutas and juntadas >= objetivo:
            log.info("Partidas ya en bronce (corte=%s): %s pagina(s), %s partidas "
                      "(>= objetivo %s). No se vuelve a pedir.",
                      corte, len(rutas), juntadas, objetivo)
            return [str(p) for p in rutas]
        if rutas:
            log.info("Bronce parcial (corte=%s): %s de %s partidas ya bajadas, "
                      "se completa la diferencia.", corte, juntadas, objetivo)

        destino_dir.mkdir(parents=True, exist_ok=True)
        siguiente_pagina = len(rutas)
        for lote in client.iter_matches(
                ventana["matches_min_ts"], ventana["matches_max_ts"],
                objetivo - juntadas,
                min_match_id=(ultimo_match_id + 1) if ultimo_match_id is not None else None):
            destino = destino_dir / f"pagina_{siguiente_pagina:05d}.json.gz"
            with gzip.open(destino, "wt", encoding="utf-8") as f:
                json.dump(lote, f)
            rutas.append(str(destino))
            log.info("Pagina %s: %s partidas -> %s", siguiente_pagina, len(lote), destino)
            siguiente_pagina += 1

        if not rutas:
            raise ValueError(
                f"la ventana de partidas ({corte} -> +{params['dias_ventana_matches']}d) "
                "no devolvio ninguna partida")
        return [str(p) for p in rutas]

    @task
    def validate_bronze(p_hero: str, p_counter: str, p_synergy: str,
                          rutas_matches: list[str]) -> bool:
        """Chequeos duros sobre el bronce. Si algo esta mal, falla acá y no
        se gasta trabajo armando la plata sobre datos rotos."""
        hero_stats = json.loads(Path(p_hero).read_text(encoding="utf-8"))
        counter_stats = json.loads(Path(p_counter).read_text(encoding="utf-8"))
        synergy_stats = json.loads(Path(p_synergy).read_text(encoding="utf-8"))

        problemas = []
        if len(hero_stats) != len(schema.HEROES):
            problemas.append(f"hero_stats trajo {len(hero_stats)} heroes, "
                              f"se esperaban {len(schema.HEROES)}")
        if not counter_stats:
            problemas.append("hero-counter-stats vino vacio")
        if not synergy_stats:
            problemas.append("hero-synergy-stats vino vacio")
        if not rutas_matches:
            problemas.append("no se bajo ninguna pagina de partidas")

        if problemas:
            raise ValueError("Bronce invalido:\n  - " + "\n  - ".join(problemas))

        log.info("Bronce OK: %s heroes, %s pares counter, %s pares sinergia, %s pagina(s) de partidas",
                  len(hero_stats), len(counter_stats), len(synergy_stats), len(rutas_matches))
        return True

    @task
    def build_stat_tables(p_hero: str, p_counter: str, p_synergy: str,
                            ventana: dict, **context) -> str:
        """Plata: aplica shrinkage y arma las 3 tablas de consulta. No toca la red."""
        hero_stats = json.loads(Path(p_hero).read_text(encoding="utf-8"))
        counter_stats = json.loads(Path(p_counter).read_text(encoding="utf-8"))
        synergy_stats = json.loads(Path(p_synergy).read_text(encoding="utf-8"))
        k = context["params"]["k_shrinkage"]

        tablas = _build_stat_tables(hero_stats, counter_stats, synergy_stats, k)

        destino = _silver_dir(ventana["fecha_corte"]) / "stat_tables.pkl"
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "wb") as f:
            pickle.dump(tablas, f)

        log.info("Tablas de stats (k=%s): %s heroes, %s pares counter, %s pares sinergia -> %s",
                  k, len(tablas["hero"]), len(tablas["counter"]), len(tablas["synergy"]), destino)
        return str(destino)

    @task
    def build_matches_table(rutas: list[str], ventana: dict) -> str:
        """Plata: de las paginas de bronce a una fila por partida, con filtro de calidad.

        Descarta partidas con algun equipo incompleto, heroes fuera de
        catalogo, o sin ganador definido -- y loguea cuantas se descartaron
        por cada motivo, que es el numero que hay que llevar a la defensa
        para explicar cualquier diferencia entre "partidas pedidas" y
        "filas en el CSV final".

        Todavia NO calcula las 66 features: solo deja heroes_a/heroes_b
        (listas de hero_id) listas para que build_features las cruce
        contra las tablas de stats.
        """
        import pandas as pd

        filas, leidas, descartadas = [], 0, 0
        for ruta in rutas:
            with gzip.open(ruta, "rt", encoding="utf-8") as f:
                lote = json.load(f)
            for raw in lote:
                leidas += 1
                base = parse_match(raw)
                if base is None:
                    descartadas += 1
                    continue
                filas.append(base)

        if not filas:
            raise ValueError("ninguna partida de la ventana paso el filtro de calidad")

        df = pd.DataFrame(filas)
        destino = _silver_dir(ventana["fecha_corte"]) / "matches_base.parquet"
        destino.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(destino, index=False)

        log.info("%s partidas leidas, %s descartadas por el filtro de calidad, "
                  "%s en la tabla base -> %s", leidas, descartadas, len(df), destino)
        return str(destino)

    @task
    def build_features(p_base: str, p_tablas: str) -> str:
        """Plata: cruza la base de partidas contra las tablas de stats -> 66 features.

        Es la unica tarea que sabe el esquema completo de 73 columnas
        (schema.COLUMNS). Si algun dia cambia k_shrinkage, alcanza con
        re-correr build_stat_tables + esta tarea: no hace falta volver a
        pedirle nada a la API ni reparsear el bronce de partidas.
        """
        import pandas as pd

        base_df = pd.read_parquet(p_base)
        with open(p_tablas, "rb") as f:
            tablas = pickle.load(f)

        filas = [build_match_row(fila.to_dict(), tablas)
                 for _, fila in base_df.iterrows()]
        df = pd.DataFrame(filas)[schema.COLUMNS]

        destino = Path(p_base).parent / "features.parquet"
        df.to_parquet(destino, index=False)
        log.info("features: %s filas x %s columnas -> %s", len(df), len(df.columns), destino)
        return str(destino)

    @task
    def quality_gate(p_features: str) -> str:
        """Los 6 chequeos medibles de la Entrega 1. Si alguno falla, el DAG falla."""
        import pandas as pd

        df = pd.read_parquet(p_features)

        problemas = []
        if not df["match_id"].is_unique:
            problemas.append(f"{df['match_id'].duplicated().sum()} match_id repetidos")
        if len(df) < 1000:
            problemas.append(f"muy pocas filas: {len(df)} (se pide >1.000)")
        if df.shape[1] < 5:
            problemas.append(f"muy pocas columnas: {df.shape[1]} (se piden >=5)")
        columnas_vacias = df.columns[df.isna().all()].tolist()
        if columnas_vacias:
            problemas.append(f"columnas 100% nulas: {columnas_vacias}")
        for c in schema.OBLIGATORIAS:
            if c in df.columns and df[c].isna().any():
                problemas.append(f"{c} tiene {df[c].isna().sum()} nulos y no deberia")

        if problemas:
            raise ValueError("quality_gate fallo:\n  - " + "\n  - ".join(problemas))

        nulos = df.isna().mean().sort_values(ascending=False)
        log.info("quality_gate OK: %s filas x %s columnas", len(df), df.shape[1])
        log.info("Nulos por columna (top 5):\n%s", nulos.head(5).to_string())
        log.info("Tipos de dato:\n%s", df.dtypes.value_counts().to_string())
        return p_features

    @task
    def export_csv(p_features: str, **context) -> str:
        """Escribe el CSV entregable, fechado con la corrida."""
        import pandas as pd

        df = pd.read_parquet(p_features)
        dag_run = context["dag_run"]
        momento = dag_run.logical_date or dag_run.run_after
        ds = momento.date().isoformat()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destino = OUTPUT_DIR / f"deadlock_{ds}.csv"
        df.to_csv(destino, index=False)
        log.info("Dataset final: %s filas x %s columnas -> %s", len(df), df.shape[1], destino)
        return str(destino)

    ventana = resolve_window()
    p_hero = extract_hero_stats(ventana)
    p_counter = extract_counter_stats(ventana)
    p_synergy = extract_synergy_stats(ventana)
    p_matches = extract_matches(ventana)

    bronce_ok = validate_bronze(p_hero, p_counter, p_synergy, p_matches)

    p_tablas = build_stat_tables(p_hero, p_counter, p_synergy, ventana)
    p_base = build_matches_table(p_matches, ventana)
    # Unica dependencia declarada a mano: no construir plata sin validar el
    # bronce primero. El resto del grafo se infiere de pasarse argumentos.
    bronce_ok >> [p_tablas, p_base]

    export_csv(quality_gate(build_features(p_base, p_tablas)))


deadlock_ingest()
