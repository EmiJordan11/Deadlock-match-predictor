"""
### Dataset canonico Deadlock -- Ciencia de Datos, UTN FRM 2026

Construye el dataset de partidas de Deadlock (deadlock-api.com) para
responder la pregunta del proyecto: ¿se puede saber quien gana una partida
sabiendo solo la composicion de heroes de los dos equipos?

El diseño completo de las columnas, y por que cada una es como es, esta
documentado en docs/decisiones-diseno-dataset.md y
docs/estructura-dataset-final.md. Este DAG materializa esas decisiones.

**Catalogo de heroes por corte.** Valve agrega heroes cada pocas semanas.
El DAG ya no usa una lista fija: `build_catalogo` arma el catalogo de cada
corte con la API de assets, las estadisticas de la ventana y las partidas
(ver `deadlock.catalogo`), y una columna de composicion por heroe. Si
aparece un heroe que el registro no tenia, lo agrega con un nombre de
columna que ya no cambia y lo avisa en el log. Las 28 ventajas no cambian:
son agregados sobre los heroes de cada equipo.

La plata va a `silver/catalogo_dinamico/`, separada de los cortes de la
Entrega 3 (`silver/corte=T/`, 38 heroes), que leen el notebook de modelado,
la aplicacion y los experimentos: correr este DAG no cambia ninguno de esos
resultados.

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
import hashlib
import json
import logging
import pickle
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, task

from deadlock import catalogo as cat
from deadlock import client, schema
from deadlock.transform import MOTIVOS_DESCARTE, build_match_row, motivo_descarte, parse_match
from deadlock.transform import build_stat_tables as _build_stat_tables

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/usr/local/airflow/include/output")
BRONCE_DIR = OUTPUT_DIR / "bronze"
# La plata con el catalogo de cada corte. Los cortes de la Entrega 3 quedan en
# silver/corte=T/ con sus 38 heroes y no se tocan.
PLATA_DIR = OUTPUT_DIR / "silver" / "catalogo_dinamico"
REGISTRO_HEROES = PLATA_DIR / "registro_heroes.json"


def _stats_bronze_path(corte: str, nombre: str, dias: int) -> Path:
    return BRONCE_DIR / f"corte={corte}" / "stats" / f"dias={dias}" / f"{nombre}.json"


def _heroes_bronze_path(corte: str) -> Path:
    return BRONCE_DIR / f"corte={corte}" / "heroes" / "heroes.json"


def _leer_pagina(ruta: str) -> list[dict]:
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def _matches_bronze_dir(corte: str) -> Path:
    return BRONCE_DIR / f"corte={corte}" / "matches"


def _silver_dir(corte: str, dias: int, k: int) -> Path:
    return PLATA_DIR / f"corte={corte}" / f"dias={dias}" / f"k={k}"


def _stats_request(ventana: dict, nombre: str) -> dict:
    endpoint = {"hero": "hero-stats", "counter": "hero-counter-stats",
                "synergy": "hero-synergy-stats"}[nombre]
    params = {"min_unix_timestamp": ventana["stats_min_ts"],
              "max_unix_timestamp": ventana["stats_max_ts"],
              "game_mode": "normal", "match_mode": "ranked",
              "min_matches": ventana["min_matches_stats"]}
    if nombre == "counter":
        params["same_lane_filter"] = "false"
    return {"endpoint": f"/v1/analytics/{endpoint}", "params": params}


def _write_stats(destino: Path, datos: list, request: dict) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(datos), encoding="utf-8")
    meta = {"request": request, "rows": len(datos),
            "sha256": hashlib.sha256(destino.read_bytes()).hexdigest(),
            "retrieved_at": pendulum.now("UTC").isoformat()}
    destino.with_suffix(".request.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _bronce_stats_cacheado(destino: Path, request: dict) -> list | None:
    """Reusa un archivo de bronce de stats solo si existe Y tiene contenido.

    Un archivo vacio (`[]`) casi siempre es la marca de una corrida anterior
    contra una ventana/filtro que no tenia datos (ej. un fecha_corte con
    poco volumen para el match_mode de ese momento) -- no una respuesta
    valida para cachear. Sin este chequeo, cambiar el filtro y volver a
    correr con el mismo fecha_corte reusaria ese vacio para siempre, sin
    volver a pedirle nada a la API.
    """
    if not destino.exists():
        return None
    meta_path = destino.with_suffix(".request.json")
    if not meta_path.exists():
        return None  # No adivinar la ventana de un caché sin procedencia.
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("request") != request or meta.get("sha256") != hashlib.sha256(destino.read_bytes()).hexdigest():
        return None
    datos = json.loads(destino.read_text(encoding="utf-8"))
    return datos or None


@dag(
    dag_id="deadlock_ingest",
    # Con T fijo por Param, agendar no aporta nada: "automatizado" acá
    # significa que ningun dato se toca a mano, no que corra solo. Se
    # dispara manualmente para cada corte que se quiera producir.
    schedule=None,
    start_date=pendulum.datetime(2026, 9, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    # Una corrida a la vez: comparten el registro de heroes y el limite de
    # 10 pedidos por minuto de /v1/matches/metadata.
    max_active_runs=1,
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
            title="Partidas a juntar",
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
        "force": Param(
            False, type="boolean",
            title="Forzar la corrida",
            description=("Ignora el bronce en cache para este fecha_corte y vuelve a "
                         "pedirle todo a la API, aunque ya existan archivos. Pensado "
                         "para cuando el bronce cacheado quedo vacio o desactualizado "
                         "por una corrida anterior con otros filtros -- misma idea que "
                         "'force' en fifa_ingest.py."),
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
            "dias_ventana_stats": params["dias_ventana_stats"],
            "dias_ventana_matches": params["dias_ventana_matches"],
            "k_shrinkage": params["k_shrinkage"],
            "objetivo_filas": params["objetivo_filas"],
            "min_matches_stats": params["min_matches_stats"],
            "game_mode": "normal", "match_mode": "ranked",
            "stats_min_ts": int(stats_inicio.timestamp()),
            "stats_max_ts": int(T.timestamp()) - 1,
            "matches_min_ts": int(T.timestamp()),
            "matches_max_ts": int(matches_fin.timestamp()) - 1,
        }
        log.info("Ventana de estadisticas: %s -> %s (T)",
                 stats_inicio.to_date_string(), ventana["fecha_corte"])
        log.info("Ventana de partidas: %s (T) -> %s",
                 ventana["fecha_corte"], matches_fin.to_date_string())
        return ventana

    @task(retries=3, retry_delay=pendulum.duration(seconds=30))
    def extract_heroes(ventana: dict, **context) -> str:
        """Bronce: /v1/assets/heroes, los heroes del juego (nombre, imagenes, si es jugable).

        La API devuelve el estado de HOY, no el del corte. Se guarda una vez
        por corte y se reusa, para que volver a correr el mismo corte arme
        el mismo catalogo.
        """
        destino = _heroes_bronze_path(ventana["fecha_corte"])
        request = client.HEROES_REQUEST
        if not context["params"]["force"] and _bronce_stats_cacheado(destino, request) is not None:
            log.info("heroes ya en bronce (corte=%s), no se vuelve a pedir.", ventana["fecha_corte"])
            return str(destino)

        datos = client.get_heroes()
        _write_stats(destino, datos, request)
        log.info("heroes: %s en la API de assets -> %s", len(datos), destino)
        return str(destino)

    @task(retries=3, retry_delay=pendulum.duration(seconds=30))
    def extract_hero_stats(ventana: dict, **context) -> str:
        """Bronce: /v1/analytics/hero-stats. Una sola llamada, una fila por heroe jugado."""
        destino = _stats_bronze_path(ventana["fecha_corte"], "hero_stats", ventana["dias_ventana_stats"])
        request = _stats_request(ventana, "hero")
        if not context["params"]["force"] and _bronce_stats_cacheado(destino, request) is not None:
            log.info("hero_stats ya en bronce (corte=%s), no se vuelve a pedir.",
                      ventana["fecha_corte"])
            return str(destino)

        datos = client.get_hero_stats(ventana["stats_min_ts"], ventana["stats_max_ts"],
                                        context["params"]["min_matches_stats"])
        _write_stats(destino, datos, request)
        log.info("hero_stats: %s heroes -> %s", len(datos), destino)
        return str(destino)

    @task(retries=3, retry_delay=pendulum.duration(seconds=30))
    def extract_counter_stats(ventana: dict, **context) -> str:
        """Bronce: /v1/analytics/hero-counter-stats. Matriz completa heroe-vs-heroe."""
        destino = _stats_bronze_path(ventana["fecha_corte"], "counter_stats", ventana["dias_ventana_stats"])
        request = _stats_request(ventana, "counter")
        if not context["params"]["force"] and _bronce_stats_cacheado(destino, request) is not None:
            log.info("counter_stats ya en bronce (corte=%s), no se vuelve a pedir.",
                      ventana["fecha_corte"])
            return str(destino)

        datos = client.get_counter_stats(ventana["stats_min_ts"], ventana["stats_max_ts"],
                                           context["params"]["min_matches_stats"])
        _write_stats(destino, datos, request)
        log.info("counter_stats: %s pares -> %s", len(datos), destino)
        return str(destino)

    @task(retries=3, retry_delay=pendulum.duration(seconds=30))
    def extract_synergy_stats(ventana: dict, **context) -> str:
        """Bronce: /v1/analytics/hero-synergy-stats. Matriz completa de parejas del mismo equipo."""
        destino = _stats_bronze_path(ventana["fecha_corte"], "synergy_stats", ventana["dias_ventana_stats"])
        request = _stats_request(ventana, "synergy")
        if not context["params"]["force"] and _bronce_stats_cacheado(destino, request) is not None:
            log.info("synergy_stats ya en bronce (corte=%s), no se vuelve a pedir.",
                      ventana["fecha_corte"])
            return str(destino)

        datos = client.get_synergy_stats(ventana["stats_min_ts"], ventana["stats_max_ts"],
                                           context["params"]["min_matches_stats"])
        _write_stats(destino, datos, request)
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
        corrida anterior (por ejemplo con un `objetivo_filas` mas chico)
        pero juntan menos partidas que el objetivo actual, se completa la
        diferencia paginando desde el ultimo match_id ya bajado, en vez de
        reusarlas tal cual (lo que dejaria el dataset corto en silencio) o
        de volver a pedir todo desde el principio.

        Con `force=True` se ignora el bronce existente por completo (se
        borra y se vuelve a bajar desde cero) -- pensado para cuando las
        paginas cacheadas quedaron de una corrida con otros filtros
        (game_mode/match_mode), donde "completar la diferencia" no sirve
        porque las partidas viejas ya no corresponden al filtro actual.
        """
        corte = ventana["fecha_corte"]
        destino_dir = _matches_bronze_dir(corte)
        params = context["params"]
        objetivo = params["objetivo_filas"]

        if params["force"] and destino_dir.exists():
            for p in destino_dir.glob("pagina_*.json.gz"):
                p.unlink()
            log.info("force=True: se borro el bronce de partidas existente (corte=%s) "
                      "y se vuelve a bajar todo.", corte)

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
                          rutas_matches: list[str], p_heroes: str) -> bool:
        """Chequeos duros sobre el bronce. Si algo esta mal, falla acá y no
        se gasta trabajo armando la plata sobre datos rotos.

        Ya no se exige que los heroes de las estadisticas esten en una lista
        fija: un heroe nuevo es un dato valido, no un error. El catalogo lo
        arma build_catalogo.
        """
        hero_stats = json.loads(Path(p_hero).read_text(encoding="utf-8"))
        counter_stats = json.loads(Path(p_counter).read_text(encoding="utf-8"))
        synergy_stats = json.loads(Path(p_synergy).read_text(encoding="utf-8"))
        heroes_api = json.loads(Path(p_heroes).read_text(encoding="utf-8"))

        problemas = []
        if not hero_stats:
            problemas.append("hero-stats vino vacio")
        if not cat.jugables(heroes_api):
            problemas.append("la API de assets no devolvio ningun heroe jugable")
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
    def build_catalogo(p_heroes: str, p_hero: str, rutas_matches: list[str], ventana: dict) -> dict:
        """Plata: el catalogo de heroes del corte. Es la tarea que reacciona a los heroes nuevos.

        Junta los heroes jugables de la API de assets, los que tienen
        estadisticas en la ventana y los que aparecen en las partidas. Un
        heroe que el registro no tenia se agrega con un nombre de columna
        que ya no cambia, y se avisa en el log. Devuelve el catalogo (lo
        usan las tareas siguientes) y lo guarda como catalogo.json junto a
        las features.
        """
        heroes_api = json.loads(Path(p_heroes).read_text(encoding="utf-8"))
        hero_stats = json.loads(Path(p_hero).read_text(encoding="utf-8"))
        en_partidas = cat.apariciones(raw for ruta in rutas_matches for raw in _leer_pagina(ruta))

        registro = cat.cargar_registro(REGISTRO_HEROES)
        catalogo = cat.armar_catalogo(heroes_api, hero_stats, en_partidas, registro, ventana["fecha_corte"])
        cat.guardar_registro(REGISTRO_HEROES, registro)

        carpeta = _silver_dir(ventana["fecha_corte"], ventana["dias_ventana_stats"], ventana["k_shrinkage"])
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "catalogo.json").write_text(json.dumps(catalogo, indent=2, ensure_ascii=False), encoding="utf-8")

        nombre = {d["id"]: d["nombre"] for d in catalogo["detalle"]}
        log.info("Catalogo del corte %s: %s heroes (%s columnas de composicion) -> %s",
                  ventana["fecha_corte"], len(catalogo["heroes"]), len(catalogo["heroes"]), carpeta / "catalogo.json")
        if catalogo["nuevos_en_registro"]:
            log.warning("HEROES NUEVOS, agregados al registro: %s",
                        ", ".join(f"{i} {nombre[i]}" for i in catalogo["nuevos_en_registro"]))
        if catalogo["fuera_del_catalogo_entrega3"]:
            log.info("Fuera del catalogo de la Entrega 3: %s",
                      ", ".join(f"{i} {nombre[i]} ({en_partidas.get(i, 0)} partidas)" for i in catalogo["fuera_del_catalogo_entrega3"]))
        if catalogo["sin_estadisticas"]:
            log.warning("Se jugaron sin estadisticas en la ventana (su ventaja de winrate vale 0, neutra): %s",
                        ", ".join(f"{i} {nombre[i]}" for i in catalogo["sin_estadisticas"]))
        if catalogo["fuera_de_la_api"]:
            log.warning("No estan en la API de assets (nombre provisorio): %s",
                        ", ".join(f"{i} {nombre[i]}" for i in catalogo["fuera_de_la_api"]))
        return catalogo

    @task
    def build_stat_tables(p_hero: str, p_counter: str, p_synergy: str,
                            ventana: dict, catalogo: dict, **context) -> str:
        """Plata: aplica shrinkage y arma las 3 tablas de consulta. No toca la red."""
        hero_stats = json.loads(Path(p_hero).read_text(encoding="utf-8"))
        counter_stats = json.loads(Path(p_counter).read_text(encoding="utf-8"))
        synergy_stats = json.loads(Path(p_synergy).read_text(encoding="utf-8"))
        k = context["params"]["k_shrinkage"]

        tablas = _build_stat_tables(hero_stats, counter_stats, synergy_stats, k)

        carpeta = _silver_dir(ventana["fecha_corte"], ventana["dias_ventana_stats"], k)
        destino = carpeta / "stat_tables.pkl"
        destino.parent.mkdir(parents=True, exist_ok=True)
        parametros = dict(ventana, hero_ids=[i for i, _ in cat.heroes_de(catalogo)])
        parametros["stats_sources"] = [
            {"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
            for p in (p_hero, p_counter, p_synergy)
        ]
        (carpeta / "parametros.json").write_text(json.dumps(parametros, indent=2), encoding="utf-8")
        with open(destino, "wb") as f:
            pickle.dump(tablas, f)

        log.info("Tablas de stats (k=%s): %s heroes, %s pares counter, %s pares sinergia -> %s",
                  k, len(tablas["hero"]), len(tablas["counter"]), len(tablas["synergy"]), destino)
        return str(destino)

    @task
    def build_matches_table(rutas: list[str], ventana: dict, catalogo: dict) -> str:
        """Plata: de las paginas de bronce a una fila por partida, con filtro de calidad.

        Descarta partidas con algun equipo incompleto, un heroe repetido o
        sin ganador definido -- y cuenta cuantas se descartaron por cada
        motivo (filtro_calidad.json y el log), que es el numero que hay que
        llevar a la defensa para explicar cualquier diferencia entre
        "partidas pedidas" y "filas en el CSV final". Los heroes se validan
        contra el catalogo del corte, asi que un heroe nuevo ya no es motivo
        de descarte.

        Todavia NO calcula las features: solo deja heroes_a/heroes_b
        (listas de hero_id) listas para que build_features las cruce
        contra las tablas de stats.
        """
        import pandas as pd

        hero_ids = {i for i, _ in cat.heroes_de(catalogo)}
        filas, motivos = [], dict.fromkeys(MOTIVOS_DESCARTE, 0)
        leidas = 0
        for ruta in rutas:
            with gzip.open(ruta, "rt", encoding="utf-8") as f:
                lote = json.load(f)
            for raw in lote:
                leidas += 1
                motivo = motivo_descarte(raw, hero_ids)
                if motivo is not None:
                    motivos[motivo] += 1
                    continue
                filas.append(parse_match(raw, hero_ids))

        if not filas:
            raise ValueError("ninguna partida de la ventana paso el filtro de calidad")

        df = pd.DataFrame(filas)
        carpeta = _silver_dir(ventana["fecha_corte"], ventana["dias_ventana_stats"], ventana["k_shrinkage"])
        destino = carpeta / "matches_base.parquet"
        destino.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(destino, index=False)
        filtro = {"leidas": leidas, "descartadas": dict(motivos), "en_la_tabla_base": len(df)}
        (carpeta / "filtro_calidad.json").write_text(json.dumps(filtro, indent=2), encoding="utf-8")

        log.info("%s partidas leidas, %s descartadas por el filtro de calidad %s, "
                  "%s en la tabla base -> %s", leidas, sum(motivos.values()), motivos, len(df), destino)
        return str(destino)

    @task
    def build_features(p_base: str, p_tablas: str, catalogo: dict) -> str:
        """Plata: cruza la base de partidas contra las tablas de stats -> features.

        Es la unica tarea que sabe el esquema completo (schema.columnas del
        catalogo del corte): una columna de composicion por heroe del
        catalogo, mas las 28 ventajas, los metadatos y el ganador. Si algun
        dia cambia k_shrinkage, alcanza con re-correr build_stat_tables +
        esta tarea: no hace falta volver a pedirle nada a la API ni
        reparsear el bronce de partidas.
        """
        import pandas as pd

        heroes = cat.heroes_de(catalogo)
        base_df = pd.read_parquet(p_base)
        with open(p_tablas, "rb") as f:
            tablas = pickle.load(f)

        filas = [build_match_row(fila.to_dict(), tablas, heroes)
                 for _, fila in base_df.iterrows()]
        df = pd.DataFrame(filas)[schema.columnas(heroes)]

        destino = Path(p_base).parent / "features.parquet"
        df.to_parquet(destino, index=False)
        p_meta = destino.parent / "parametros.json"
        parametros = json.loads(p_meta.read_text(encoding="utf-8"))
        parametros["features_sha256"] = hashlib.sha256(destino.read_bytes()).hexdigest()
        p_meta.write_text(json.dumps(parametros, indent=2), encoding="utf-8")
        log.info("features: %s filas x %s columnas -> %s", len(df), len(df.columns), destino)
        return str(destino)

    @task
    def quality_gate(p_features: str, catalogo: dict) -> str:
        """Los 6 chequeos medibles de la Entrega 1. Si alguno falla, el DAG falla."""
        import pandas as pd

        df = pd.read_parquet(p_features)
        heroes = cat.heroes_de(catalogo)

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
        faltan = [c for c in schema.columnas(heroes) if c not in df.columns]
        if faltan:
            problemas.append(f"faltan columnas del catalogo: {faltan}")
        for c in schema.obligatorias(heroes):
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
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Cada combinación tiene su propia salida: dos ventanas no se pisan.
        destino = Path(p_features).parent / "dataset.csv"
        df.to_csv(destino, index=False)
        log.info("Dataset final: %s filas x %s columnas -> %s", len(df), df.shape[1], destino)
        return str(destino)

    ventana = resolve_window()
    p_heroes = extract_heroes(ventana)
    p_hero = extract_hero_stats(ventana)
    p_counter = extract_counter_stats(ventana)
    p_synergy = extract_synergy_stats(ventana)
    p_matches = extract_matches(ventana)

    bronce_ok = validate_bronze(p_hero, p_counter, p_synergy, p_matches, p_heroes)

    catalogo = build_catalogo(p_heroes, p_hero, p_matches, ventana)
    # Unica dependencia declarada a mano: no construir plata sin validar el
    # bronce primero. El resto del grafo se infiere de pasarse argumentos
    # (toda la plata recibe el catalogo, que espera a la validacion).
    bronce_ok >> catalogo

    p_tablas = build_stat_tables(p_hero, p_counter, p_synergy, ventana, catalogo)
    p_base = build_matches_table(p_matches, ventana, catalogo)
    export_csv(quality_gate(build_features(p_base, p_tablas, catalogo), catalogo))


deadlock_ingest()
