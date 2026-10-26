"""
Cliente HTTP de la Deadlock API (https://api.deadlock-api.com).

A diferencia de sofifa.com (fifa/sofifa.py), esta fuente es una API REST JSON
estable, sin Cloudflare ni necesidad de navegador: alcanza con `requests`.

Los 3 endpoints de analytics (hero-stats, hero-counter-stats,
hero-synergy-stats) NO reciben hero_id: cada uno devuelve la tabla completa
en una sola llamada (confirmado en vivo antes de escribir este modulo).
matches/metadata se pagina por match_id porque un solo pedido puede traer
como maximo 10.000 filas.
"""
import logging
import time

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.deadlock-api.com"

# Limites compartidos entre los 3 endpoints de analytics: 200req/min por IP.
# matches/metadata tiene un limite mas estricto: 10req/min por IP.
_ANALYTICS_SLEEP_S = 0.3
_MATCHES_SLEEP_S = 6.5


def _get(path: str, params: dict, retries: int = 3, backoff_s: float = 5.0,
         status_vacio: int | None = None) -> dict | list:
    """GET con reintentos ante fallas transitorias (429/5xx/timeout).

    No hace falta un sensor ni una rama de degradacion como en fifa_ingest:
    esta fuente no tiene bloqueo tipo Cloudflare, asi que el unico modo de
    falla esperable es rate limiting transitorio, y eso lo resuelve un
    reintento con backoff a nivel de tarea.

    `status_vacio`: algunos endpoints (matches/metadata) devuelven ese
    codigo -- no 200 con lista vacia -- cuando el filtro no matchea ninguna
    fila. Eso NO es una falla transitoria (reintentarlo no cambia nada: la
    ventana sigue sin partidas), asi que se devuelve `[]` de una sin gastar
    los `retries`. Confirmado en vivo: matches/metadata devuelve
    404 {"error": "No matches found"} en ese caso.
    """
    url = f"{BASE_URL}{path}"
    ultimo_error = None
    for intento in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if status_vacio is not None and resp.status_code == status_vacio:
                log.info("%s: sin resultados para este filtro (HTTP %s). "
                          "No es un error, la ventana esta vacia.", path, status_vacio)
                return []
            if resp.status_code == 429:
                espera = backoff_s * intento
                log.warning("429 de %s (intento %s/%s). Espero %ss.",
                            path, intento, retries, espera)
                time.sleep(espera)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            ultimo_error = e
            log.warning("Error en %s (intento %s/%s): %s", path, intento, retries, e)
            time.sleep(backoff_s * intento)
    raise RuntimeError(f"{path} fallo tras {retries} intentos: {ultimo_error}")


def get_hero_stats(min_ts: int, max_ts: int, min_matches: int = 1) -> list[dict]:
    """/v1/analytics/hero-stats - tabla completa de heroes, una sola llamada."""
    return _get("/v1/analytics/hero-stats", {
        "min_unix_timestamp": min_ts,
        "max_unix_timestamp": max_ts,
        "game_mode": "normal",
        "match_mode": "ranked",
        "min_matches": min_matches,
    })


def get_counter_stats(min_ts: int, max_ts: int, min_matches: int = 1) -> list[dict]:
    """/v1/analytics/hero-counter-stats - matriz completa heroe-vs-heroe."""
    return _get("/v1/analytics/hero-counter-stats", {
        "min_unix_timestamp": min_ts,
        "max_unix_timestamp": max_ts,
        "game_mode": "normal",
        "match_mode": "ranked",
        "min_matches": min_matches,
        # requests serializa el bool de Python como "False" (mayuscula); la
        # API espera literalmente "true"/"false" en minuscula y devuelve
        # 400 si no coincide.
        "same_lane_filter": "false",
    })


def get_synergy_stats(min_ts: int, max_ts: int, min_matches: int = 1) -> list[dict]:
    """/v1/analytics/hero-synergy-stats - matriz completa de parejas del mismo equipo."""
    return _get("/v1/analytics/hero-synergy-stats", {
        "min_unix_timestamp": min_ts,
        "max_unix_timestamp": max_ts,
        "game_mode": "normal",
        "match_mode": "ranked",
        "min_matches": min_matches,
    })


def iter_matches(min_ts: int, max_ts: int, objetivo_filas: int, limit: int = 10000,
                  min_match_id: int | None = None):
    """Pagina /v1/matches/metadata por match_id ascendente.

    Devuelve un generador de paginas (listas de partidas). En la practica,
    dado el volumen medido de la fuente (decenas de miles de partidas por
    dia), suele alcanzar con 1-2 paginas -- pero el loop esta preparado para
    ventanas mas angostas donde haga falta paginar de verdad.

    `min_match_id` permite retomar la paginacion desde donde quedo un
    bronce parcial (ver extract_matches en el DAG: si ya hay paginas de una
    corrida anterior con menos partidas de las pedidas, se completa la
    diferencia en vez de volver a bajar todo desde el principio).

    Respeta el limite de 10req/min durmiendo entre paginas.
    """
    juntadas = 0
    pagina = 0
    while juntadas < objetivo_filas:
        params = {
            "min_unix_timestamp": min_ts,
            "max_unix_timestamp": max_ts,
            "game_mode": "normal",
            "match_mode": "ranked",
            "order_by": "match_id",
            "order_direction": "asc",
            "limit": min(limit, objetivo_filas - juntadas),
            "include_player_info": "true",  # ver nota de same_lane_filter arriba
            "format": "json",
        }
        if min_match_id is not None:
            params["min_match_id"] = min_match_id

        if pagina > 0:
            time.sleep(_MATCHES_SLEEP_S)
        lote = _get("/v1/matches/metadata", params, status_vacio=404)
        pagina += 1

        if not lote:
            if pagina == 1:
                log.warning("La ventana de partidas no tiene NINGUNA partida que cumpla "
                             "el filtro (game_mode=%s, match_mode=%s). Revisar si "
                             "fecha_corte cae en un periodo con poco volumen para ese modo.",
                             params["game_mode"], params["match_mode"])
            else:
                log.info("Ventana de partidas agotada tras %s pagina(s).", pagina - 1)
            return

        yield lote
        juntadas += len(lote)
        min_match_id = max(m["match_id"] for m in lote) + 1

        if len(lote) < params["limit"]:
            log.info("Ultima pagina de la ventana (%s filas, se pidieron %s).",
                      len(lote), params["limit"])
            return
