"""
De JSON crudo de la Deadlock API a las 73 columnas de schema.COLUMNS.

Dos pasos separados a proposito (ver docs/decisiones-diseno-dataset.md y el
plan de la Entrega 1):

  1. parse_match(...)      -- SOLO parsea una partida cruda: metadatos +
                               la lista de heroes de cada equipo. No toca
                               las tablas de estadisticas. Lo usa
                               build_matches_table (tarea del DAG).
  2. build_match_row(...)  -- cruza esa base contra las tablas de stats
                               (ya con shrinkage aplicado) y arma las 66
                               columnas de features. Lo usa build_features.

Separarlos permite recalcular las features (por ejemplo, si cambia
k_shrinkage) sin volver a parsear las paginas de bronce de partidas.
"""
import statistics
from datetime import datetime, timezone
from itertools import combinations

from deadlock import schema
from deadlock.shrink import K_DEFAULT, shrink_diff, shrink_rate

# --- paso 1: partida cruda -> fila base --------------------------------------


def _parse_start_time(valor) -> str:
    """`start_time` llega como texto "YYYY-MM-DD HH:MM:SS" (UTC), no como
    Unix timestamp -- verificado con una llamada real antes de asumir el
    formato del OpenAPI spec (que no lo documenta con precision)."""
    if isinstance(valor, (int, float)):
        return datetime.fromtimestamp(valor, tz=timezone.utc).date().isoformat()
    return datetime.strptime(valor, "%Y-%m-%d %H:%M:%S").date().isoformat()


def parse_match(raw_match: dict) -> dict | None:
    """Devuelve la fila base de una partida, o None si no pasa el filtro de calidad.

    Motivos de descarte (se cuentan en build_matches_table para el log de
    "partidas leidas vs. partidas finales" que pide la defensa de la
    Entrega 1):
      - algun equipo no tiene exactamente 6 heroes validos (abandono u
        otro dato parcial)
      - hay un hero_id fuera del catalogo de 38 heroes activos
      - la partida no tiene un equipo ganador definido
    """
    equipos = {"Team0": [], "Team1": []}
    for jugador in raw_match.get("players") or []:
        team = jugador.get("team")
        hero_id = jugador.get("hero_id")
        if team in equipos and hero_id in schema.HERO_IDS:
            equipos[team].append(hero_id)

    if len(equipos["Team0"]) != 6 or len(equipos["Team1"]) != 6:
        return None

    winning_team = raw_match.get("winning_team")
    if winning_team not in ("Team0", "Team1"):
        return None

    badge0 = raw_match.get("average_badge_team0")
    badge1 = raw_match.get("average_badge_team1")
    average_badge = (badge0 + badge1) / 2 if badge0 is not None and badge1 is not None else None
    average_badge_diff = (badge0 - badge1) if badge0 is not None and badge1 is not None else None

    return {
        "match_id": raw_match["match_id"],
        "match_date": _parse_start_time(raw_match["start_time"]),
        "game_mode": raw_match.get("game_mode"),
        "match_mode": raw_match.get("match_mode"),
        "average_badge": average_badge,
        "average_badge_diff": average_badge_diff,
        "winner": "A" if winning_team == "Team0" else "B",
        "heroes_a": equipos["Team0"],
        "heroes_b": equipos["Team1"],
    }


# --- paso 2: tablas de estadisticas (con shrinkage ya aplicado) -------------


def build_stat_tables(hero_stats_raw: list[dict], counter_raw: list[dict],
                       synergy_raw: list[dict], k: int = K_DEFAULT) -> dict:
    """Convierte los 3 JSON crudos de analytics en tablas de consulta rapida.

    Todas las ventajas quedan ya centradas en 0 (positivo favorece al
    hero_id/pareja/equipo "propio") y ya ajustadas por shrinkage segun su
    propio matches_played -- ver Problema 4. build_match_row solo necesita
    buscar en estos diccionarios, sin volver a calcular nada estadistico.
    """
    hero_adv = {
        row["hero_id"]: shrink_rate(row["wins"], row["matches"], k)
        for row in hero_stats_raw
    }

    counter_adv = {}
    for row in counter_raw:
        n = row["matches_played"]
        clave = (row["hero_id"], row["enemy_hero_id"])
        counter_adv[clave] = {
            "winrate": shrink_rate(row["wins"], n, k),
            "kills": shrink_diff(_tasa_dif(row, "kills", n), n, k),
            "deaths": shrink_diff(_tasa_dif(row, "deaths", n), n, k),
            "assists": shrink_diff(_tasa_dif(row, "assists", n), n, k),
            "souls": shrink_diff(_tasa_dif(row, "networth", n), n, k),
        }

    synergy_adv = {
        frozenset((row["hero_id1"], row["hero_id2"])): shrink_rate(
            row["wins"], row["matches_played"], k)
        for row in synergy_raw
    }

    return {"hero": hero_adv, "counter": counter_adv, "synergy": synergy_adv}


def _tasa_dif(row: dict, campo: str, matches_played: int) -> float:
    """(propio - enemigo) / matches_played para kills/deaths/assists/networth."""
    if not matches_played:
        return 0.0
    return (row[campo] - row[f"enemy_{campo}"]) / matches_played


# --- lookups con valor neutro cuando falta el dato --------------------------


def _hero_adv(tablas: dict, hero_id: int) -> float:
    return tablas["hero"].get(hero_id, 0.0)


def _counter_adv(tablas: dict, hero_a: int, hero_b: int, metric: str) -> float:
    fila = tablas["counter"].get((hero_a, hero_b))
    return fila[metric] if fila else 0.0


def _synergy_adv(tablas: dict, hero_x: int, hero_y: int) -> float:
    return tablas["synergy"].get(frozenset((hero_x, hero_y)), 0.0)


# --- paso 2b: fila base + tablas -> las 66 columnas de features -------------


def build_match_row(base: dict, tablas: dict) -> dict:
    """A partir de una fila de build_matches_table, arma las 66 features.

    No toca la red ni vuelve a mirar el JSON crudo de la partida: todo lo
    que necesita ya esta en `base` (heroes_a/heroes_b) y en `tablas`
    (build_stat_tables). Por eso esta funcion se puede re-ejecutar barato
    si cambia k_shrinkage, sin reprocesar el bronce.
    """
    heroes_a, heroes_b = base["heroes_a"], base["heroes_b"]
    fila = {k: v for k, v in base.items() if k not in ("heroes_a", "heroes_b")}

    # --- composicion: +1 en A, -1 en B, 0 si no participa ---
    for hero_id, slug in schema.HEROES:
        if hero_id in heroes_a:
            fila[f"hero_{slug}"] = 1
        elif hero_id in heroes_b:
            fila[f"hero_{slug}"] = -1
        else:
            fila[f"hero_{slug}"] = 0

    # --- counters: agregados sobre las 36 parejas cruzadas reales ---
    for metric in schema.COUNTER_METRICS:
        valores = [_counter_adv(tablas, a, b, metric)
                   for a in heroes_a for b in heroes_b]
        fila[f"counter_{metric}_adv_mean"] = statistics.mean(valores)
        fila[f"counter_{metric}_adv_min"] = min(valores)
        fila[f"counter_{metric}_adv_max"] = max(valores)
        fila[f"counter_{metric}_adv_std"] = statistics.pstdev(valores)

    # --- sinergia: agregados de las 15+15 parejas internas, diferencia A-B ---
    syn_a = [_synergy_adv(tablas, x, y) for x, y in combinations(heroes_a, 2)]
    syn_b = [_synergy_adv(tablas, x, y) for x, y in combinations(heroes_b, 2)]
    fila["synergy_winrate_adv_mean"] = statistics.mean(syn_a) - statistics.mean(syn_b)
    fila["synergy_winrate_adv_min"] = min(syn_a) - min(syn_b)
    fila["synergy_winrate_adv_max"] = max(syn_a) - max(syn_b)
    fila["synergy_winrate_adv_std"] = statistics.pstdev(syn_a) - statistics.pstdev(syn_b)

    # --- winrate individual: agregados de los 6+6 heroes, diferencia A-B ---
    wr_a = [_hero_adv(tablas, h) for h in heroes_a]
    wr_b = [_hero_adv(tablas, h) for h in heroes_b]
    fila["hero_winrate_adv_mean"] = statistics.mean(wr_a) - statistics.mean(wr_b)
    fila["hero_winrate_adv_min"] = min(wr_a) - min(wr_b)
    fila["hero_winrate_adv_max"] = max(wr_a) - max(wr_b)
    fila["hero_winrate_adv_std"] = statistics.pstdev(wr_a) - statistics.pstdev(wr_b)

    return fila
