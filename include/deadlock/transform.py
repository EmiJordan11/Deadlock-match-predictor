"""
De JSON crudo de la Deadlock API a las columnas de schema.columnas(heroes).

Dos pasos separados a proposito (ver docs/decisiones-diseno-dataset.md y el
plan de la Entrega 1):

  1. parse_match(...)      -- SOLO parsea una partida cruda: metadatos +
                               la lista de heroes de cada equipo. No toca
                               las tablas de estadisticas. Lo usa
                               build_matches_table (tarea del DAG).
  2. build_match_row(...)  -- cruza esa base contra las tablas de stats
                               (ya con shrinkage aplicado) y arma las
                               columnas de features. Lo usa build_features.

Las dos funciones reciben el catalogo de heroes del corte (deadlock.catalogo).
Sin catalogo usan el de la Entrega 3 (schema.HEROES, 38 heroes y 73 columnas),
y dan exactamente lo mismo que antes.

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


MOTIVOS_DESCARTE = ("equipo_incompleto", "heroe_fuera_de_catalogo", "heroe_repetido", "sin_ganador")


def motivo_descarte(raw_match: dict, hero_ids: set[int] | None = None) -> str | None:
    """Por que una partida no pasa el filtro de calidad, o None si pasa.

    Se revisa en este orden y se devuelve el primer motivo:
      - equipo_incompleto: algun equipo no tiene exactamente 6 jugadores
        con heroe (abandono u otro dato parcial)
      - heroe_fuera_de_catalogo: un hero_id que no esta en el catalogo del
        corte. Con el catalogo dinamico no deberia pasar nunca: el catalogo
        incluye a todos los heroes que aparecen en las partidas.
      - heroe_repetido: el mismo heroe en los dos equipos
      - sin_ganador: la partida no tiene un equipo ganador definido
    """
    hero_ids = schema.HERO_IDS if hero_ids is None else hero_ids
    equipos = {"Team0": [], "Team1": []}
    for jugador in raw_match.get("players") or []:
        if jugador.get("team") in equipos and jugador.get("hero_id") is not None:
            equipos[jugador["team"]].append(jugador["hero_id"])
    todos = equipos["Team0"] + equipos["Team1"]
    if len(equipos["Team0"]) != 6 or len(equipos["Team1"]) != 6:
        return "equipo_incompleto"
    if any(h not in hero_ids for h in todos):
        return "heroe_fuera_de_catalogo"
    if len(set(todos)) != 12:
        return "heroe_repetido"
    if raw_match.get("winning_team") not in ("Team0", "Team1"):
        return "sin_ganador"
    return None


def parse_match(raw_match: dict, hero_ids: set[int] | None = None) -> dict | None:
    """Devuelve la fila base de una partida, o None si no pasa el filtro de calidad.

    Los motivos de descarte estan en motivo_descarte(); build_matches_table
    los cuenta uno por uno para el log de "partidas leidas vs. partidas
    finales" que pide la defensa de la Entrega 1. `hero_ids` es el catalogo
    del corte; sin el, se usa el de la Entrega 3.

    Nota sobre badges (verificado leyendo el bronce crudo, en ranked y en
    unranked): `average_badge_team0`/`average_badge_team1` los devuelve la
    API siempre en 0 en este endpoint -- no es un problema de match_mode,
    el desglose por equipo directamente no esta disponible aca. El campo
    agregado `average_badge` si viene poblado con datos reales (None en
    partidas unranked, un numero real en ranked), asi que se usa tal cual
    viene. Para `average_badge_diff` se reconstruye a mano un proxy
    promediando `player_rank_initial_display_rank` (mismo esquema
    tier*10+subrank que el badge oficial, documentado en el OpenAPI spec)
    de los jugadores de cada equipo. ADVERTENCIA para la defensa: este
    promedio es una aproximacion, no una descomposicion exacta de
    `average_badge` -- verificado con muestras reales, puede diferir del
    agregado de la API en varios puntos (probablemente porque `average_badge`
    usa una senal interna, tipo MMR oculto, no el simple promedio del rank
    de entrada visible). Se documenta como concesion explicita, no como el
    mismo numero partido en dos.
    """
    if motivo_descarte(raw_match, hero_ids) is not None:
        return None

    equipos = {"Team0": [], "Team1": []}
    ranks_iniciales = {"Team0": [], "Team1": []}
    for jugador in raw_match.get("players") or []:
        team = jugador.get("team")
        if team in equipos and jugador.get("hero_id") is not None:
            equipos[team].append(jugador["hero_id"])
        rango = jugador.get("player_rank_initial_display_rank")
        if team in ranks_iniciales and rango is not None:
            ranks_iniciales[team].append(rango)

    winning_team = raw_match["winning_team"]

    badge_a = statistics.mean(ranks_iniciales["Team0"]) if ranks_iniciales["Team0"] else None
    badge_b = statistics.mean(ranks_iniciales["Team1"]) if ranks_iniciales["Team1"] else None
    average_badge_diff = (badge_a - badge_b) if badge_a is not None and badge_b is not None else None

    return {
        "match_id": raw_match["match_id"],
        "match_date": _parse_start_time(raw_match["start_time"]),
        "game_mode": raw_match.get("game_mode"),
        "match_mode": raw_match.get("match_mode"),
        "average_badge": raw_match.get("average_badge"),
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


def build_match_row(base: dict, tablas: dict, heroes: list[tuple[int, str]] | None = None) -> dict:
    """A partir de una fila de build_matches_table, arma las features.

    No toca la red ni vuelve a mirar el JSON crudo de la partida: todo lo
    que necesita ya esta en `base` (heroes_a/heroes_b) y en `tablas`
    (build_stat_tables). Por eso esta funcion se puede re-ejecutar barato
    si cambia k_shrinkage, sin reprocesar el bronce.

    `heroes` es el catalogo del corte, [(id, slug)]: una columna de
    composicion por heroe. Sin el, se usa el de la Entrega 3. Un heroe sin
    estadisticas en la ventana (recien lanzado) aporta ventaja 0: neutro.
    """
    heroes_a, heroes_b = base["heroes_a"], base["heroes_b"]
    fila = {k: v for k, v in base.items() if k not in ("heroes_a", "heroes_b")}

    # --- composicion: +1 en A, -1 en B, 0 si no participa ---
    for hero_id, slug in (schema.HEROES if heroes is None else heroes):
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
