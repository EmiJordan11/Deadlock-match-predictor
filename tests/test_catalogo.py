"""El catalogo de heroes por corte: que el DAG siga funcionando cuando Valve agrega heroes."""
from collections import Counter

from deadlock import catalogo as cat
from deadlock import schema
from deadlock.transform import build_match_row, motivo_descarte, parse_match

VIEJOS = [i for i, _ in schema.HEROES]


def heroe_api(i, nombre, **extra):
    return {"id": i, "name": nombre, "player_selectable": True, "disabled": False, "in_development": False, **extra}


def partida(a, b, ganador="Team0", match_id=1):
    jugadores = [{"team": "Team0", "hero_id": h} for h in a] + [{"team": "Team1", "hero_id": h} for h in b]
    return {"match_id": match_id, "start_time": "2026-11-02 10:00:00", "game_mode": 1, "match_mode": 4,
            "average_badge": 50, "winning_team": ganador, "players": jugadores}


API = [heroe_api(i, s.replace("_", " ").title()) for i, s in schema.HEROES] + [
    heroe_api(78, "Deadman Danny"),
    heroe_api(86, "Violet", in_development=True),          # en desarrollo, pero se juega en ranked
    {"id": 99, "name": "Prototipo", "player_selectable": False, "disabled": False},   # no jugable: afuera
]


def test_slug():
    assert cat.slug("Mo & Krill") == "mo_and_krill"
    assert cat.slug("Deadman Danny") == "deadman_danny"
    assert cat.slug("The Doorman") == "the_doorman"
    assert cat.slug("McGinnis") == "mcginnis"


def test_el_registro_arranca_con_los_38_de_la_entrega_3():
    reg = cat.registro_inicial()
    assert [(int(i), v["slug"]) for i, v in reg.items()] == schema.HEROES


def test_catalogo_con_heroes_nuevos():
    reg = cat.registro_inicial()
    stats = [{"hero_id": i} for i in VIEJOS + [78]]                 # Violet todavia sin estadisticas
    en_partidas = Counter({1: 10, 78: 7, 86: 9, 90: 3})             # el 90 aparece antes que en la API
    c = cat.armar_catalogo(API, stats, en_partidas, reg, "2026-11-02")
    ids = [i for i, _ in cat.heroes_de(c)]
    assert ids == sorted(VIEJOS + [78, 86, 90])
    assert 99 not in ids                                             # no jugable y nunca jugado
    assert c["nuevos_en_registro"] == [78, 86, 90]
    assert c["sin_estadisticas"] == [86, 90]
    assert c["fuera_de_la_api"] == [90]
    assert dict(cat.heroes_de(c))[78] == "deadman_danny"
    assert dict(cat.heroes_de(c))[90] == "heroe_90"
    assert dict(cat.heroes_de(c))[18] == "mo_and_krill"             # los viejos conservan su slug


def test_el_slug_no_cambia_aunque_cambie_el_nombre():
    reg = cat.registro_inicial()
    cat.armar_catalogo(API, [], Counter({78: 1}), reg, "2026-11-02")
    renombrado = [h if h["id"] != 78 else heroe_api(78, "Danny the Dead") for h in API]
    c = cat.armar_catalogo(renombrado, [], Counter({78: 1}), reg, "2026-11-09")
    assert dict(cat.heroes_de(c))[78] == "deadman_danny"
    assert reg["78"]["nombre"] == "Danny the Dead"
    assert c["nuevos_en_registro"] == []


def test_un_heroe_nuevo_ya_no_descarta_la_partida():
    raw = partida([1, 2, 3, 4, 6, 78], [7, 8, 10, 11, 12, 13])
    assert motivo_descarte(raw) == "heroe_fuera_de_catalogo"         # catalogo de la Entrega 3: igual que antes
    assert parse_match(raw) is None
    ids = set(VIEJOS) | {78}
    assert motivo_descarte(raw, ids) is None
    base = parse_match(raw, ids)
    assert base["heroes_a"] == [1, 2, 3, 4, 6, 78] and base["winner"] == "A"


def test_motivos_de_descarte():
    ids = set(VIEJOS)
    assert motivo_descarte(partida([1, 2, 3, 4, 6], [7, 8, 10, 11, 12, 13]), ids) == "equipo_incompleto"
    assert motivo_descarte(partida([1, 2, 3, 4, 6, 7], [7, 8, 10, 11, 12, 13]), ids) == "heroe_repetido"
    assert motivo_descarte(partida([1, 2, 3, 4, 6, 14], [7, 8, 10, 11, 12, 13], ganador=None), ids) == "sin_ganador"


def test_las_features_tienen_una_columna_por_heroe_del_catalogo():
    reg = cat.registro_inicial()
    c = cat.armar_catalogo(API, [], Counter({78: 1, 86: 1}), reg, "2026-11-02")
    heroes = cat.heroes_de(c)
    ids = {i for i, _ in heroes}
    base = parse_match(partida([1, 2, 3, 4, 6, 78], [7, 8, 10, 11, 12, 86]), ids)
    fila = build_match_row(base, {"hero": {}, "counter": {}, "synergy": {}}, heroes)
    columnas = schema.columnas(heroes)
    assert set(columnas) <= set(fila)
    assert len(columnas) == len(schema.COLUMNS) + 2
    assert fila["hero_deadman_danny"] == 1 and fila["hero_violet"] == -1 and fila["hero_kelvin"] == -1
    assert sum(fila[k] for k in schema.composicion(heroes)) == 0      # seis +1 y seis -1
    assert fila["hero_winrate_adv_mean"] == 0.0                       # sin estadisticas: neutro


def test_el_esquema_de_la_entrega_3_no_cambia():
    assert len(schema.COLUMNS) == 73 and len(schema.FEATURE_COLUMNS) == 66
    assert schema.columnas() == schema.COLUMNS
    assert schema.composicion(schema.HEROES) == schema.COMPOSICION
