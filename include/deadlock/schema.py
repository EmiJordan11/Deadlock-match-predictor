"""
Esquema canonico del dataset de partidas de Deadlock - Ciencia de Datos,
UTN FRM 2026.

Documentado en detalle en docs/decisiones-diseno-dataset.md y
docs/estructura-dataset-final.md. Resumen de las decisiones que este modulo
materializa:

  * Composicion: una columna +-1/0 por cada heroe activo del juego (no
    columnas posicionales a_hero_1..6) para que el modelo trate a cada
    equipo como conjunto, no como tupla ordenada (Problema 1).
  * Counters/sinergia/winrate individual: agregados (media, minimo, maximo,
    desvio) sobre las ventajas ya normalizadas y ajustadas por shrinkage,
    no columnas por pareja posicional (Problemas 1, 2 y 4).
  * average_badge_diff es metadato, NUNCA feature de entrenamiento: un
    desequilibrio de matchmaking predice el ganador sin relacion con la
    composicion, y promoverlo a X contaminaria la pregunta del proyecto.

Unidad de observacion: una partida (ranked, modo normal). Clave: match_id.
"""

# --- catalogo de la Entrega 3: los 38 heroes de agosto/septiembre 2026 -------
# (id, slug). Es el catalogo CONGELADO con el que se construyeron los cortes
# del 20/08 al 20/09, el notebook entrega3_modelado_v3 y la aplicacion de la
# Entrega 4: no se toca, para que esos resultados se puedan reproducir.
#
# El DAG ya no depende de esta lista: arma el catalogo de cada corte con
# deadlock.catalogo (heroes de la API de assets + los que aparecen en las
# estadisticas y en las partidas), asi que sigue funcionando cuando Valve
# agrega heroes. Estos 38 slugs son la semilla del registro de heroes: un
# heroe conserva su nombre de columna para siempre.
HEROES = [
    (1, "infernus"), (2, "seven"), (3, "vindicta"), (4, "lady_geist"),
    (6, "abrams"), (7, "wraith"), (8, "mcginnis"), (10, "paradox"),
    (11, "dynamo"), (12, "kelvin"), (13, "haze"), (14, "holliday"),
    (15, "bebop"), (16, "calico"), (17, "grey_talon"), (18, "mo_and_krill"),
    (19, "shiv"), (20, "ivy"), (25, "warden"), (27, "yamato"),
    (31, "lash"), (35, "viscous"), (50, "pocket"), (52, "mirage"),
    (58, "vyper"), (60, "sinclair"), (63, "mina"), (64, "drifter"),
    (65, "venator"), (66, "victor"), (67, "paige"), (69, "the_doorman"),
    (72, "billy"), (76, "graves"), (77, "apollo"), (79, "rem"),
    (80, "silver"), (81, "celeste"),
]
HERO_IDS = {hero_id for hero_id, _ in HEROES}
HERO_SLUG_BY_ID = dict(HEROES)

# --- bloques del esquema, en el orden en que salen al CSV -------------------

METADATA = [
    "match_id", "match_date", "game_mode", "match_mode",
    "average_badge", "average_badge_diff",
]

COMPOSICION = [f"hero_{slug}" for _, slug in HEROES]                  # 38

_AGREGADOS = ["mean", "min", "max", "std"]

COUNTER_METRICS = ["winrate", "kills", "deaths", "assists", "souls"]
COUNTER = [f"counter_{metric}_adv_{agg}"
           for metric in COUNTER_METRICS for agg in _AGREGADOS]        # 20

SYNERGY = [f"synergy_winrate_adv_{agg}" for agg in _AGREGADOS]          # 4

HERO_WINRATE = [f"hero_winrate_adv_{agg}" for agg in _AGREGADOS]        # 4

TARGET = ["winner"]

COLUMNS = METADATA + COMPOSICION + COUNTER + SYNERGY + HERO_WINRATE + TARGET   # 6+38+20+4+4+1 = 73

# Columnas que SI entran a la matriz de entrenamiento (Unidad 3).
# match_id/match_date son identificadores; game_mode/match_mode quedan
# constantes por el filtro de descarga (Problema 6); average_badge y
# average_badge_diff son metadatos de trazabilidad/filtrado, nunca features.
FEATURE_COLUMNS = COMPOSICION + COUNTER + SYNERGY + HERO_WINRATE        # 66

# Columnas sin las cuales la fila no sirve (ver validate_bronze/quality_gate).
OBLIGATORIAS = ["match_id", "match_date", "winner"] + COMPOSICION

ENTEROS = ["match_id"] + COMPOSICION
TEXTO = ["game_mode", "match_mode", "winner"]
FECHA = ["match_date"]
FLOTANTES = ["average_badge", "average_badge_diff"] + COUNTER + SYNERGY + HERO_WINRATE


# --- el mismo esquema para cualquier catalogo -----------------------------------
# Lo usa el DAG con el catalogo de cada corte. Con el catalogo de la Entrega 3
# devuelve exactamente las constantes de arriba. Solo cambia el bloque de
# composicion: las 28 ventajas son agregados y no dependen de cuantos heroes hay.

def composicion(heroes=HEROES) -> list[str]:
    """Una columna +1/-1/0 por heroe del catalogo, en el orden del catalogo (por id)."""
    return [f"hero_{slug}" for _, slug in heroes]


def columnas(heroes=HEROES) -> list[str]:
    return METADATA + composicion(heroes) + COUNTER + SYNERGY + HERO_WINRATE + TARGET


def feature_columns(heroes=HEROES) -> list[str]:
    return composicion(heroes) + COUNTER + SYNERGY + HERO_WINRATE


def obligatorias(heroes=HEROES) -> list[str]:
    return ["match_id", "match_date", "winner"] + composicion(heroes)


assert columnas() == COLUMNS and feature_columns() == FEATURE_COLUMNS and obligatorias() == OBLIGATORIAS
