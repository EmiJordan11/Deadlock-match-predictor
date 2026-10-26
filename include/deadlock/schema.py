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

# --- catalogo de heroes, relevado en vivo de /v1/assets/heroes ---------------
# (id, slug). Si Valve agrega un heroe nuevo despues de esta corrida, hay que
# regenerar esta lista antes de correr el pipeline: HEROES no se arma
# dinamicamente para que el esquema del CSV sea estable entre corridas.
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
    (78, "deadman_danny"), (84, "rat_king"), (85, "solomon"),
    (86, "violet"), (87, "nurse_harrow"), (88, "baba"),
]
HERO_IDS = {hero_id for hero_id, _ in HEROES}
HERO_SLUG_BY_ID = dict(HEROES)

# --- bloques del esquema, en el orden en que salen al CSV -------------------

METADATA = [
    "match_id", "match_date", "game_mode", "match_mode",
    "average_badge", "average_badge_diff",
]

COMPOSICION = [f"hero_{slug}" for _, slug in HEROES]                  # 44 al 26/10/2026

_AGREGADOS = ["mean", "min", "max", "std"]

COUNTER_METRICS = ["winrate", "kills", "deaths", "assists", "souls"]
COUNTER = [f"counter_{metric}_adv_{agg}"
           for metric in COUNTER_METRICS for agg in _AGREGADOS]        # 20

SYNERGY = [f"synergy_winrate_adv_{agg}" for agg in _AGREGADOS]          # 4

HERO_WINRATE = [f"hero_winrate_adv_{agg}" for agg in _AGREGADOS]        # 4

TARGET = ["winner"]

COLUMNS = METADATA + COMPOSICION + COUNTER + SYNERGY + HERO_WINRATE + TARGET   # 79

# Columnas que SI entran a la matriz de entrenamiento (Unidad 3).
# match_id/match_date son identificadores; game_mode/match_mode quedan
# constantes por el filtro de descarga (Problema 6); average_badge y
# average_badge_diff son metadatos de trazabilidad/filtrado, nunca features.
FEATURE_COLUMNS = COMPOSICION + COUNTER + SYNERGY + HERO_WINRATE        # 72

# Columnas sin las cuales la fila no sirve (ver validate_bronze/quality_gate).
OBLIGATORIAS = ["match_id", "match_date", "winner"] + COMPOSICION

ENTEROS = ["match_id"] + COMPOSICION
TEXTO = ["game_mode", "match_mode", "winner"]
FECHA = ["match_date"]
FLOTANTES = ["average_badge", "average_badge_diff"] + COUNTER + SYNERGY + HERO_WINRATE
