"""
Catalogo de heroes de cada corte: que heroes existen y que columna tiene cada uno.

Por que existe. Valve agrega heroes cada pocas semanas. Con el catalogo fijo
de 38, el filtro de calidad descartaba toda partida con un heroe nuevo: el
19/10 eran 9.940 de 10.000. El DAG no fallaba, pero dejaba un dataset casi
vacio y sesgado (solo las partidas sin heroes nuevos).

Ahora el catalogo de un corte junta tres fuentes:

  1. /v1/assets/heroes: los heroes jugables (player_selectable y no
     disabled). NO se filtra por in_development: Violet y Baba tienen esa
     marca y estaban en el 83 % y el 25 % de las partidas ranked del 19/10.
  2. Las estadisticas de la ventana (hero-stats): los heroes que se jugaron
     en los dias anteriores al corte.
  3. Las partidas del corte: un heroe recien lanzado puede aparecer en las
     partidas antes que en la API de assets o en las estadisticas.

Un heroe que existe pero no se jugo en un corte tiene su columna en cero en
todas las filas, que es justo lo que significa la codificacion +1/-1/0. Por
eso juntar cortes con catalogos distintos es completar las columnas que
faltan con 0.

El registro de heroes fija el nombre de columna de cada id la primera vez que
aparece y no lo cambia nunca, aunque el heroe cambie de nombre en el juego:
un modelo identifica a cada heroe por su columna. Los 38 de la Entrega 3
entran con los mismos slugs de siempre (schema.HEROES).
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from deadlock import schema


def slug(nombre: str) -> str:
    """'Mo & Krill' -> 'mo_and_krill', 'Deadman Danny' -> 'deadman_danny'."""
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s.lower().replace("&", " and ")).strip("_")
    return s or "heroe"


def jugables(heroes_api: list[dict]) -> dict[int, str]:
    """id -> nombre de los heroes que la API de assets da por jugables."""
    return {h["id"]: h.get("name") or f"Heroe {h['id']}" for h in heroes_api
            if h.get("player_selectable") and not h.get("disabled")}


def apariciones(partidas: Iterable[dict]) -> Counter:
    """En cuantas partidas aparece cada hero_id (un heroe aparece una vez por partida)."""
    c = Counter()
    for raw in partidas:
        ids = {j.get("hero_id") for j in raw.get("players") or []}
        c.update(i for i in ids if isinstance(i, int) and i > 0)
    return c


def registro_inicial(nombres: dict[int, str] | None = None) -> dict:
    """El registro arranca con los 38 heroes de la Entrega 3 y sus slugs de siempre."""
    nombres = nombres or {}
    return {str(i): {"slug": s, "nombre": nombres.get(i, s.replace("_", " ").title()), "desde": "entrega 3"}
            for i, s in schema.HEROES}


def cargar_registro(ruta: Path) -> dict:
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return registro_inicial()


def guardar_registro(ruta: Path, registro: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    orden = dict(sorted(registro.items(), key=lambda kv: int(kv[0])))
    ruta.write_text(json.dumps(orden, indent=2, ensure_ascii=False), encoding="utf-8")


def actualizar_registro(registro: dict, ids: Iterable[int], nombres: dict[int, str], corte: str) -> list[int]:
    """Agrega al registro los ids que no tiene y devuelve cuales eran nuevos.

    El slug sale del nombre de la API la primera vez que se ve el heroe. Si
    un heroe aparece en las partidas antes que en la API, entra como
    'heroe_<id>' y ese slug queda fijo; el nombre se actualiza cuando la API
    lo publique.
    """
    usados = {v["slug"] for v in registro.values()}
    nuevos = []
    for i in sorted(ids):
        clave = str(i)
        if clave in registro:
            if i in nombres:                      # el nombre se actualiza; el slug, nunca
                registro[clave]["nombre"] = nombres[i]
            continue
        s = slug(nombres[i]) if i in nombres else f"heroe_{i}"
        if s in usados:
            s = f"{s}_{i}"
        registro[clave] = {"slug": s, "nombre": nombres.get(i, f"Heroe {i}"), "desde": corte}
        usados.add(s)
        nuevos.append(i)
    return nuevos


def armar_catalogo(heroes_api: list[dict], hero_stats: list[dict], en_partidas: Counter,
                   registro: dict, corte: str) -> dict:
    """El catalogo del corte, listo para guardar como JSON y pasar entre tareas.

    `heroes` es la lista [(id, slug)] ordenada por id que usan schema.columnas()
    y transform.build_match_row(). El resto es lo que hay que mirar en el log:
    que heroes son nuevos, cuales se jugaron sin tener estadisticas todavia
    (su ventaja de winrate vale 0, neutra) y cuales no estan en la API.
    """
    nombres = jugables(heroes_api)
    ids_stats = {r["hero_id"] for r in hero_stats}
    ids = set(nombres) | ids_stats | set(en_partidas)
    nuevos = actualizar_registro(registro, ids, nombres, corte)
    heroes = [(i, registro[str(i)]["slug"]) for i in sorted(ids)]
    return {
        "corte": corte,
        "heroes": heroes,
        "fuera_del_catalogo_entrega3": [i for i in sorted(ids) if i not in schema.HERO_IDS],
        "nuevos_en_registro": nuevos,
        "sin_estadisticas": sorted(i for i in en_partidas if i not in ids_stats),
        "fuera_de_la_api": sorted(i for i in ids if i not in nombres),
        "detalle": [{"id": i, "slug": registro[str(i)]["slug"], "nombre": registro[str(i)]["nombre"],
                     "en_api": i in nombres, "en_estadisticas": i in ids_stats, "partidas": en_partidas.get(i, 0)}
                    for i in sorted(ids)],
    }


def heroes_de(catalogo: dict) -> list[tuple[int, str]]:
    """[(id, slug)] de un catalogo leido de JSON (las tuplas vuelven como listas)."""
    return [(int(i), s) for i, s in catalogo["heroes"]]
