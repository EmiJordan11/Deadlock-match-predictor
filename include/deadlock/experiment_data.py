"""Datos auditables para el estudio temporal; comparte transformaciones con el DAG."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from deadlock import schema
from deadlock.transform import build_match_row, build_stat_tables

TRAIN_CUTS = ['2026-08-20', '2026-08-27', '2026-09-03', '2026-09-10']
TEST_CUT = '2026-09-20'
STRESS_CUT = '2026-10-19'
WINDOWS = [7, 14, 20]


def window_params(cut: str, days: int, min_matches: int = 1) -> dict:
    t = datetime.fromisoformat(cut).replace(tzinfo=timezone.utc)
    return dict(min_unix_timestamp=int((t-timedelta(days=days)).timestamp()),
                max_unix_timestamp=int(t.timestamp())-1, game_mode='normal',
                match_mode='ranked', min_matches=min_matches)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def vector_features(bases: list[dict], tables: dict) -> pd.DataFrame:
    """Mismos agregados que build_match_row, calculados en bloque sin muestrear filas."""
    ids = [i for i, _ in schema.HEROES]
    pos = {h: i for i, h in enumerate(ids)}
    a = np.array([[pos[h] for h in b['heroes_a']] for b in bases])
    b = np.array([[pos[h] for h in b['heroes_b']] for b in bases])
    out = pd.DataFrame([{k: v for k, v in row.items() if k not in ('heroes_a', 'heroes_b')}
                        for row in bases])
    for h, slug in schema.HEROES:
        out[f'hero_{slug}'] = ((a == pos[h]).any(axis=1).astype('int8')
                                  - (b == pos[h]).any(axis=1).astype('int8'))

    def aggs(values):
        return [values.mean(axis=1), values.min(axis=1), values.max(axis=1), values.std(axis=1)]

    for metric in schema.COUNTER_METRICS:
        matrix = np.array([[tables['counter'].get((x, y), {}).get(metric, 0.) for y in ids] for x in ids])
        vals = matrix[a[:, :, None], b[:, None, :]].reshape(len(bases), 36)
        for agg, values in zip(('mean','min','max','std'), aggs(vals)):
            out[f'counter_{metric}_adv_{agg}'] = values
    matrix = np.array([[tables['synergy'].get(frozenset((x, y)), 0.) for y in ids] for x in ids])
    p, q = np.triu_indices(6, 1)
    sa, sb = matrix[a[:,p], a[:,q]], matrix[b[:,p], b[:,q]]
    for agg, va, vb in zip(('mean','min','max','std'), aggs(sa), aggs(sb)):
        out[f'synergy_winrate_adv_{agg}'] = va-vb
    hero = np.array([tables['hero'].get(h, 0.) for h in ids])
    for agg, va, vb in zip(('mean','min','max','std'), aggs(hero[a]), aggs(hero[b])):
        out[f'hero_winrate_adv_{agg}'] = va-vb
    out = out[schema.COLUMNS]
    # Verificación contra la implementación canónica, incluidos extremos del lote.
    for i in np.unique(np.linspace(0, len(bases)-1, min(12, len(bases))).astype(int)):
        expected = build_match_row(bases[i], tables)
        np.testing.assert_allclose(out.loc[i, schema.FEATURE_COLUMNS].to_numpy(dtype=float),
                                   [expected[c] for c in schema.FEATURE_COLUMNS], atol=1e-10)
    assert out['match_id'].is_unique
    assert np.isfinite(out[schema.FEATURE_COLUMNS].to_numpy()).all()
    return out


def build_dataset(bases: list[dict], hero: list, counter: list, synergy: list) -> pd.DataFrame:
    return vector_features(bases, build_stat_tables(hero, counter, synergy, k=100))
