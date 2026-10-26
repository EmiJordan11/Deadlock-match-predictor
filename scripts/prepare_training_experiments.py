"""Descarga inmutable por ventana y construcción de plata para los experimentos."""
from __future__ import annotations

import gzip
import json
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'include'))
from deadlock import client, schema
from deadlock.experiment_data import (TRAIN_CUTS, TEST_CUT, STRESS_CUT, WINDOWS,
                                     build_dataset, sha256, window_params, write_json)
from deadlock.transform import parse_match

OUTPUT = ROOT/'include/output'
AUDIT = ROOT/'experiments/training_strategy'


def fetch_stats(cut, days, name):
    folder = OUTPUT/'bronze'/f'corte={cut}'/'stats'/f'dias={days}'
    path, meta = folder/f'{name}_stats.json', folder/f'{name}_stats.request.json'
    params = window_params(cut, days)
    endpoint = {'hero':'hero-stats','counter':'hero-counter-stats','synergy':'hero-synergy-stats'}[name]
    if name == 'counter': params['same_lane_filter'] = 'false'
    request = dict(endpoint=f'/v1/analytics/{endpoint}', params=params)
    if path.exists() and meta.exists():
        old = json.loads(meta.read_text(encoding='utf-8'))
        if old['request'] == request and old['sha256'] == sha256(path):
            data = json.loads(path.read_text(encoding='utf-8'))
            if data: return data
    data = client._get(request['endpoint'], params)
    if not data: raise ValueError(f'API sin estadísticas: {cut}/{days}/{name}')
    write_json(path, data)
    write_json(meta, dict(request=request, sha256=sha256(path),
                          retrieved_at=datetime.now(timezone.utc).isoformat(), rows=len(data)))
    print(f'DESCARGA {cut} {days}d {name}: {len(data)}', flush=True)
    time.sleep(.35)
    return data


def load_matches(cut):
    folder = OUTPUT/'bronze'/f'corte={cut}'/'matches'
    pages = sorted(folder.glob('pagina_*.json.gz'))
    if not pages:
        folder.mkdir(parents=True, exist_ok=True)
        t = datetime.fromisoformat(cut).replace(tzinfo=timezone.utc)
        for i, batch in enumerate(client.iter_matches(int(t.timestamp()),
                    int((t+timedelta(days=3)).timestamp())-1, 10000)):
            path = folder/f'pagina_{i:05d}.json.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as f: json.dump(batch, f)
        pages = sorted(folder.glob('pagina_*.json.gz'))
    bases, seen, raw_count, old_valid, hero_counts = [], set(), 0, 0, Counter()
    for path in pages:
        with gzip.open(path, 'rt', encoding='utf-8') as f: batch = json.load(f)
        for raw in batch:
            raw_count += 1
            assert raw.get('match_mode','').lower() == 'ranked'
            assert raw.get('game_mode','').lower() == 'normal'
            base = parse_match(raw)
            if base is None: continue
            assert cut <= base['match_date'] < (datetime.fromisoformat(cut)+timedelta(days=3)).date().isoformat()
            assert len(set(base['heroes_a']+base['heroes_b'])) == 12
            assert base['match_id'] not in seen
            seen.add(base['match_id'])
            hero_counts.update(base['heroes_a']+base['heroes_b'])
            old_valid += all(h not in (78,84,85,86,87,88) for h in base['heroes_a']+base['heroes_b'])
            bases.append(base)
    if len(bases)<1000: raise ValueError(f'{cut}: solo {len(bases)} filas válidas')
    info = dict(corte=cut, raw_rows=raw_count, valid_rows=len(bases), discarded=raw_count-len(bases),
                valid_with_original_38=old_valid, hero_counts=dict(hero_counts),
                first_date=min(b['match_date'] for b in bases), last_date=max(b['match_date'] for b in bases),
                first_timestamp=min(m['start_time'] for p in pages for m in read_page(p)),
                last_timestamp=max(m['start_time'] for p in pages for m in read_page(p)),
                sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for p in pages])
    return bases, info


def read_page(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f: return json.load(f)


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    inventory, all_ids = [], set()
    # El corte más antiguo y su ventana más ancha primero por retención.
    for cut in TRAIN_CUTS+[TEST_CUT, STRESS_CUT]:
        stats = {}
        for days in [20,14,7]:
            stats[days] = [fetch_stats(cut, days, name) for name in ('hero','counter','synergy')]
        bases, info = load_matches(cut)
        assert not all_ids.intersection(b['match_id'] for b in bases)
        all_ids.update(b['match_id'] for b in bases)
        for days in WINDOWS:
            folder = OUTPUT/'silver'/f'corte={cut}'/f'dias={days}'/'k=100'
            params = dict(fecha_corte=cut,dias_ventana_stats=days,dias_ventana_matches=3,
                          k_shrinkage=100,game_mode='normal',match_mode='ranked',
                          min_matches_stats=1,objetivo_filas=10000,hero_ids=sorted(schema.HERO_IDS))
            folder.mkdir(parents=True, exist_ok=True)
            df = build_dataset(bases, *stats[days])
            df.to_parquet(folder/'features.parquet', index=False)
            params['features_sha256'] = sha256(folder/'features.parquet')
            params['stats_sources'] = [dict(path=str((OUTPUT/'bronze'/f'corte={cut}'/'stats'/f'dias={days}'/f'{name}_stats.json').relative_to(ROOT)),
                sha256=sha256(OUTPUT/'bronze'/f'corte={cut}'/'stats'/f'dias={days}'/f'{name}_stats.json')) for name in ('hero','counter','synergy')]
            write_json(folder/'parametros.json', params)
            print(f'PLATA {cut} {days}d: {df.shape}', flush=True)
        inventory.append(info)
        write_json(AUDIT/'data_audit.json', inventory)
    print('DATOS COMPLETOS', flush=True)


if __name__ == '__main__': main()
