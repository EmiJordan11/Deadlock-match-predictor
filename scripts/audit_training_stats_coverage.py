"""Verifica retención y cobertura diaria del extremo más antiguo y del estrés."""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'include'))
from deadlock import client
from deadlock.experiment_data import window_params, write_json


def main():
    report=[]
    for cut in ('2026-08-20','2026-10-19'):
        params=dict(window_params(cut,20),bucket='start_time_day')
        data=client._get('/v1/analytics/hero-stats',params)
        days=sorted({datetime.fromtimestamp(r['bucket'],tz=timezone.utc).date().isoformat()
                     for r in data if r['matches']>0})
        assert len(days)==20,(cut,days)
        report.append(dict(cut=cut,request=params,distinct_days=len(days),first_day=days[0],last_day=days[-1]))
        print(report[-1],flush=True)
    write_json(ROOT/'experiments/training_strategy/stats_coverage.json',report)


if __name__=='__main__': main()
