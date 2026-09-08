"""Chequeos minimos sobre el DAG: que parsee, que tenga las tareas esperadas."""
from airflow.models import DagBag


def test_no_import_errors():
    dagbag = DagBag(include_examples=False)
    assert not dagbag.import_errors, f"errores de import: {dagbag.import_errors}"


def test_dag_deadlock():
    dagbag = DagBag(include_examples=False)
    dag = dagbag.get_dag("deadlock_ingest")
    assert dag is not None
    ids = set(dag.task_dict)
    esperadas = {
        "resolve_window", "extract_hero_stats", "extract_counter_stats",
        "extract_synergy_stats", "extract_matches", "validate_bronze",
        "build_stat_tables", "build_matches_table", "build_features",
        "quality_gate", "export_csv",
    }
    assert esperadas <= ids, f"faltan tareas: {esperadas - ids}"


def test_deadlock_se_dispara_a_mano():
    """T es fijo por Param, asi que agendar no aporta: se dispara manualmente."""
    dagbag = DagBag(include_examples=False)
    assert dagbag.get_dag("deadlock_ingest").schedule is None


def test_solo_este_dag_registrado():
    dagbag = DagBag(include_examples=False)
    assert set(dagbag.dag_ids) == {"deadlock_ingest"}
