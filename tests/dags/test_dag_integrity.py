"""Chequeos minimos sobre el DAG: que parsee, que tenga las tareas esperadas.

Se lee el DAG ya parseado (`dagbag.dags`) y no con `get_dag()`: en Airflow 3,
`get_dag()` consulta la base de metadatos, que no existe en el contenedor
de `astro dev pytest`.
"""
from airflow.models import DagBag


def _dag():
    dagbag = DagBag(include_examples=False)
    assert "deadlock_ingest" in dagbag.dags, f"no se parseo el DAG: {dagbag.import_errors}"
    return dagbag.dags["deadlock_ingest"]


def test_no_import_errors():
    dagbag = DagBag(include_examples=False)
    assert not dagbag.import_errors, f"errores de import: {dagbag.import_errors}"


def test_dag_deadlock():
    ids = set(_dag().task_dict)
    esperadas = {
        "resolve_window", "extract_heroes", "extract_hero_stats", "extract_counter_stats",
        "extract_synergy_stats", "extract_matches", "validate_bronze", "build_catalogo",
        "build_stat_tables", "build_matches_table", "build_features",
        "quality_gate", "export_csv",
    }
    assert esperadas <= ids, f"faltan tareas: {esperadas - ids}"


def test_el_catalogo_alimenta_toda_la_plata():
    """Las features, las tablas y la base dependen del catalogo del corte, no de una lista fija."""
    dag = _dag()
    aguas_abajo = dag.task_dict["build_catalogo"].get_flat_relative_ids(upstream=False)
    assert {"build_stat_tables", "build_matches_table", "build_features", "quality_gate"} <= aguas_abajo
    assert "validate_bronze" in dag.task_dict["build_catalogo"].upstream_task_ids


def test_una_corrida_a_la_vez():
    """Las corridas comparten el registro de heroes y el limite de la API de partidas."""
    assert _dag().max_active_runs == 1


def test_deadlock_se_dispara_a_mano():
    """T es fijo por Param, asi que agendar no aporta: se dispara manualmente."""
    assert _dag().schedule is None


def test_solo_este_dag_registrado():
    dagbag = DagBag(include_examples=False)
    assert set(dagbag.dag_ids) == {"deadlock_ingest"}
