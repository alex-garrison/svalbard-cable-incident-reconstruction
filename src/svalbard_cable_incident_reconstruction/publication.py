"""The publication export contains one complete model run and its metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from svalbard_cable_incident_reconstruction.database import (
    DEFAULT_DB_PATH,
    connect,
    current_model_run,
)


def _nested_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in metrics.items():
        target = result
        *parents, leaf = name.split(".")
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = value
    return result


@dataclass(frozen=True)
class _PublishedRun:
    effective_config: dict[str, Any]
    ais_acquisition_id: str
    ais_content_sha256: str
    ais_source_manifest_sha256: str
    cable_sha256: str
    bathymetry_sha256: str


def _published_run(row: tuple[Any, ...] | None) -> _PublishedRun:
    if row is None:
        raise RuntimeError("The current complete model run is missing")

    (
        effective_config_json,
        ais_acquisition_id,
        ais_content_sha256,
        ais_source_manifest_sha256,
        cable_sha256,
        bathymetry_sha256,
    ) = row
    return _PublishedRun(
        effective_config=json.loads(str(effective_config_json)),
        ais_acquisition_id=str(ais_acquisition_id),
        ais_content_sha256=str(ais_content_sha256),
        ais_source_manifest_sha256=str(ais_source_manifest_sha256),
        cable_sha256=str(cable_sha256),
        bathymetry_sha256=str(bathymetry_sha256),
    )


def _crossing_components_by_scenario(
    crossing_rows: list[tuple[Any, ...]],
) -> dict[str, list[str]]:
    components_by_scenario: dict[str, list[str]] = {}

    for scenario_id, component in crossing_rows:
        components = components_by_scenario.setdefault(str(scenario_id), [])
        component_name = str(component)
        if component_name not in components:
            components.append(component_name)

    return components_by_scenario


def _reference_case(
    ratios: list[float],
    spreads: list[float],
    accepted_cases: list[tuple[float, float]],
) -> tuple[float, float]:
    if not accepted_cases:
        raise RuntimeError("Model result grid has no accepted cases")

    target_ratio = (ratios[0] + ratios[-1]) / 2.0
    target_spread = (spreads[0] + spreads[-1]) / 2.0
    ratio_range = ratios[-1] - ratios[0] or 1.0
    spread_range = spreads[-1] - spreads[0] or 1.0

    def distance_from_grid_centre(case: tuple[float, float]) -> float:
        ratio, spread = case
        ratio_distance = abs(ratio - target_ratio) / ratio_range
        spread_distance = abs(spread - target_spread) / spread_range
        return ratio_distance + spread_distance

    return min(accepted_cases, key=distance_from_grid_centre)


def _result_matrix(
    scenario_rows: list[tuple[Any, ...]], crossing_rows: list[tuple[Any, ...]]
) -> dict[str, Any]:
    components_by_scenario = _crossing_components_by_scenario(crossing_rows)
    ratio_values: set[float] = set()
    spread_values: set[float] = set()
    scenarios: dict[tuple[float, float], dict[str, Any]] = {}
    accepted_cases: list[tuple[float, float]] = []

    for scenario_id, ratio, spread, accepted, rejection_reason in scenario_rows:
        ratio_value = float(ratio)
        spread_value = float(spread)
        ratio_values.add(ratio_value)
        spread_values.add(spread_value)

        scenario_key = (ratio_value, spread_value)
        accepted_value = bool(accepted)
        scenarios[scenario_key] = {
            "accepted": accepted_value,
            "components": components_by_scenario.get(str(scenario_id), []),
            "rejection_reason": rejection_reason,
        }
        if accepted_value:
            accepted_cases.append(scenario_key)

    ratios = sorted(ratio_values)
    spreads = sorted(spread_values)

    expected_cases = len(ratios) * len(spreads)
    if len(scenarios) != expected_cases:
        raise RuntimeError(
            "Model result grid is incomplete: "
            f"expected {expected_cases} cases, found {len(scenarios)}"
        )
    reference_ratio, reference_spread = _reference_case(
        ratios,
        spreads,
        accepted_cases,
    )

    matrix_rows = []
    for ratio in ratios:
        cells = [scenarios[(ratio, spread)] for spread in spreads]
        matrix_rows.append(
            {
                "warp_depth_ratio": ratio,
                "cells": cells,
            }
        )

    return {
        "door_spreads_m": spreads,
        "warp_depth_ratios": ratios,
        "reference_case": {
            "warp_depth_ratio": reference_ratio,
            "door_spread_m": reference_spread,
        },
        "rows": matrix_rows,
    }


def _publication_metrics(
    rows: list[tuple[Any, ...]],
    run_id: str,
    config_sha256: str,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Publication metrics are missing; run metrics first")

    metrics: dict[str, Any] = {}
    mismatched_metrics: list[str] = []

    for name, value, metric_run_id, metric_config_sha256 in rows:
        if metric_run_id != run_id or metric_config_sha256 != config_sha256:
            mismatched_metrics.append(str(name))

        metrics[str(name)] = json.loads(str(value))

    if mismatched_metrics:
        raise RuntimeError(
            "Publication metrics do not match the current model run: "
            + ", ".join(mismatched_metrics)
        )

    return metrics


def _failure_interval_crossings(
    rows: list[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    crossings = []

    for row in rows:
        scenario_id, component, time_s, x_m, y_m, in_failure_interval = row
        crossings.append(
            {
                "scenario_id": str(scenario_id),
                "component": str(component),
                "time_s": float(time_s),
                "x_m": float(x_m),
                "y_m": float(y_m),
                "within_failure_interval": bool(in_failure_interval),
            }
        )

    return crossings


def publication_data(db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """The export is read from DuckDB after its run and metrics are checked."""
    with connect(db_path, read_only=True) as connection:
        run_id, config_sha256, implementation_sha256 = current_model_run(connection)
        published_run = _published_run(
            connection.execute(
                """
            SELECT effective_config_json, ais_acquisition_id,
                   ais_content_sha256, ais_source_manifest_sha256,
                   cable_sha256, bathymetry_sha256
            FROM analysis.model_runs
            WHERE run_id=?
            """,
                [run_id],
            ).fetchone()
        )

        metric_rows = connection.execute(
            """
            SELECT metric_name, metric_value_json, run_id, config_sha256
            FROM analysis.publication_metrics
            ORDER BY metric_name
            """
        ).fetchall()
        scenario_rows = connection.execute(
            """
            SELECT scenario_id, warp_depth_ratio, door_spread_m,
                   accepted, rejection_reason
            FROM analysis.gear_scenarios
            WHERE run_id=?
            ORDER BY warp_depth_ratio, door_spread_m
            """,
            [run_id],
        ).fetchall()
        crossing_rows = connection.execute(
            """
            SELECT scenario_id, component
            FROM analysis.gear_horizontal_crossings
            WHERE run_id=? AND within_failure_interval
            ORDER BY scenario_id, component, crossing_time_s
            """,
            [run_id],
        ).fetchall()

    metrics = _publication_metrics(metric_rows, run_id, config_sha256)

    return {
        "provenance": {
            "run_id": run_id,
            "config_sha256": config_sha256,
            "implementation_sha256": implementation_sha256,
            "ais_acquisition_id": published_run.ais_acquisition_id,
            "ais_content_sha256": published_run.ais_content_sha256,
            "ais_source_manifest_sha256": published_run.ais_source_manifest_sha256,
            "cable_sha256": published_run.cable_sha256,
            "bathymetry_sha256": published_run.bathymetry_sha256,
        },
        "config": published_run.effective_config,
        "metrics": _nested_metrics(metrics),
        "result_matrix": _result_matrix(scenario_rows, crossing_rows),
    }


def playback_data(db_path: Path, run_id: str, replay: dict[str, Any]) -> dict[str, Any]:
    results = publication_data(db_path)
    if results["provenance"]["run_id"] != run_id:
        raise RuntimeError("Playback model does not match the published results")

    with connect(db_path, read_only=True) as connection:
        crossing_rows = connection.execute(
            """
            SELECT scenario_id, component, crossing_time_s, crossing_x_m,
                   crossing_y_m, within_failure_interval
            FROM analysis.gear_horizontal_crossings
            WHERE run_id=? AND within_failure_interval
            ORDER BY scenario_id, crossing_time_s, component
            """,
            [run_id],
        ).fetchall()

    return {
        "provenance": {"run_id": run_id},
        "failure_interval_crossings": _failure_interval_crossings(crossing_rows),
        "replay": replay,
    }
