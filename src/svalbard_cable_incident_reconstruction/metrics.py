"""Publication metrics combine model results with independent source checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from shapely import Point, from_wkb
from shapely.ops import transform

from svalbard_cable_incident_reconstruction.config import (
    CableVigilanceConfig,
    MelkartConfig,
)
from svalbard_cable_incident_reconstruction.database import (
    connect,
    current_model_run,
    ensure_spatial,
    implementation_sha256,
)
from svalbard_cable_incident_reconstruction.ers import calculate_ers_metrics
from svalbard_cable_incident_reconstruction.gfw import calculate_gfw_metrics
from svalbard_cable_incident_reconstruction.slope import calculate_slope_metrics
from svalbard_cable_incident_reconstruction.yagry import calculate_yagry_metrics


def _required_row[T](row: T | None) -> T:
    if row is None:
        raise RuntimeError("The current model run is incomplete")
    return row


def _validate_run(run: tuple[Any, ...], config: MelkartConfig) -> None:
    if run[1] != config.sha256():
        raise RuntimeError(
            "Current model run uses a different scientific configuration; "
            "run model before metrics"
        )
    if run[2] != implementation_sha256():
        raise RuntimeError("Different package code produced the current model run")


def _cable_vigilance_metrics(
    frame: pd.DataFrame, settings: CableVigilanceConfig
) -> dict[str, float | int]:
    if frame.empty:
        raise RuntimeError("Cable Vigilance AIS is missing; run ingest first")

    project = Transformer.from_crs(4326, 25833, always_xy=True)
    unproject = Transformer.from_crs(25833, 4326, always_xy=True)

    frame = frame.copy()
    frame["x"], frame["y"] = project.transform(
        frame.longitude.to_numpy(), frame.latitude.to_numpy()
    )
    frame["grid_x"] = np.rint(frame.x / settings.density_grid_m).astype(int)
    frame["grid_y"] = np.rint(frame.y / settings.density_grid_m).astype(int)
    cells = (
        frame.groupby(["grid_x", "grid_y"])
        .size()
        .to_frame("position_count")
        .reset_index()
        .sort_values(
            ["position_count", "grid_x", "grid_y"],
            ascending=[False, True, True],
        )
        .head(2)
        .sort_values("grid_x")
    )
    if len(cells) < 2:
        raise RuntimeError("Cable Vigilance AIS has fewer than two occupied grid cells")

    centroids: dict[str, tuple[float, float, int]] = {}
    rows = cells.itertuples(index=False, name=None)
    for name, (grid_x, grid_y, position_count) in zip(
        ("west", "east"), rows, strict=True
    ):
        longitude, latitude = unproject.transform(
            grid_x * settings.density_grid_m,
            grid_y * settings.density_grid_m,
        )
        centroids[name] = (longitude, latitude, int(position_count))

    geod = Geod(ellps="WGS84")
    _, _, separation = geod.inv(*centroids["west"][:2], *centroids["east"][:2])
    return {
        "validation.cable_vigilance.west.longitude": centroids["west"][0],
        "validation.cable_vigilance.west.latitude": centroids["west"][1],
        "validation.cable_vigilance.west.ais_row_count": centroids["west"][2],
        "validation.cable_vigilance.east.longitude": centroids["east"][0],
        "validation.cable_vigilance.east.latitude": centroids["east"][1],
        "validation.cable_vigilance.east.ais_row_count": centroids["east"][2],
        "validation.cable_vigilance.dwell_separation_m": float(separation),
    }


def _ais_screen_metrics(
    connection: duckdb.DuckDBPyConnection,
    run_id: str,
) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT mmsi, cable_id, minimum_vessel_cable_distance_m,
               conservative_gear_reach_m, within_conservative_gear_reach
        FROM analysis.ais_candidate_screen
        WHERE run_id=? ORDER BY mmsi, cable_id
        """,
        [run_id],
    ).fetchall()
    observed_mmsis = set()
    retained_mmsis = set()

    for mmsi, _, _, _, retained in rows:
        observed_mmsis.add(mmsi)
        if retained:
            retained_mmsis.add(mmsi)

    metrics: dict[str, Any] = {
        "ais.observed_mmsi_count": len(observed_mmsis),
        "ais.retained_mmsi_count": len(retained_mmsis),
    }

    for mmsi, cable, distance, reach, retained in rows:
        prefix = f"ais.mmsi_{mmsi}.{cable}"
        metrics[f"{prefix}.minimum_vessel_cable_distance_m"] = distance
        metrics[f"{prefix}.conservative_gear_reach_m"] = reach
        metrics[f"{prefix}.within_conservative_gear_reach"] = retained
    return metrics


def _spatial_validation_metrics(
    efs_geometry_wkb: bytes,
    vigilance_metrics: dict[str, float | int],
    candidate_longitude: float | None,
    candidate_latitude: float | None,
) -> dict[str, Any]:
    project = Transformer.from_crs(4326, 25833, always_xy=True)
    efs_track = transform(project.transform, from_wkb(efs_geometry_wkb))

    metrics: dict[str, Any] = {
        "validation.efs_71611.track_length_m": efs_track.length,
        "validation.efs_71611.candidate_distance_m": None,
        "validation.cable_vigilance.closest_low_speed_cluster_to_incident": None,
        "validation.cable_vigilance.closest_low_speed_cluster_to_incident_m": None,
        "validation.svendsen_age_determination_sediment_1996.core_to_incident_centroid_m": None,
    }

    if candidate_longitude is None or candidate_latitude is None:
        return metrics

    candidate_x, candidate_y = project.transform(
        candidate_longitude,
        candidate_latitude,
    )
    candidate = Point(candidate_x, candidate_y)
    metrics["validation.efs_71611.candidate_distance_m"] = candidate.distance(efs_track)

    geod = Geod(ellps="WGS84")
    vigilance_distances: dict[str, float] = {}

    for location in ("west", "east"):
        location_longitude = float(
            vigilance_metrics[f"validation.cable_vigilance.{location}.longitude"]
        )
        location_latitude = float(
            vigilance_metrics[f"validation.cable_vigilance.{location}.latitude"]
        )
        _, _, distance = geod.inv(
            candidate_longitude,
            candidate_latitude,
            location_longitude,
            location_latitude,
        )
        vigilance_distances[location] = distance

    closest_location = "west"
    if vigilance_distances["east"] < vigilance_distances["west"]:
        closest_location = "east"

    closest_distance = vigilance_distances[closest_location]

    metrics["validation.cable_vigilance.closest_low_speed_cluster_to_incident"] = (
        closest_location
    )
    metrics["validation.cable_vigilance.closest_low_speed_cluster_to_incident_m"] = (
        closest_distance
    )

    sediment_core_longitude = 9.942667
    sediment_core_latitude = 78.188333
    _, _, sediment_core_distance = geod.inv(
        candidate_longitude,
        candidate_latitude,
        sediment_core_longitude,
        sediment_core_latitude,
    )
    metrics[
        "validation.svendsen_age_determination_sediment_1996.core_to_incident_centroid_m"
    ] = sediment_core_distance

    return metrics


def _model_geometry_metrics(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    geometry_row = connection.execute(
        """
        SELECT count(*), count_if(accepted), count_if(NOT accepted)
        FROM analysis.gear_scenarios
        """
    ).fetchone()
    geometry_count, accepted_geometry_count, rejected_geometry_count = _required_row(
        geometry_row
    )

    return {
        "model.geometry_count": int(geometry_count),
        "model.accepted_geometry_count": int(accepted_geometry_count),
        "model.rejected_geometry_count": int(rejected_geometry_count),
    }


def _ais_track_metrics(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    quality_row = connection.execute(
        """
        SELECT source_position_count, position_group_count,
               rejected_group_count, spline_group_count
        FROM analysis.ais_position_quality
        """
    ).fetchone()
    (
        source_position_count,
        position_group_count,
        rejected_group_count,
        spline_group_count,
    ) = _required_row(quality_row)

    return {
        "ais.track.source_position_count": int(source_position_count),
        "ais.track.position_group_count": int(position_group_count),
        "ais.track.rejected_group_count": int(rejected_group_count),
        "ais.track.spline_group_count": int(spline_group_count),
    }


def _crossing_metrics(
    connection: duckdb.DuckDBPyConnection,
    run_id: str,
) -> dict[str, Any]:
    count_row = connection.execute(
        """
        SELECT count(*), count(DISTINCT scenario_id),
               coalesce(count_if(component='port_door'), 0),
               coalesce(count_if(component='starboard_door'), 0),
               coalesce(count_if(component='centre_clump'), 0)
        FROM analysis.gear_horizontal_crossings
        WHERE run_id=? AND within_failure_interval
        """,
        [run_id],
    ).fetchone()
    (
        crossing_event_count,
        crossing_geometry_count,
        port_door_count,
        starboard_door_count,
        centre_clump_count,
    ) = _required_row(count_row)

    centroid_row = connection.execute(
        """
        SELECT avg(crossing_latitude), avg(crossing_longitude), count(*),
               count(DISTINCT scenario_id)
        FROM analysis.gear_horizontal_crossings
        WHERE run_id=? AND within_failure_interval
        """,
        [run_id],
    ).fetchone()
    (
        candidate_latitude_value,
        candidate_longitude_value,
        candidate_position_count,
        candidate_geometry_count,
    ) = _required_row(centroid_row)

    if candidate_position_count:
        candidate_latitude = float(candidate_latitude_value)
        candidate_longitude = float(candidate_longitude_value)
    else:
        candidate_latitude = None
        candidate_longitude = None

    return {
        "candidate.crossing_centroid.latitude": candidate_latitude,
        "candidate.crossing_centroid.longitude": candidate_longitude,
        "candidate.crossing_centroid.position_count": int(candidate_position_count),
        "candidate.crossing_centroid.geometry_count": int(candidate_geometry_count),
        "crossing.cable_a.event_count": int(crossing_event_count),
        "crossing.cable_a.geometry_count": int(crossing_geometry_count),
        "crossing.cable_a.port_door_count": int(port_door_count),
        "crossing.cable_a.starboard_door_count": int(starboard_door_count),
        "crossing.cable_a.centre_clump_count": int(centre_clump_count),
    }


def _validation_inputs(
    connection: duckdb.DuckDBPyConnection,
    maximum_dwell_speed_knots: float,
) -> tuple[bytes, pd.DataFrame]:
    efs_row = connection.execute(
        "SELECT ST_AsWKB(geom) FROM geo.efs_71611_repair_notice_track_raw"
    ).fetchone()
    (efs_geometry_wkb,) = _required_row(efs_row)

    vigilance = connection.execute(
        """
        SELECT timestamp_utc, longitude, latitude
        FROM clean.ais
        WHERE dataset_id='cable_vigilance'
          AND speed_over_ground_knots <= ?
        ORDER BY timestamp_utc
        """,
        [maximum_dwell_speed_knots],
    ).fetchdf()

    return bytes(efs_geometry_wkb), vigilance


def _calculate_model_metrics(db_path: Path, config: MelkartConfig) -> dict[str, Any]:
    with connect(db_path, read_only=True) as connection:
        ensure_spatial(connection)
        run_id, config_sha256, implementation_hash = current_model_run(connection)
        _validate_run((run_id, config_sha256, implementation_hash), config)

        metrics: dict[str, Any] = {}
        metrics.update(_model_geometry_metrics(connection))
        metrics.update(_ais_track_metrics(connection))
        metrics.update(_crossing_metrics(connection, run_id))
        efs_geometry_wkb, vigilance = _validation_inputs(
            connection,
            config.cable_vigilance.maximum_dwell_speed_knots,
        )

    candidate_longitude = metrics["candidate.crossing_centroid.longitude"]
    candidate_latitude = metrics["candidate.crossing_centroid.latitude"]
    vigilance_metrics = _cable_vigilance_metrics(vigilance, config.cable_vigilance)
    validation_metrics = _spatial_validation_metrics(
        efs_geometry_wkb,
        vigilance_metrics,
        candidate_longitude,
        candidate_latitude,
    )
    metrics.update(vigilance_metrics)
    metrics.update(validation_metrics)

    return metrics


def _yagry_delay_hours(
    first_intersection_utc: str | None,
    failure_start_utc: str,
) -> float | None:
    if first_intersection_utc is None:
        return None

    first_intersection = datetime.fromisoformat(
        first_intersection_utc.replace("Z", "+00:00")
    )
    failure_start = datetime.fromisoformat(failure_start_utc).replace(tzinfo=UTC)
    delay = first_intersection - failure_start
    return delay.total_seconds() / 3600.0


def _merge_metrics(metric_groups: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    for group in metric_groups:
        duplicate_names = metrics.keys() & group.keys()
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise RuntimeError(f"Duplicate publication metrics: {names}")

        metrics.update(group)

    return metrics


def calculate_metrics(db_path: Path, config: MelkartConfig) -> dict[str, Any]:
    model_metrics = _calculate_model_metrics(db_path, config)
    latitude = float(model_metrics["candidate.crossing_centroid.latitude"])
    longitude = float(model_metrics["candidate.crossing_centroid.longitude"])

    with connect(db_path, read_only=True) as connection:
        run_id, _, _ = current_model_run(connection)
        ensure_spatial(connection)

        yagry_metrics = calculate_yagry_metrics(connection)
        first_intersection = yagry_metrics["ais.yagry.cable_a.first_intersection_utc"]
        yagry_metrics["ais.yagry.cable_a.first_intersection_delay_hours"] = (
            _yagry_delay_hours(first_intersection, config.failure_interval.start_utc)
        )

        database_metrics = [
            model_metrics,
            _ais_screen_metrics(connection, run_id),
            calculate_ers_metrics(connection),
            yagry_metrics,
        ]

    evidence_metrics = [
        {
            "validation.cable_vigilance.mmsi": config.cable_vigilance.mmsi,
            "validation.cable_vigilance.selection_start_utc": config.cable_vigilance.start_utc,
            "validation.cable_vigilance.selection_end_utc": config.cable_vigilance.end_utc,
            "validation.cable_vigilance.maximum_dwell_speed_knots": config.cable_vigilance.maximum_dwell_speed_knots,
            "validation.cable_vigilance.density_grid_m": config.cable_vigilance.density_grid_m,
        },
        calculate_gfw_metrics(longitude, latitude),
        calculate_slope_metrics(longitude, latitude, config.bathymetry),
    ]

    return _merge_metrics([*database_metrics, *evidence_metrics])


def write_publication_metrics(
    db_path: Path, config: MelkartConfig, metrics: dict[str, Any]
) -> None:
    with connect(db_path, read_only=True) as connection:
        run = current_model_run(connection)
    _validate_run(run, config)

    generated_at = datetime.now(UTC)
    rows = [
        (
            run[0],
            run[1],
            name,
            json.dumps(value, sort_keys=True),
            generated_at,
        )
        for name, value in sorted(metrics.items())
    ]
    with connect(db_path) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                """
                CREATE OR REPLACE TABLE analysis.publication_metrics (
                    run_id VARCHAR NOT NULL,
                    config_sha256 VARCHAR NOT NULL,
                    metric_name VARCHAR PRIMARY KEY,
                    metric_value_json JSON NOT NULL,
                    generated_at_utc TIMESTAMPTZ NOT NULL
                )
                """
            )
            connection.executemany(
                "INSERT INTO analysis.publication_metrics VALUES (?, ?, ?, ?::JSON, ?)",
                rows,
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
