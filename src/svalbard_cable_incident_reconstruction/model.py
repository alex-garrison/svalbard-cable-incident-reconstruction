"""The model estimates gear paths from AIS and publishes one complete run."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from rasterio.io import DatasetReader
from shapely import from_wkb
from shapely.geometry import LineString
from shapely.ops import transform

from svalbard_cable_incident_reconstruction.config import MelkartConfig
from svalbard_cable_incident_reconstruction.database import (
    DEFAULT_DB_PATH,
    LOCAL_CRS,
    WGS84,
    connect,
    ensure_core_tables,
    ensure_spatial,
    implementation_sha256,
)
from svalbard_cable_incident_reconstruction.gear import (
    GearParameters,
    GeometryResult,
    SmoothedTrack,
    estimate_towing_point,
    measure_horizontal_crossings,
    smooth_ais_track,
    solve_path_lag_geometry,
)
from svalbard_cable_incident_reconstruction.sources import validate_bathymetry

EPOCH = np.datetime64("2022-01-07T00:00:00", "us")


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    config_sha256: str
    replay: dict[str, Any]


@dataclass(frozen=True)
class AisModelInput:
    frame: pd.DataFrame
    content_sha256: str
    source_manifest_sha256: str
    acquisition_id: str


def build_geometry_grid(config: MelkartConfig) -> list[GearParameters]:
    scenarios: list[GearParameters] = []
    for index, (ratio, spread) in enumerate(
        itertools.product(
            config.geometry_grid.warp_depth_ratios,
            config.geometry_grid.door_spreads_m,
        )
    ):
        scenarios.append(
            GearParameters(
                scenario_id=f"G{index:04d}",
                antenna_offset_m=config.gear.antenna_offset_m,
                warp_depth_ratio=ratio,
                max_warp_length_m=config.gear.max_warp_length_m,
                depth_offset_m=config.gear.depth_offset_m,
                door_spread_m=spread,
                clump_along_track_offset_m=config.gear.clump_along_track_offset_m,
            )
        )
    return scenarios


def _seconds(timestamp: str) -> float:
    value = np.datetime64(timestamp, "us")
    return float((value - EPOCH) / np.timedelta64(1, "s"))


def _ais_content_sha256(frame: pd.DataFrame, mmsi: int) -> str:
    """A hash records the ordered AIS values used as model inputs."""
    digest = hashlib.sha256(b"svalbard-cable-canonical-ais-v2\0")
    digest.update(struct.pack(">q", mmsi))

    timestamps = pd.to_datetime(frame["timestamp_utc"]).to_numpy()
    timestamps_microseconds = timestamps.astype("datetime64[us]").view("i8")
    timestamp_bytes = np.ascontiguousarray(
        timestamps_microseconds,
        dtype=">i8",
    ).view(np.uint8)
    timestamp_bytes = timestamp_bytes.reshape(-1, 8)

    numeric_bytes = []
    numeric_columns = (
        "longitude",
        "latitude",
        "speed_over_ground_knots",
        "course_over_ground_deg",
    )
    for column in numeric_columns:
        values = np.ascontiguousarray(frame[column].to_numpy(), dtype=">f8")
        numeric_bytes.append(values.view(np.uint8).reshape(-1, 8))

    hash_rows = np.concatenate([timestamp_bytes, *numeric_bytes], axis=1)
    digest.update(hash_rows.tobytes())

    return digest.hexdigest()


def _ais_manifest_sha256(manifest_rows: list[tuple[Any, ...]]) -> str:
    """A hash records the source responses used for the selected AIS data."""
    response_identities = []

    for row in manifest_rows:
        response_identities.append(
            {
                "request_id": str(row[1]),
                "response_sha256": str(row[2]),
            }
        )

    manifest = json.dumps(
        response_identities,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(manifest).hexdigest()


def _load_ais(
    db_path: Path,
    config: MelkartConfig,
) -> AisModelInput:
    with connect(db_path, read_only=True) as connection:
        frame = connection.execute(
            """
            SELECT timestamp_utc, longitude, latitude,
                   speed_over_ground_knots, course_over_ground_deg
            FROM clean.ais
            WHERE dataset_id = 'incident'
              AND mmsi = ? AND timestamp_utc >= ? AND timestamp_utc <= ?
            ORDER BY timestamp_utc, longitude, latitude
            """,
            [
                config.project.mmsi,
                config.project.smooth_start_utc,
                config.project.smooth_end_utc,
            ],
        ).fetchdf()
        manifest_rows = connection.execute(
            """
            SELECT acquisition_id, request_id, response_sha256
            FROM raw.ais_api_requests
            WHERE dataset_id = 'incident'
            ORDER BY request_id
            """
        ).fetchall()
    if frame.empty:
        raise RuntimeError("No Melkart-5 AIS rows in the configured smoothing window")
    acquisitions = {str(row[0]) for row in manifest_rows}
    if len(acquisitions) != 1:
        raise RuntimeError("Expected one canonical AIS API acquisition manifest")
    if frame[["speed_over_ground_knots", "course_over_ground_deg"]].isna().any().any():
        raise RuntimeError(
            "Incident AIS requires SOG and COG for position quality checks"
        )

    acquisition_id = acquisitions.pop()
    longitudes = frame["longitude"].to_numpy(dtype=np.float64)
    latitudes = frame["latitude"].to_numpy(dtype=np.float64)
    project = Transformer.from_crs(WGS84, LOCAL_CRS, always_xy=True)
    projected_x, projected_y = project.transform(longitudes, latitudes)
    frame = frame.assign(
        x=np.asarray(projected_x),
        y=np.asarray(projected_y),
        time_s=np.array(
            [
                float((np.datetime64(value, "us") - EPOCH) / np.timedelta64(1, "s"))
                for value in frame.timestamp_utc
            ],
            dtype=np.float64,
        ),
    )
    return AisModelInput(
        frame=frame,
        content_sha256=_ais_content_sha256(frame, config.project.mmsi),
        source_manifest_sha256=_ais_manifest_sha256(manifest_rows),
        acquisition_id=acquisition_id,
    )


@dataclass(frozen=True)
class AisQualitySummary:
    source_position_count: int
    position_group_count: int
    rejected_group_count: int
    spline_group_count: int


def _build_position_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    groups: list[dict[str, Any]] = []
    for _, group in frame.groupby("position_group_id", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        groups.append(
            {
                "position_group_id": len(groups),
                "time_s": (float(first.time_s) + float(last.time_s)) / 2.0,
                "x": float(first.x),
                "y": float(first.y),
                "group_size": len(group),
            }
        )
    return pd.DataFrame(groups)


def _implied_speed(first: pd.Series, second: pd.Series) -> float:
    elapsed_seconds = float(second.time_s - first.time_s)
    if elapsed_seconds <= 0:
        raise RuntimeError("AIS representative times must increase")

    distance = math.hypot(
        float(second.x - first.x),
        float(second.y - first.y),
    )
    return distance / elapsed_seconds


def _is_isolated_position_jump(
    previous: pd.Series,
    current: pd.Series,
    following: pd.Series,
    maximum_speed_m_s: float,
) -> bool:
    inbound_speed = _implied_speed(previous, current)
    outbound_speed = _implied_speed(current, following)
    return inbound_speed > maximum_speed_m_s and outbound_speed > maximum_speed_m_s


def _reject_speed_outliers(
    representatives: pd.DataFrame,
    max_implied_speed_m_s: float,
) -> list[int]:
    """Interior positions are removed when both adjacent speeds exceed the limit."""
    active_indexes = list(range(len(representatives)))
    removed_position = True

    while removed_position and len(active_indexes) >= 4:
        removed_position = False
        accepted_indexes: list[int] = [active_indexes[0]]

        for position in range(1, len(active_indexes) - 1):
            previous = representatives.iloc[active_indexes[position - 1]]
            current = representatives.iloc[active_indexes[position]]
            following = representatives.iloc[active_indexes[position + 1]]

            if _is_isolated_position_jump(
                previous,
                current,
                following,
                max_implied_speed_m_s,
            ):
                removed_position = True
            else:
                accepted_indexes.append(active_indexes[position])

        accepted_indexes.append(active_indexes[-1])
        active_indexes = accepted_indexes

    return active_indexes


def _prepare_ais_track(
    frame: pd.DataFrame,
    config: MelkartConfig,
) -> tuple[Any, AisQualitySummary]:
    """Repeated AIS positions are reduced before isolated jumps are removed and the path is fitted."""
    frame = frame.copy().reset_index(drop=True)
    changed = frame.longitude.ne(frame.longitude.shift()) | frame.latitude.ne(
        frame.latitude.shift()
    )
    frame["position_group_id"] = changed.cumsum().astype(int) - 1

    representatives = _build_position_representatives(frame)
    if len(representatives) < 4:
        raise RuntimeError("Need at least four distinct AIS position groups")

    active = _reject_speed_outliers(
        representatives, config.track_smoothing.max_implied_speed_m_s
    )

    if len(active) < 4:
        raise RuntimeError(
            "Fewer than four AIS position groups remain after quality checks"
        )
    accepted = representatives.iloc[active]
    track = smooth_ais_track(
        accepted.time_s.to_numpy(dtype=np.float64),
        accepted.x.to_numpy(dtype=np.float64),
        accepted.y.to_numpy(dtype=np.float64),
        eval_start_s=_seconds(config.project.smooth_start_utc),
        eval_end_s=_seconds(config.project.smooth_end_utc),
        interval_s=config.project.model_interval_seconds,
        ais_position_rms_m=config.track_smoothing.ais_position_rms_m,
    )

    summary = AisQualitySummary(
        source_position_count=len(frame),
        position_group_count=len(representatives),
        rejected_group_count=len(representatives) - len(active),
        spline_group_count=len(active),
    )
    return track, summary


def _load_cables(db_path: Path, selected_id: str) -> tuple[dict[str, LineString], str]:
    with connect(db_path, read_only=True) as connection:
        ensure_spatial(connection)
        rows = connection.execute(
            """
            SELECT cable_id, ST_AsWKB(geometry), crs_authid, source_sha256
            FROM geo.cable_lines
            """
        ).fetchall()
    cables: dict[str, LineString] = {}
    selected_hash: str | None = None
    for cable_id, geometry_wkb, crs, source_hash in rows:
        geometry = from_wkb(bytes(geometry_wkb))
        if not isinstance(geometry, LineString):
            continue
        project = Transformer.from_crs(crs, LOCAL_CRS, always_xy=True)
        cable_id = str(cable_id)
        cables[cable_id] = transform(project.transform, geometry)
        if cable_id == selected_id:
            selected_hash = str(source_hash)
    if selected_hash is None:
        raise RuntimeError(f"Expected one LineString cable {selected_id!r}")
    return cables, selected_hash


def _candidate_screen_frame(
    *,
    db_path: Path,
    config: MelkartConfig,
    cables: dict[str, LineString],
    run_id: str,
) -> pd.DataFrame:
    interval_start = datetime.fromisoformat(config.failure_interval.start_utc)
    interval_end = datetime.fromisoformat(config.failure_interval.end_utc)
    with connect(db_path, read_only=True) as connection:
        frame = connection.execute(
            """
            SELECT DISTINCT ON (mmsi, timestamp_utc, longitude, latitude)
                   mmsi, longitude, latitude
            FROM clean.ais
            WHERE dataset_id = 'incident'
              AND timestamp_utc >= ? AND timestamp_utc < ?
              AND longitude BETWEEN -180 AND 180
              AND latitude BETWEEN -90 AND 90
            ORDER BY mmsi, timestamp_utc, longitude, latitude
            """,
            [interval_start, interval_end],
        ).fetchdf()
    if frame.empty:
        raise RuntimeError("No AIS positions fall inside the failure interval")

    reach_m = (
        config.gear.max_warp_length_m
        + max(config.geometry_grid.door_spreads_m) / 2.0
        + config.gear.antenna_offset_m
    )
    project = Transformer.from_crs(WGS84, LOCAL_CRS, always_xy=True)
    frame["x"], frame["y"] = project.transform(
        frame.longitude.to_numpy(), frame.latitude.to_numpy()
    )
    rows: list[dict[str, Any]] = []
    for mmsi, vessel in frame.groupby("mmsi", sort=True):
        vessel_coordinates = vessel[["x", "y"]].to_numpy()
        vessel_points = shapely.points(vessel_coordinates)

        for cable_id, cable in sorted(cables.items()):
            distances = np.asarray(shapely.distance(vessel_points, cable))
            minimum_distance = float(np.min(distances))

            rows.append(
                {
                    "run_id": run_id,
                    "mmsi": int(str(mmsi)),
                    "cable_id": cable_id,
                    "minimum_vessel_cable_distance_m": minimum_distance,
                    "conservative_gear_reach_m": reach_m,
                    "within_conservative_gear_reach": minimum_distance <= reach_m,
                }
            )
    return pd.DataFrame(rows)


def _bilinear_value(
    dataset: DatasetReader, longitude: float, latitude: float
) -> float | None:
    inverse = ~dataset.transform
    pixel_x, pixel_y = inverse * (longitude, latitude)
    column = math.floor(pixel_x - 0.5)
    row = math.floor(pixel_y - 0.5)
    dx = pixel_x - (column + 0.5)
    dy = pixel_y - (row + 0.5)
    if (
        row < 0
        or column < 0
        or row + 1 >= dataset.height
        or column + 1 >= dataset.width
    ):
        return None
    values = dataset.read(1, window=((row, row + 2), (column, column + 2)))
    if values.shape != (2, 2) or (
        dataset.nodata is not None and np.any(values == dataset.nodata)
    ):
        return None

    upper_left_weight = (1 - dx) * (1 - dy)
    upper_right_weight = dx * (1 - dy)
    lower_left_weight = (1 - dx) * dy
    lower_right_weight = dx * dy

    elevation = (
        values[0, 0] * upper_left_weight
        + values[0, 1] * upper_right_weight
        + values[1, 0] * lower_left_weight
        + values[1, 1] * lower_right_weight
    )
    if not math.isfinite(elevation) or elevation >= 0:
        return None

    return -float(elevation)


def _depth_function(dataset: DatasetReader) -> Any:
    transformer = Transformer.from_crs(LOCAL_CRS, WGS84, always_xy=True)

    def depth(x: float, y: float) -> float | None:
        longitude, latitude = transformer.transform(x, y)
        return _bilinear_value(dataset, longitude, latitude)

    return depth


def _run_geometry_scenarios(
    scenarios: list[GearParameters],
    *,
    track: Any,
    depth: Any,
    config: MelkartConfig,
) -> list[GeometryResult]:
    analysis_start = _seconds(config.project.analysis_start_utc)
    analysis_end = _seconds(config.project.analysis_end_utc)
    tow_x, tow_y = estimate_towing_point(track, scenarios[0].antenna_offset_m)

    results = []
    for parameters in scenarios:
        result = solve_path_lag_geometry(
            track=track,
            tow_x=tow_x,
            tow_y=tow_y,
            depth=depth,
            params=parameters,
            analysis_start_s=analysis_start,
            analysis_end_s=analysis_end,
        )
        results.append(result)

    return results


def _trajectory_frame(results: list[GeometryResult], run_id: str) -> pd.DataFrame:
    to_wgs84 = Transformer.from_crs(LOCAL_CRS, WGS84, always_xy=True)
    frames: list[pd.DataFrame] = []

    for result in results:
        if result.rejected:
            continue

        for state in result.component_states:
            longitudes, latitudes = to_wgs84.transform(state.x, state.y)
            elapsed_microseconds = (state.time_s * 1_000_000.0).round()
            elapsed = elapsed_microseconds.astype("timedelta64[us]")
            timestamps = EPOCH + elapsed

            frames.append(
                pd.DataFrame(
                    {
                        "run_id": run_id,
                        "scenario_id": result.params.scenario_id,
                        "timestamp_utc": pd.to_datetime(timestamps, utc=False),
                        "component": state.component.value,
                        "latitude": latitudes,
                        "longitude": longitudes,
                    }
                )
            )

    if not frames:
        return pd.DataFrame(
            {
                "run_id": pd.Series(dtype="string"),
                "scenario_id": pd.Series(dtype="string"),
                "timestamp_utc": pd.Series(dtype="datetime64[ns]"),
                "component": pd.Series(dtype="string"),
                "latitude": pd.Series(dtype="float64"),
                "longitude": pd.Series(dtype="float64"),
            }
        )

    return pd.concat(frames, ignore_index=True)


def _replay_data(
    track: SmoothedTrack, results: list[GeometryResult], config: MelkartConfig
) -> dict[str, Any]:
    analysis_start_s = _seconds(config.project.analysis_start_utc)
    analysis_end_s = _seconds(config.project.analysis_end_utc)
    selected = (track.time_s >= analysis_start_s) & (track.time_s <= analysis_end_s)
    tow_x, tow_y = estimate_towing_point(track, config.gear.antenna_offset_m)

    def coordinates(values: Any) -> list[float]:
        return np.round(values, 3).tolist()

    scenarios = []
    for result in results:
        states = {state.component.value: state for state in result.component_states}
        scenario: dict[str, Any] = {
            "scenarioId": result.params.scenario_id,
            "warpDepthRatio": result.params.warp_depth_ratio,
            "doorSpreadM": result.params.door_spread_m,
            "accepted": not result.rejected,
            "rejectionReason": result.rejection_reason or None,
        }
        for component, prefix in (
            ("port_door", "port"),
            ("starboard_door", "stbd"),
            ("centre_clump", "clump"),
        ):
            state = states.get(component)
            scenario[f"{prefix}X"] = coordinates(state.x) if state else []
            scenario[f"{prefix}Y"] = coordinates(state.y) if state else []
        scenarios.append(scenario)

    return {
        "positionBasis": "assumed_path_lag_sensitivity",
        "epochDate": "2022-01-07",
        "aeqdLat0": 78.34,
        "aeqdLon0": 9.55,
        "analysisStartS": analysis_start_s,
        "analysisEndS": analysis_end_s,
        "signalLossS": _seconds(config.failure_interval.start_utc),
        "times": track.time_s[selected].tolist(),
        "x": coordinates(track.x[selected]),
        "y": coordinates(track.y[selected]),
        "towX": coordinates(tow_x[selected]),
        "towY": coordinates(tow_y[selected]),
        "scenarios": scenarios,
    }


def _scenario_frame(results: list[GeometryResult], run_id: str) -> pd.DataFrame:
    rows = []

    for result in results:
        row = {
            "run_id": run_id,
            "scenario_id": result.params.scenario_id,
            "warp_depth_ratio": result.params.warp_depth_ratio,
            "door_spread_m": result.params.door_spread_m,
            "accepted": not result.rejected,
            "rejection_reason": result.rejection_reason or None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _horizontal_crossing_frame(
    results: list[GeometryResult],
    cable: LineString,
    config: MelkartConfig,
    run_id: str,
) -> pd.DataFrame:
    analysis_start_s = _seconds(config.project.analysis_start_utc)
    analysis_end_s = _seconds(config.project.analysis_end_utc)
    failure_start_s = _seconds(config.failure_interval.start_utc)
    failure_end_s = _seconds(config.failure_interval.end_utc)
    to_wgs84 = Transformer.from_crs(LOCAL_CRS, WGS84, always_xy=True)
    rows: list[dict[str, Any]] = []

    for result in results:
        if result.rejected:
            continue

        crossings = measure_horizontal_crossings(
            result.component_states,
            cable,
            interval_start_s=analysis_start_s,
            interval_end_s=analysis_end_s,
        )

        for crossing in crossings:
            longitude, latitude = to_wgs84.transform(
                crossing.crossing_x, crossing.crossing_y
            )
            within_failure_interval = (
                failure_start_s <= crossing.crossing_time_s < failure_end_s
            )

            rows.append(
                {
                    "run_id": run_id,
                    "scenario_id": result.params.scenario_id,
                    "component": crossing.component.value,
                    "crossing_time_s": crossing.crossing_time_s,
                    "crossing_x_m": crossing.crossing_x,
                    "crossing_y_m": crossing.crossing_y,
                    "crossing_latitude": latitude,
                    "crossing_longitude": longitude,
                    "within_failure_interval": within_failure_interval,
                }
            )

    crossings = pd.DataFrame(rows)
    if crossings.empty:
        crossings = pd.DataFrame(
            {
                "run_id": pd.Series(dtype="string"),
                "scenario_id": pd.Series(dtype="string"),
                "component": pd.Series(dtype="string"),
                "crossing_time_s": pd.Series(dtype="float64"),
                "crossing_x_m": pd.Series(dtype="float64"),
                "crossing_y_m": pd.Series(dtype="float64"),
                "crossing_latitude": pd.Series(dtype="float64"),
                "crossing_longitude": pd.Series(dtype="float64"),
                "within_failure_interval": pd.Series(dtype="bool"),
            }
        )
    if not crossings.empty:
        invalid_time = ~crossings.crossing_time_s.between(
            analysis_start_s, analysis_end_s, inclusive="left"
        )
        if invalid_time.any():
            raise RuntimeError("Horizontal crossing lies outside the analysis interval")

        if crossings.duplicated(["scenario_id", "component", "crossing_time_s"]).any():
            raise RuntimeError("Duplicate horizontal crossing event detected")

    return crossings


def run_model(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    config: MelkartConfig,
    publish_results: bool = True,
) -> RunSummary:
    # Fix the run identity before calculation so every output has one provenance key.
    validate_bathymetry(config.bathymetry.path, config.bathymetry)
    config_hash = config.sha256()
    ais_input = _load_ais(db_path, config)
    cables, cable_hash = _load_cables(db_path, config.project.cable_id)
    cable = cables[config.project.cable_id]
    implementation_hash = implementation_sha256()

    run_identity_parts = [
        config_hash,
        ais_input.content_sha256,
        ais_input.source_manifest_sha256,
        cable_hash,
        config.bathymetry.array_sha256,
        implementation_hash,
    ]
    run_identity = "\0".join(run_identity_parts).encode()
    run_digest = hashlib.sha256(run_identity).hexdigest()
    run_id = f"run_{run_digest[:16]}"
    # Calculate all model outputs before the write transaction starts.
    track, ais_summary = _prepare_ais_track(ais_input.frame, config)
    with rasterio.open(config.bathymetry.path) as raster:
        results = _run_geometry_scenarios(
            build_geometry_grid(config),
            track=track,
            depth=_depth_function(raster),
            config=config,
        )
    replay = _replay_data(track, results, config)
    summary = RunSummary(run_id=run_id, config_sha256=config_hash, replay=replay)
    if not publish_results:
        return summary

    positions = _trajectory_frame(results, run_id)
    scenarios = _scenario_frame(results, run_id)
    crossings = _horizontal_crossing_frame(results, cable, config, run_id)
    candidate_screen = _candidate_screen_frame(
        db_path=db_path, config=config, cables=cables, run_id=run_id
    )

    # Replace the current analysis tables together. A reader cannot see a partial run.
    with connect(db_path) as connection:
        ensure_core_tables(connection)

        connection.register("scenarios_frame", scenarios)
        connection.register("positions_frame", positions)
        connection.register("crossings_frame", crossings)
        connection.register("candidate_screen_frame", candidate_screen)

        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                "CREATE OR REPLACE TABLE analysis.gear_scenarios AS SELECT * FROM scenarios_frame"
            )
            connection.execute(
                "CREATE OR REPLACE TABLE analysis.gear_component_positions AS SELECT * FROM positions_frame"
            )
            connection.execute(
                "CREATE OR REPLACE TABLE analysis.gear_horizontal_crossings AS SELECT * FROM crossings_frame"
            )
            connection.execute(
                "CREATE OR REPLACE TABLE analysis.ais_candidate_screen AS SELECT * FROM candidate_screen_frame"
            )
            connection.execute(
                """
                CREATE OR REPLACE TABLE analysis.ais_position_quality AS
                SELECT ?::VARCHAR AS run_id,
                       ?::INTEGER AS source_position_count,
                       ?::INTEGER AS position_group_count,
                       ?::INTEGER AS rejected_group_count,
                       ?::INTEGER AS spline_group_count
                """,
                [
                    run_id,
                    ais_summary.source_position_count,
                    ais_summary.position_group_count,
                    ais_summary.rejected_group_count,
                    ais_summary.spline_group_count,
                ],
            )
            connection.execute(
                """
                INSERT INTO analysis.model_runs (
                    run_id, config_sha256, effective_config_json,
                    ais_acquisition_id, ais_content_sha256,
                    ais_source_manifest_sha256, cable_sha256, bathymetry_sha256,
                    implementation_sha256
                ) VALUES (?, ?, ?::JSON, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                [
                    run_id,
                    config_hash,
                    config.canonical_json(),
                    ais_input.acquisition_id,
                    ais_input.content_sha256,
                    ais_input.source_manifest_sha256,
                    cable_hash,
                    config.bathymetry.array_sha256,
                    implementation_hash,
                ],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    return summary
