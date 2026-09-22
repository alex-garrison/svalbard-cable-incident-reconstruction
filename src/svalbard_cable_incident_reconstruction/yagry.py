"""The Yagry AIS track is used to estimate cable crossing times on the incident day."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
from pyproj import Transformer
from shapely import from_wkb
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from svalbard_cable_incident_reconstruction.config import MelkartConfig
from svalbard_cable_incident_reconstruction.database import WGS84
from svalbard_cable_incident_reconstruction.sources import (
    AisIngestSummary,
    AisSelection,
    ingest_ais,
)

YAGRY_MMSI = 273550600
YAGRY_DATASET_ID = "yagry_2022_01_07"
YAGRY_DAY_START = datetime(2022, 1, 7)


def ingest_yagry(db_path: Path, config: MelkartConfig) -> AisIngestSummary:
    return ingest_ais(
        db_path=db_path,
        config=config.ais,
        selection=AisSelection(
            start_utc=YAGRY_DAY_START,
            end_utc=YAGRY_DAY_START + timedelta(days=1),
            mmsis=(YAGRY_MMSI,),
        ),
        dataset_id=YAGRY_DATASET_ID,
    )


def _load_cable(connection: duckdb.DuckDBPyConnection) -> BaseGeometry:
    row = connection.execute(
        """
        SELECT ST_AsWKB(geometry), crs_authid
        FROM geo.cable_lines
        WHERE cable_id='cable_a'
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("The ingested cable_a geometry is missing")
    project = Transformer.from_crs(row[1], "EPSG:25833", always_xy=True).transform
    return transform(project, from_wkb(bytes(row[0])))


def _load_positions(connection: duckdb.DuckDBPyConnection) -> list[tuple[Any, ...]]:
    rows = connection.execute(
        """
        SELECT timestamp_utc, longitude, latitude
        FROM clean.ais
        WHERE dataset_id=? AND mmsi=?
        ORDER BY timestamp_utc
        """,
        [YAGRY_DATASET_ID, YAGRY_MMSI],
    ).fetchall()
    if len(rows) < 2:
        raise RuntimeError("Yagry AIS day has fewer than two positions")
    return rows


def _intersection_points(geometry: BaseGeometry) -> list[Point]:
    if isinstance(geometry, Point):
        return [geometry]
    if geometry.geom_type == "MultiPoint":
        return list(geometry.geoms)
    return []


def _crossing_timestamps(
    rows: list[tuple[Any, ...]], cable: BaseGeometry
) -> list[datetime]:
    project = Transformer.from_crs(WGS84, "EPSG:25833", always_xy=True).transform
    timestamps: list[datetime] = []

    for start_row, end_row in itertools.pairwise(rows):
        start_time, start_longitude, start_latitude = start_row
        end_time, end_longitude, end_latitude = end_row

        start_point = project(start_longitude, start_latitude)
        end_point = project(end_longitude, end_latitude)
        segment = LineString([start_point, end_point])

        if segment.length == 0:
            continue

        intersection = segment.intersection(cable)
        if intersection.is_empty:
            continue

        for point in _intersection_points(intersection):
            fraction = segment.project(point) / segment.length
            crossing_time = start_time + (end_time - start_time) * fraction
            timestamps.append(crossing_time)

    return sorted(set(timestamps))


def calculate_yagry_metrics(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    timestamps = _crossing_timestamps(
        _load_positions(connection),
        _load_cable(connection),
    )

    first_intersection = None
    last_intersection = None
    if timestamps:
        first_intersection = timestamps[0].isoformat() + "Z"
        last_intersection = timestamps[-1].isoformat() + "Z"

    return {
        "ais.yagry.cable_a.intersection_count": len(timestamps),
        "ais.yagry.cable_a.intersection_timestamps_utc": [
            timestamp.isoformat() + "Z" for timestamp in timestamps
        ],
        "ais.yagry.cable_a.first_intersection_utc": first_intersection,
        "ais.yagry.cable_a.last_intersection_utc": last_intersection,
    }
