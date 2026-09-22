"""The source pipeline acquires and checks the data used in the reconstruction."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import socket
import sys
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as datetime_time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import duckdb
import numpy as np
import pandas as pd
import rasterio
import requests
from netCDF4 import Dataset
from pyproj import Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds, xy
from scipy.ndimage import binary_dilation, label
from shapely.geometry import LineString, Point
from shapely.ops import transform

from svalbard_cable_incident_reconstruction.config import (
    AisConfig,
    BathymetryConfig,
    CableConfig,
    ErsConfig,
    ErsSourceConfig,
    MelkartConfig,
)
from svalbard_cable_incident_reconstruction.database import (
    DEFAULT_DB_PATH,
    PROJECT_ROOT,
    connect,
    ensure_core_tables,
    ensure_spatial,
    portable_path,
    require_table_columns,
)

EFS_CACHE = PROJECT_ROOT / "data/raw/efs/NoticeToMarinerList_2023_11_EN.XML"
CABLE_REQUEST_MARGIN_M = 2725.0
MAX_WMS_TILE_PIXELS = 8000
WMS_CONNECT_GAP_PIXELS = 4


def _efs_notice(xml: bytes, notice_number: int) -> list[tuple[float, float]]:
    root = ElementTree.fromstring(xml)

    for notice in root.findall(".//notice"):
        if notice.findtext("noticeIdentifier/noticeNumber") != str(notice_number):
            continue

        coordinates: list[tuple[float, float]] = []
        for position in notice.findall(".//positions/position"):
            point = position.find("positionPoint")
            if point is None:
                continue
            # The source swaps the latitude and longitude field names.
            longitude = float(point.findtext("latitudeDecimal", "nan"))
            latitude = float(point.findtext("longitudeDecimal", "nan"))
            if np.isfinite(longitude) and np.isfinite(latitude):
                coordinates.append((longitude, latitude))

        if len(coordinates) >= 2:
            return coordinates

    raise RuntimeError(f"EFS notice {notice_number} was not found")


def ingest_efs(db_path: Path, config: MelkartConfig) -> dict[str, Any]:
    if EFS_CACHE.exists():
        xml = EFS_CACHE.read_bytes()
    else:
        response = requests.get(config.efs.url, timeout=60)
        response.raise_for_status()
        xml = response.content
        EFS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        EFS_CACHE.write_bytes(xml)

    digest = hashlib.sha256(xml).hexdigest()
    if digest != config.efs.sha256:
        raise RuntimeError(f"EFS XML hash mismatch: {digest}")

    coordinates = _efs_notice(xml, config.efs.notice_number)
    line = LineString(coordinates)

    with connect(db_path) as connection:
        ensure_spatial(connection)
        require_table_columns(connection, "raw", "efs_sources", {"sha256", "xml"})
        require_table_columns(
            connection,
            "geo",
            "efs_71611_repair_notice_track_raw",
            {"source_sha256", "crs_authid", "geom"},
        )
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS raw.efs_sources (
                    sha256 VARCHAR PRIMARY KEY,
                    xml VARCHAR
                )
                """,
            )
            connection.execute(
                """
                INSERT INTO raw.efs_sources VALUES (?, ?)
                ON CONFLICT DO NOTHING
                """,
                [digest, xml.decode()],
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS geo.efs_71611_repair_notice_track_raw (
                    source_sha256 VARCHAR PRIMARY KEY,
                    crs_authid VARCHAR,
                    geom GEOMETRY
                )
                """
            )
            connection.execute(
                """
                INSERT INTO geo.efs_71611_repair_notice_track_raw
                VALUES (?, 'OGC:CRS84', ST_GeomFromText(?))
                ON CONFLICT DO NOTHING
                """,
                [digest, line.wkt],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    return {"sha256": digest, "vertex_count": len(coordinates)}


def ingest_sources(db_path: Path, config: MelkartConfig) -> dict[str, Any]:
    print("  Reading EFS notice...", file=sys.stderr, flush=True)
    efs = ingest_efs(db_path, config)
    efs_coordinates = _efs_notice(EFS_CACHE.read_bytes(), config.efs.notice_number)
    cable_bounds = _cable_bounds_from_efs(
        efs_coordinates, config.cable.request_crs, CABLE_REQUEST_MARGIN_M
    )
    print("  Fetching cable geometry...", file=sys.stderr, flush=True)
    cable = ingest_cable(
        db_path=db_path,
        config=config.cable,
        canonical_id=config.project.cable_id,
        bounds=cable_bounds,
    )
    print("  Checking GEBCO bathymetry...", file=sys.stderr, flush=True)
    bathymetry = ingest_bathymetry(config.bathymetry)

    print("  Loading incident AIS...", file=sys.stderr, flush=True)
    incident = ingest_ais(
        db_path=db_path,
        config=config.ais,
        selection=selection_from_config(config.ais),
        dataset_id="incident",
    )

    selection = AisSelection(
        start_utc=datetime.fromisoformat(config.cable_vigilance.start_utc),
        end_utc=datetime.fromisoformat(config.cable_vigilance.end_utc),
        mmsis=(config.cable_vigilance.mmsi,),
    )
    print("  Loading Cable Vigilance AIS...", file=sys.stderr, flush=True)
    vigilance = ingest_ais(
        db_path=db_path,
        config=config.ais,
        selection=selection,
        dataset_id="cable_vigilance",
    )
    return {
        "cable": asdict(cable),
        "bathymetry": portable_path(bathymetry),
        "incident_ais": asdict(incident),
        "cable_vigilance_ais": asdict(vigilance),
        "efs": efs,
    }


def _wms_tiles(
    bounds: tuple[float, float, float, float], resolution: float
) -> list[list[float]]:
    minx, miny, maxx, maxy = bounds
    tile_span = MAX_WMS_TILE_PIXELS * resolution
    tiles: list[list[float]] = []

    for tile_minx in np.arange(minx, maxx, tile_span):
        for tile_miny in np.arange(miny, maxy, tile_span):
            tiles.append(
                [
                    float(tile_minx),
                    float(tile_miny),
                    float(min(tile_minx + tile_span, maxx)),
                    float(min(tile_miny + tile_span, maxy)),
                ]
            )

    return tiles


def _cable_bounds_from_efs(
    coordinates: list[tuple[float, float]], crs: str, margin_m: float
) -> tuple[float, float, float, float]:
    projected = transform(
        Transformer.from_crs("OGC:CRS84", crs, always_xy=True).transform,
        LineString(coordinates),
    )
    minx, miny, maxx, maxy = projected.bounds
    return (
        minx - margin_m,
        miny - margin_m,
        maxx + margin_m,
        maxy + margin_m,
    )


def _largest_connected_cable_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask

    expanded = binary_dilation(mask, iterations=WMS_CONNECT_GAP_PIXELS)
    labels, count = label(expanded, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels[mask], minlength=count + 1)
    largest = int(np.argmax(sizes[1:]) + 1)

    return np.asarray(mask & (labels == largest))


def _initial_wms_segment(
    segments: list[LineString],
) -> tuple[list[tuple[float, float]], list[LineString]]:
    remaining_segments = list(segments)
    initial_segment_index = 0
    initial_endpoint = min(
        remaining_segments[0].coords[0],
        remaining_segments[0].coords[-1],
    )

    for segment_index, segment in enumerate(remaining_segments[1:], start=1):
        minimum_endpoint = min(segment.coords[0], segment.coords[-1])
        if minimum_endpoint < initial_endpoint:
            initial_segment_index = segment_index
            initial_endpoint = minimum_endpoint

    initial_coordinates = list(remaining_segments.pop(initial_segment_index).coords)
    if initial_coordinates[-1] < initial_coordinates[0]:
        initial_coordinates.reverse()

    return initial_coordinates, remaining_segments


def _nearest_wms_segment(
    current_endpoint: Point,
    segments: list[LineString],
) -> tuple[float, int, int]:
    endpoint_candidates: list[tuple[float, int, int]] = []

    for segment_index, segment in enumerate(segments):
        for endpoint_index in (0, -1):
            endpoint = Point(segment.coords[endpoint_index])
            gap_m = current_endpoint.distance(endpoint)
            endpoint_candidates.append((gap_m, segment_index, endpoint_index))

    return min(endpoint_candidates)


def _stitch_wms_segments(segments: list[LineString]) -> LineString:
    if not segments:
        raise RuntimeError(
            "WMS returned no connected kabel_l pixels in the requested footprint"
        )

    stitched_coordinates, remaining_segments = _initial_wms_segment(segments)

    while remaining_segments:
        current_endpoint = Point(stitched_coordinates[-1])
        gap_m, segment_index, endpoint_index = _nearest_wms_segment(
            current_endpoint,
            remaining_segments,
        )
        if gap_m > 10.0:
            raise RuntimeError(f"Connected WMS cable tiles have a {gap_m:.1f} m gap")

        next_coordinates = list(remaining_segments.pop(segment_index).coords)
        if endpoint_index == -1:
            next_coordinates.reverse()

        stitched_coordinates.extend(next_coordinates[1:])

    return LineString(stitched_coordinates)


@dataclass(frozen=True)
class _WmsTileImage:
    bounds: tuple[float, float, float, float]
    width: int
    height: int
    pixels: np.ndarray


def _request_wms_tile(
    config: CableConfig,
    bounds: list[float],
) -> _WmsTileImage:
    minx, miny, maxx, maxy = bounds
    width = round((maxx - minx) / config.resolution_m)
    height = round((maxy - miny) / config.resolution_m)
    params: dict[str, str | int] = {
        "language": "eng",
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": config.wms_layer,
        "STYLES": "default",
        "CRS": config.request_crs,
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
    }
    response = requests.get(config.wms_url, params=params, timeout=120)
    response.raise_for_status()
    # The WMS request bounds locate this PNG; the image has no geotransform.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile(response.content) as memory, memory.open() as dataset:
            pixels = dataset.read()

    return _WmsTileImage(
        bounds=(minx, miny, maxx, maxy),
        width=width,
        height=height,
        pixels=pixels,
    )


def _wms_tile_segment(tile: _WmsTileImage) -> LineString | None:
    if len(tile.pixels) >= 4:
        alpha = tile.pixels[3]
    else:
        visible_pixels = np.any(tile.pixels[:3], axis=0)
        alpha = visible_pixels * 255

    selected_mask = _largest_connected_cable_mask(alpha > 0)
    rows, columns = np.where(selected_mask)
    if not len(rows):
        return None

    minx, miny, maxx, maxy = tile.bounds
    raster_transform = from_bounds(
        minx,
        miny,
        maxx,
        maxy,
        tile.width,
        tile.height,
    )
    points: list[tuple[float, float]] = []

    for column in range(int(columns.min()), int(columns.max()) + 1, 4):
        valid_rows = np.where(selected_mask[:, column])[0]
        if not len(valid_rows):
            continue

        weighted_row = float(np.average(valid_rows, weights=alpha[valid_rows, column]))
        x, y = xy(raster_transform, weighted_row, column, offset="center")
        points.append((float(x), float(y)))

    if len(points) < 2:
        return None

    return LineString(points)


def _vectorize_wms(
    config: CableConfig,
    tile_bounds: list[list[float]],
) -> LineString:
    segments: list[LineString] = []

    for bounds in tile_bounds:
        tile = _request_wms_tile(config, bounds)
        segment = _wms_tile_segment(tile)
        if segment is not None:
            segments.append(segment)

    return _stitch_wms_segments(segments)


API_POSITION_WIDTH = 12
API_MMSI_INDEX = 0
API_TIMESTAMP_INDEX = 1
API_LONGITUDE_INDEX = 2
API_LATITUDE_INDEX = 3
API_COURSE_INDEX = 4
API_SPEED_INDEX = 5
API_MESSAGE_TYPE_INDEX = 6


@dataclass(frozen=True)
class AisSelection:
    start_utc: datetime
    end_utc: datetime
    mmsis: tuple[int, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    min_speed_knots: float = 0.0

    def validate(self) -> None:
        if self.start_utc >= self.end_utc:
            raise ValueError("AIS start must be before end")
        if self.bbox is not None:
            west, south, east, north = self.bbox
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError(f"Invalid AIS bounding box: {self.bbox}")
        if self.min_speed_knots < 0:
            raise ValueError("Minimum AIS speed cannot be negative")


@dataclass(frozen=True)
class AisIngestSummary:
    dataset_id: str
    acquisition_id: str
    row_count: int
    mmsi_count: int
    api_partitions: int
    first_seen_utc: datetime | None
    last_seen_utc: datetime | None


def selection_from_config(config: AisConfig) -> AisSelection:
    return AisSelection(
        start_utc=datetime.fromisoformat(config.selection_start_utc),
        end_utc=datetime.fromisoformat(config.selection_end_utc),
        mmsis=config.mmsis,
        bbox=config.bbox,
        min_speed_knots=config.min_speed_knots,
    )


def requested_dates(selection: AisSelection) -> tuple[date, ...]:
    final_instant = selection.end_utc - timedelta(microseconds=1)
    day = selection.start_utc.date()
    days: list[date] = []

    while day <= final_instant.date():
        days.append(day)
        day += timedelta(days=1)

    return tuple(days)


def _day_interval(day: date, selection: AisSelection) -> tuple[datetime, datetime]:
    start = datetime.combine(day, datetime_time.min)
    end = start + timedelta(days=1)
    return max(start, selection.start_utc), min(end, selection.end_utc)


def _request_id(endpoint: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"endpoint": endpoint, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _api_request(
    selection: AisSelection, start: datetime, end: datetime
) -> tuple[str, dict[str, Any]]:
    if start.second or start.microsecond or end.second or end.microsecond:
        raise ValueError("AIS API request bounds must use whole-minute precision")

    if selection.mmsis:
        endpoint = "ais/positions/for-mmsis-time"
        payload: dict[str, Any] = {
            "mmsiIds": list(selection.mmsis),
            "start": start.strftime("%Y%m%d%H%M"),
            "end": end.strftime("%Y%m%d%H%M"),
            "minSpeed": selection.min_speed_knots,
        }
    elif selection.bbox:
        endpoint = "ais/positions/within-bbox-time"
        payload = {
            "bbox": ",".join(str(value) for value in selection.bbox),
            "start": start.strftime("%Y%m%d%H%M"),
            "end": end.strftime("%Y%m%d%H%M"),
            "minSpeed": selection.min_speed_knots,
        }
    else:
        raise ValueError("AIS API request requires MMSIs or bbox")

    return endpoint, payload


def _post_with_retry(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    *,
    max_retries: int,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            response = session.post(url, json=payload, timeout=235)

            if response.status_code in {408, 429, 500, 502, 503, 504}:
                if attempt >= max_retries:
                    response.raise_for_status()

                retry_after = response.headers.get("retry-after")
                if retry_after:
                    delay = float(retry_after)
                else:
                    delay = min(60.0, 2.0**attempt)

                time.sleep(max(1.0, delay))
                continue

            response.raise_for_status()
            body = response.json()

            if not isinstance(body, dict) or body.get("success") is not True:
                raise RuntimeError(f"Kystdatahuset request failed: {body}")

            return body
        except (requests.ConnectionError, requests.Timeout):
            if attempt >= max_retries:
                raise
            time.sleep(min(60.0, 2.0**attempt))

    raise RuntimeError("Kystdatahuset retry loop ended unexpectedly")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    return int(value)


def _map_api_position(
    row: list[Any],
    row_index: int,
    selection: AisSelection,
    start: datetime,
    end: datetime,
    request_id: str,
) -> dict[str, Any] | None:
    if not isinstance(row, list) or len(row) != API_POSITION_WIDTH:
        raise RuntimeError(f"Unexpected AIS API row {row_index}: expected 12 values")

    mmsi_value = row[API_MMSI_INDEX]
    timestamp_value = row[API_TIMESTAMP_INDEX]
    longitude_value = row[API_LONGITUDE_INDEX]
    latitude_value = row[API_LATITUDE_INDEX]
    course_value = row[API_COURSE_INDEX]
    speed_value = row[API_SPEED_INDEX]
    message_type_value = row[API_MESSAGE_TYPE_INDEX]

    timestamp = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00"))
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(UTC).replace(tzinfo=None)

    longitude = float(longitude_value)
    latitude = float(latitude_value)

    if not start <= timestamp < end:
        return None

    valid_world_coordinate = -180 <= longitude <= 180 and -90 <= latitude <= 90
    if not valid_world_coordinate:
        raise RuntimeError(f"Invalid API coordinate: {longitude}, {latitude}")

    if selection.bbox:
        west, south, east, north = selection.bbox
        inside_selection = west <= longitude <= east and south <= latitude <= north
        if not inside_selection:
            return None

    return {
        "timestamp_utc": timestamp,
        "mmsi": int(mmsi_value),
        "longitude": longitude,
        "latitude": latitude,
        "course_over_ground_deg": _optional_float(course_value),
        "speed_over_ground_knots": _optional_float(speed_value),
        "message_type": _optional_int(message_type_value),
        "source_request_id": request_id,
    }


def map_api_positions(
    rows: list[list[Any]],
    *,
    selection: AisSelection,
    start: datetime,
    end: datetime,
    request_id: str,
) -> pd.DataFrame:
    """Kystdatahuset reports are checked and mapped to the common AIS fields."""
    mapped = []

    for row_index, row in enumerate(rows):
        position = _map_api_position(
            row,
            row_index,
            selection,
            start,
            end,
            request_id,
        )
        if position is not None:
            mapped.append(position)

    return pd.DataFrame(mapped)


def _load_or_request_api_response(
    *,
    config: AisConfig,
    endpoint: str,
    payload: dict[str, Any],
    raw_path: Path,
    manifest_path: Path,
    session: requests.Session,
) -> tuple[dict[str, Any], str]:
    if raw_path.exists():
        response_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if manifest_path.exists():
            expected_sha256 = json.loads(manifest_path.read_text()).get(
                "response_sha256"
            )
            if expected_sha256 and response_sha256 != expected_sha256:
                raise RuntimeError(f"Cached AIS response changed: {raw_path}")
        return json.loads(raw_path.read_text()), response_sha256

    body = _post_with_retry(
        session,
        f"{config.api_base_url}/{endpoint}",
        payload,
        max_retries=config.max_retries,
    )
    raw_path.write_text(json.dumps(body, indent=2) + "\n")
    response_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    return body, response_sha256


def fetch_api_partition(
    *,
    config: AisConfig,
    selection: AisSelection,
    day: date,
    session: requests.Session,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start, end = _day_interval(day, selection)
    endpoint, payload = _api_request(selection, start, end)
    request_id = _request_id(endpoint, payload)

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = config.cache_dir / f"{day.isoformat()}_{request_id}.json"
    manifest_path = raw_path.with_suffix(".manifest.json")
    body, response_sha256 = _load_or_request_api_response(
        config=config,
        endpoint=endpoint,
        payload=payload,
        raw_path=raw_path,
        manifest_path=manifest_path,
        session=session,
    )

    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text())
        retrieved_at_utc = previous_manifest.get("retrieved_at_utc")
    else:
        retrieved_at_utc = datetime.now(UTC).isoformat()

    manifest = {
        "endpoint": endpoint,
        "payload": payload,
        "request_id": request_id,
        "retrieved_at_utc": retrieved_at_utc,
        "response_sha256": response_sha256,
        "row_count": len(body.get("data") or []),
        "source_file": portable_path(raw_path),
        "response_json": json.dumps(body, sort_keys=True, separators=(",", ":")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    data = body.get("data")
    if body.get("success") is not True or not isinstance(data, list):
        raise RuntimeError(f"Invalid cached Kystdatahuset response: {raw_path}")

    return (
        map_api_positions(
            data,
            selection=selection,
            start=start,
            end=end,
            request_id=request_id,
        ),
        manifest,
    )


def _acquisition_identity(
    dataset_id: str,
    selection: AisSelection,
    manifests: list[dict[str, Any]],
) -> str:
    partitions = []

    for manifest in manifests:
        partitions.append(
            {
                "request_id": manifest["request_id"],
                "response_sha256": manifest["response_sha256"],
            }
        )

    acquisition = {
        "dataset_id": dataset_id,
        "selection": {
            "start_utc": selection.start_utc.isoformat(),
            "end_utc": selection.end_utc.isoformat(),
            "mmsis": list(selection.mmsis),
            "bbox": selection.bbox,
            "min_speed_knots": selection.min_speed_knots,
        },
        "partitions": partitions,
    }
    acquisition_json = json.dumps(
        acquisition,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(acquisition_json.encode()).hexdigest()


def _manifest_frame(
    dataset_id: str,
    acquisition_id: str,
    manifests: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for manifest in manifests:
        rows.append(
            {
                "dataset_id": dataset_id,
                "acquisition_id": acquisition_id,
                "request_id": manifest["request_id"],
                "payload_json": json.dumps(manifest["payload"], sort_keys=True),
                "response_sha256": manifest["response_sha256"],
                "response_json": manifest["response_json"],
            }
        )

    return pd.DataFrame(rows)


def _acquire_ais_partitions(
    config: AisConfig,
    selection: AisSelection,
    dataset_id: str,
    days: tuple[date, ...],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    api_frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []

    with requests.Session() as session:
        session.headers.update(
            {"User-Agent": "svalbard-cable-incident-reconstruction/0.1"}
        )

        for day_index, day in enumerate(days):
            frame, manifest = fetch_api_partition(
                config=config,
                selection=selection,
                day=day,
                session=session,
            )
            frame.insert(0, "dataset_id", dataset_id)
            api_frames.append(frame)
            manifests.append(manifest)

            has_next_day = day_index + 1 < len(days)
            if has_next_day:
                time.sleep(config.request_delay_seconds)

    return api_frames, manifests


def _stage_ais_frames(
    connection: duckdb.DuckDBPyConnection,
    api_frames: list[pd.DataFrame],
) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE ais_stage (
            dataset_id VARCHAR,
            timestamp_utc TIMESTAMP,
            mmsi BIGINT,
            longitude DOUBLE,
            latitude DOUBLE,
            course_over_ground_deg DOUBLE,
            speed_over_ground_knots DOUBLE,
            message_type SMALLINT,
            source_request_id VARCHAR
        )
        """
    )

    for frame in api_frames:
        if frame.empty:
            continue

        connection.register("api_frame", frame)
        connection.execute("INSERT INTO ais_stage SELECT * FROM api_frame")
        connection.unregister("api_frame")


def _replace_ais_positions(
    connection: duckdb.DuckDBPyConnection,
    dataset_id: str,
) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE ais_ranked AS
        SELECT *,
            -- Prefer high-rate position reports when one timestamp has
            -- more than one AIS message type.
            CASE
                WHEN message_type IN (1, 2, 3) THEN 0
                WHEN message_type IN (18, 19) THEN 1
                WHEN message_type = 27 THEN 2
                ELSE 3
            END AS message_priority,
            row_number() OVER (
                PARTITION BY dataset_id, mmsi, timestamp_utc
                ORDER BY message_priority,
                         coalesce(message_type, 999),
                         longitude, latitude,
                         coalesce(source_request_id, '')
            ) AS position_rank
        FROM ais_stage;
        """
    )
    connection.execute("DELETE FROM clean.ais WHERE dataset_id = ?", [dataset_id])
    connection.execute(
        """
        INSERT INTO clean.ais BY NAME
        SELECT dataset_id, timestamp_utc, mmsi, longitude, latitude,
               course_over_ground_deg, speed_over_ground_knots
        FROM ais_ranked
        WHERE position_rank = 1
        """
    )


def _replace_ais_manifests(
    connection: duckdb.DuckDBPyConnection,
    dataset_id: str,
    acquisition_id: str,
    manifests: list[dict[str, Any]],
) -> None:
    manifest_frame = _manifest_frame(dataset_id, acquisition_id, manifests)
    connection.register("ais_manifest_frame", manifest_frame)
    connection.execute(
        "DELETE FROM raw.ais_api_requests WHERE dataset_id = ?",
        [dataset_id],
    )
    connection.execute(
        """
        INSERT INTO raw.ais_api_requests BY NAME
        SELECT
            dataset_id::VARCHAR AS dataset_id,
            acquisition_id::VARCHAR AS acquisition_id,
            request_id::VARCHAR AS request_id,
            payload_json::JSON AS payload_json,
            response_json::JSON AS response_json,
            response_sha256::VARCHAR AS response_sha256
        FROM ais_manifest_frame
        """
    )
    connection.unregister("ais_manifest_frame")


def _ais_ingest_summary(
    connection: duckdb.DuckDBPyConnection,
    dataset_id: str,
    acquisition_id: str,
    partition_count: int,
) -> AisIngestSummary:
    row = connection.execute(
        """
        SELECT count(*), count(DISTINCT mmsi), min(timestamp_utc), max(timestamp_utc)
        FROM clean.ais
        WHERE dataset_id = ?
        """,
        [dataset_id],
    ).fetchone()
    if row is None:
        raise RuntimeError("AIS summary query returned no result")

    row_count, mmsi_count, first_seen, last_seen = row
    return AisIngestSummary(
        dataset_id=dataset_id,
        acquisition_id=acquisition_id,
        row_count=int(row_count),
        mmsi_count=int(mmsi_count),
        api_partitions=partition_count,
        first_seen_utc=first_seen,
        last_seen_utc=last_seen,
    )


def ingest_ais(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    config: AisConfig,
    selection: AisSelection,
    dataset_id: str = "incident",
) -> AisIngestSummary:
    if not dataset_id or not dataset_id.replace("_", "").isalnum():
        raise ValueError(f"Invalid AIS dataset_id: {dataset_id!r}")

    selection.validate()
    days = requested_dates(selection)
    api_frames, manifests = _acquire_ais_partitions(
        config,
        selection,
        dataset_id,
        days,
    )
    acquisition_id = _acquisition_identity(dataset_id, selection, manifests)

    with connect(db_path) as connection:
        ensure_core_tables(connection)
        ensure_spatial(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            _stage_ais_frames(connection, api_frames)
            _replace_ais_positions(connection, dataset_id)
            _replace_ais_manifests(
                connection,
                dataset_id,
                acquisition_id,
                manifests,
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        return _ais_ingest_summary(
            connection,
            dataset_id,
            acquisition_id,
            len(days),
        )


NODATA = -32767


@contextlib.contextmanager
def _network_timeout(seconds: float):
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _download_file(
    url: str,
    path: Path,
    timeout_seconds: float,
    max_retries: int,
) -> None:
    transient_statuses = {408, 429, 500, 502, 503, 504}

    for attempt in range(max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout_seconds) as response:
                if response.status_code in transient_statuses:
                    raise requests.HTTPError(response=response)
                response.raise_for_status()

                with path.open("wb") as output:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            output.write(block)
                return
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            path.unlink(missing_ok=True)
            is_transient_http = (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code in transient_statuses
            )
            if attempt == max_retries or (
                isinstance(exc, requests.HTTPError) and not is_transient_http
            ):
                raise
            time.sleep(2**attempt)

    raise RuntimeError(f"Download did not complete: {url}")


def _validate_file_sha256(path: Path, expected: str, label: str) -> str:
    digest = _file_sha256(path)
    if digest != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {digest}")
    return digest


def _ers_paths(config: ErsConfig, source: ErsSourceConfig) -> tuple[Path, Path]:
    stem = f"elektronisk-rapportering-ers-{source.year}"
    archive = config.directory / f"{stem}.zip"
    report = config.directory / stem / f"{stem}-fangstmelding-dca.csv"
    return archive, report


def _extract_ers_report(archive: Path, report: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    partial = report.with_suffix(report.suffix + ".part")

    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if Path(name).name == report.name]
        if len(members) != 1:
            raise RuntimeError(
                f"ERS archive must contain one {report.name}; found {len(members)}"
            )
        with bundle.open(members[0]) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output)


def _acquire_ers_source(config: ErsConfig, source: ErsSourceConfig) -> Path:
    archive, report = _ers_paths(config, source)
    if report.exists():
        _validate_file_sha256(report, source.dca_sha256, f"ERS {source.year} DCA")
        return report.resolve()

    archive.parent.mkdir(parents=True, exist_ok=True)
    downloaded_archive = not archive.exists()
    archive_input = archive
    if downloaded_archive:
        archive_input = archive.with_suffix(archive.suffix + ".part")
        _download_file(
            source.url,
            archive_input,
            config.request_timeout_seconds,
            config.max_retries,
        )

    report_partial = report.with_suffix(report.suffix + ".part")
    try:
        _extract_ers_report(archive_input, report)
        _validate_file_sha256(
            report_partial,
            source.dca_sha256,
            f"ERS {source.year} DCA",
        )
        report_partial.replace(report)
        if downloaded_archive:
            archive_input.replace(archive)
    except Exception:
        report_partial.unlink(missing_ok=True)
        if downloaded_archive:
            archive_input.unlink(missing_ok=True)
        raise

    return report.resolve()


def acquire_ers_files(config: ErsConfig) -> list[Path]:
    return [_acquire_ers_source(config, source) for source in config.sources]


def validate_bathymetry(path: Path, config: BathymetryConfig) -> str:
    """The GEBCO raster is checked against its dimensions, CRS, bounds, and elevation hash."""
    if not path.exists():
        raise FileNotFoundError(f"GEBCO raster not found: {path}")

    with rasterio.open(path) as dataset:
        if (dataset.height, dataset.width) != (config.height, config.width):
            raise RuntimeError("GEBCO dimensions do not match config")
        if dataset.crs is None or dataset.crs.to_epsg() != 4326:
            raise RuntimeError(f"Unexpected GEBCO CRS: {dataset.crs}")
        expected = (config.west, config.south, config.east, config.north)
        if not np.allclose(tuple(dataset.bounds), expected, atol=1e-9):
            raise RuntimeError(f"Unexpected GEBCO bounds: {dataset.bounds}")
        digest = array_sha256(dataset.read(1))

    if digest != config.array_sha256:
        raise RuntimeError(
            f"GEBCO hash mismatch: expected {config.array_sha256}, got {digest}"
        )

    return digest


def ingest_bathymetry(config: BathymetryConfig) -> Path:
    if config.path.exists():
        validate_bathymetry(config.path, config)
        return config.path

    with _network_timeout(300), Dataset(config.opendap_url) as dataset:
        longitudes = np.asarray(dataset.variables["lon"][:])
        latitudes = np.asarray(dataset.variables["lat"][:])
        x = np.where((longitudes >= config.west) & (longitudes < config.east))[0]
        y = np.where((latitudes >= config.south) & (latitudes < config.north))[0]
        if len(x) != config.width or len(y) != config.height:
            raise RuntimeError("GEBCO coordinate selection does not match config")
        elevations = np.asarray(
            dataset.variables["elevation"][y[0] : y[-1] + 1, x[0] : x[-1] + 1],
            dtype=np.int16,
        )[::-1]

    digest = array_sha256(elevations)
    if digest != config.array_sha256:
        raise RuntimeError(
            f"Downloaded GEBCO hash mismatch: expected {config.array_sha256}, got {digest}"
        )

    config.path.parent.mkdir(parents=True, exist_ok=True)
    partial = config.path.with_suffix(config.path.suffix + ".part")

    with rasterio.open(
        partial,
        "w",
        driver="GTiff",
        width=config.width,
        height=config.height,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=from_bounds(
            config.west,
            config.south,
            config.east,
            config.north,
            config.width,
            config.height,
        ),
        nodata=NODATA,
    ) as output:
        output.write(elevations, 1)

    partial.replace(config.path)
    config.path.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": config.opendap_url,
                "bounds": [config.west, config.south, config.east, config.north],
                "array_sha256": digest,
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )

    validate_bathymetry(config.path, config)

    return config.path


@dataclass(frozen=True)
class CableIngestSummary:
    row_count: int
    cable_id: str
    source_sha256: str
    vertex_count: int


def ingest_cable(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    config: CableConfig,
    canonical_id: str,
    bounds: tuple[float, float, float, float],
) -> CableIngestSummary:
    """The cable centreline is extracted from the live WMS image and checked against the accepted hash."""
    tile_bounds = _wms_tiles(bounds, config.resolution_m)
    projected = _vectorize_wms(config, tile_bounds)
    to_wgs84 = Transformer.from_crs(config.request_crs, 4326, always_xy=True)
    geometry = transform(to_wgs84.transform, projected)
    source_sha256 = hashlib.sha256(geometry.wkb).hexdigest()

    if source_sha256 != config.accepted_vector_wkb_sha256:
        raise RuntimeError(
            "Live WMS cable geometry is not accepted: "
            f"expected {config.accepted_vector_wkb_sha256}, got {source_sha256}"
        )
    with connect(db_path) as connection:
        ensure_core_tables(connection)
        ensure_spatial(connection)
        require_table_columns(
            connection,
            "geo",
            "cable_lines",
            {"cable_id", "source_sha256", "crs_authid", "geometry"},
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS geo.cable_lines (
                cable_id VARCHAR,
                source_sha256 VARCHAR,
                crs_authid VARCHAR,
                geometry GEOMETRY
            )
            """
        )
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                """
                DELETE FROM geo.cable_lines WHERE cable_id = ?
                """,
                [canonical_id],
            )
            connection.execute(
                """
                INSERT INTO geo.cable_lines (
                    cable_id, source_sha256, crs_authid, geometry
                ) VALUES (?, ?, 'OGC:CRS84', ST_GeomFromText(?))
                """,
                [
                    canonical_id,
                    source_sha256,
                    geometry.wkt,
                ],
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    return CableIngestSummary(
        row_count=1,
        cable_id=canonical_id,
        source_sha256=source_sha256,
        vertex_count=len(geometry.coords),
    )
