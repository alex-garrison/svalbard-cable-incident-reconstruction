"""Global Fishing Watch reports provide a separate measure of fishing activity."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from pyproj import Geod

from svalbard_cable_incident_reconstruction.config import load_env
from svalbard_cable_incident_reconstruction.database import PROJECT_ROOT

REPORT_CACHE = PROJECT_ROOT / "data/raw/gfw/fishing_effort_reports"
REPORT_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
DATASET = "public-global-fishing-effort:v3.0"
START_UTC = "2012-01-01T00:00:00Z"
END_UTC = "2022-01-07T03:10:00Z"


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _annual_windows(start_utc: str, end_utc: str) -> list[tuple[str, str]]:
    start = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
    windows = []
    cursor = start
    while cursor < end:
        boundary = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
        window_end = min(boundary, end)
        windows.append(
            (
                cursor.isoformat().replace("+00:00", "Z"),
                window_end.isoformat().replace("+00:00", "Z"),
            )
        )
        cursor = window_end
    return windows


def _geodesic_circle(longitude: float, latitude: float) -> dict[str, Any]:
    geod = Geod(ellps="WGS84")
    ring: list[list[float]] = []

    for bearing_index in range(72):
        bearing_degrees = bearing_index * 5
        point = geod.fwd(longitude, latitude, bearing_degrees, 2_000)
        point_longitude, point_latitude = point[:2]
        ring.append([point_longitude, point_latitude])

    closed_ring = [*ring, ring[0]]
    return {"type": "Polygon", "coordinates": [closed_ring]}


def _hour_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("hours"), int | float):
            return [value]

        records: list[dict[str, Any]] = []
        for nested_value in value.values():
            records.extend(_hour_records(nested_value))
        return records

    if isinstance(value, list):
        records = []
        for nested_value in value:
            records.extend(_hour_records(nested_value))
        return records

    return []


def _response_datasets(value: Any) -> set[str]:
    datasets: set[str] = set()

    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key.startswith("public-global-fishing-effort:"):
                datasets.add(key)
            datasets.update(_response_datasets(nested_value))
        return datasets

    if isinstance(value, list):
        for nested_value in value:
            datasets.update(_response_datasets(nested_value))

    return datasets


def _download_report(
    parameters: dict[str, str], body: dict[str, Any], token: str
) -> bytes:
    maximum_attempts = 12

    for attempt_number in range(1, maximum_attempts + 1):
        try:
            response = requests.post(
                REPORT_URL,
                params=parameters,
                json=body,
                headers={"Authorization": f"Bearer {token}"},
                timeout=180,
            )
        except requests.RequestException:
            if attempt_number == maximum_attempts:
                raise

            time.sleep(1)
            continue

        if response.status_code != 429:
            response.raise_for_status()
            response.json()
            return response.content

        if attempt_number == maximum_attempts:
            response.raise_for_status()

        retry_delay_seconds = float(response.headers.get("Retry-After", 5))
        time.sleep(retry_delay_seconds)

    raise RuntimeError("GFW retry loop ended unexpectedly")


def _load_report(
    parameters: dict[str, str], body: dict[str, Any], token: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_document = {
        "method": "POST",
        "url": REPORT_URL,
        "params": parameters,
        "body": body,
    }
    request_hash = hashlib.sha256(
        json.dumps(request_document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_path = REPORT_CACHE / f"{request_hash}.json"

    if not cache_path.exists():
        content = _download_report(parameters, body, token)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)

    report = json.loads(cache_path.read_bytes())
    records = _hour_records(report)
    provenance = {
        "request_sha256": request_hash,
        "response_sha256": _sha256(cache_path),
        "date_range": parameters["date-range"],
        "row_count": len(records),
    }
    return report, provenance


def calculate_gfw_metrics(longitude: float, latitude: float) -> dict[str, Any]:
    load_env()
    token = os.environ.get("GFW_API_ACCESS_TOKEN")
    if not token:
        return {}

    body = {"geojson": _geodesic_circle(longitude, latitude)}
    requests_provenance = []
    records: list[dict[str, Any]] = []
    datasets: set[str] = set()

    for start_utc, end_utc in _annual_windows(START_UTC, END_UTC):
        parameters = {
            "spatial-resolution": "HIGH",
            "temporal-resolution": "ENTIRE",
            "spatial-aggregation": "true",
            "group-by": "VESSEL_ID",
            "datasets[0]": DATASET,
            "date-range": f"{start_utc},{end_utc}",
            "format": "JSON",
        }
        report, provenance = _load_report(parameters, body, token)
        records.extend(_hour_records(report))
        datasets.update(_response_datasets(report))
        requests_provenance.append(provenance)

    trawler_records = []
    for record in records:
        if record.get("geartype") == "TRAWLERS":
            trawler_records.append(record)

    trawler_vessel_ids = set()
    for record in trawler_records:
        vessel_id = record.get("vesselId")
        if vessel_id:
            trawler_vessel_ids.add(str(vessel_id))

    trawler_hours = sum(float(record["hours"]) for record in trawler_records)
    all_gear_hours = sum(float(record["hours"]) for record in records)

    request_hashes = [item["request_sha256"] for item in requests_provenance]
    response_hashes = [item["response_sha256"] for item in requests_provenance]
    manifest_json = json.dumps(
        requests_provenance,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()

    return {
        "gfw.activity.fishing_hours": round(trawler_hours, 1),
        "gfw.activity.all_gear_fishing_hours": all_gear_hours,
        "gfw.activity.raw_report_row_count": len(records),
        "gfw.activity.trawler_report_row_count": len(trawler_records),
        "gfw.activity.unique_vessel_count": len(trawler_vessel_ids),
        "gfw.activity.request_count": len(requests_provenance),
        "gfw.activity.radius_km": 2.0,
        "gfw.activity.start_utc": START_UTC,
        "gfw.activity.end_utc_exclusive": END_UTC,
        "gfw.activity.requested_dataset": DATASET,
        "gfw.activity.response_datasets": sorted(datasets),
        "gfw.activity.request_sha256": request_hashes,
        "gfw.activity.response_sha256": response_hashes,
        "gfw.activity.request_manifest_sha256": manifest_hash,
        "gfw.activity.centre_latitude": latitude,
        "gfw.activity.centre_longitude": longitude,
    }
