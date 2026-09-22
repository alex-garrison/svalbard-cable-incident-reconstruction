"""DuckDB stores the source and model records used in the reconstruction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "db/svalbard_cable_incident_reconstruction.duckdb"
SCHEMAS = ("raw", "geo", "clean", "analysis")
LOCAL_CRS = "+proj=aeqd +lat_0=78.34 +lon_0=9.55 +datum=WGS84 +units=m +no_defs"
WGS84 = "EPSG:4326"


def portable_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (PROJECT_ROOT / path).resolve()

    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def implementation_sha256() -> str:
    digest = hashlib.sha256(b"svalbard-cable-incident-reconstruction-package-v1\0")
    core_modules = (
        "config.py",
        "database.py",
        "ers.py",
        "gear.py",
        "gfw.py",
        "metrics.py",
        "model.py",
        "slope.py",
        "sources.py",
        "yagry.py",
    )

    for path in (Path(__file__).parent / name for name in core_modules):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def ensure_spatial(connection: duckdb.DuckDBPyConnection) -> None:
    try:
        connection.execute("LOAD spatial")
    except duckdb.Error:
        connection.execute("INSTALL spatial")
        connection.execute("LOAD spatial")


def connect(
    db_path: Path = DEFAULT_DB_PATH, *, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    if read_only and not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")

    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(db_path), read_only=read_only)

    if not read_only:
        for schema in SCHEMAS:
            connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    return connection


def require_table_columns(
    connection: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
    required_columns: set[str],
) -> None:
    existing_columns = {
        str(row[0])
        for row in connection.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema=? AND table_name=?
            """,
            [schema, table],
        ).fetchall()
    }

    if existing_columns and existing_columns != required_columns:
        raise RuntimeError(
            f"Existing {schema}.{table} schema is incompatible; use a new --db path"
        )


def ensure_core_tables(connection: duckdb.DuckDBPyConnection) -> None:
    for schema, table, required in (
        (
            "clean",
            "ais",
            {
                "dataset_id",
                "timestamp_utc",
                "mmsi",
                "longitude",
                "latitude",
                "course_over_ground_deg",
                "speed_over_ground_knots",
            },
        ),
        (
            "raw",
            "ais_api_requests",
            {
                "dataset_id",
                "acquisition_id",
                "request_id",
                "payload_json",
                "response_json",
                "response_sha256",
            },
        ),
        (
            "analysis",
            "model_runs",
            {
                "run_id",
                "config_sha256",
                "effective_config_json",
                "ais_acquisition_id",
                "ais_content_sha256",
                "ais_source_manifest_sha256",
                "cable_sha256",
                "bathymetry_sha256",
                "implementation_sha256",
            },
        ),
    ):
        require_table_columns(connection, schema, table, required)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS clean.ais (
            dataset_id VARCHAR NOT NULL,
            timestamp_utc TIMESTAMP NOT NULL,
            mmsi BIGINT NOT NULL,
            longitude DOUBLE NOT NULL,
            latitude DOUBLE NOT NULL,
            course_over_ground_deg DOUBLE,
            speed_over_ground_knots DOUBLE
        );

        CREATE TABLE IF NOT EXISTS analysis.model_runs (
            run_id VARCHAR PRIMARY KEY,
            config_sha256 VARCHAR NOT NULL,
            effective_config_json JSON NOT NULL,
            ais_acquisition_id VARCHAR,
            ais_content_sha256 VARCHAR,
            ais_source_manifest_sha256 VARCHAR,
            cable_sha256 VARCHAR,
            bathymetry_sha256 VARCHAR,
            implementation_sha256 VARCHAR
        );

        CREATE TABLE IF NOT EXISTS raw.ais_api_requests (
            dataset_id VARCHAR NOT NULL,
            acquisition_id VARCHAR NOT NULL,
            request_id VARCHAR NOT NULL,
            payload_json JSON NOT NULL,
            response_json JSON NOT NULL,
            response_sha256 VARCHAR NOT NULL,
            PRIMARY KEY (dataset_id, request_id)
        );
        """
    )


def current_model_run(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[str, str, str]:
    rows = connection.execute(
        """
        SELECT DISTINCT runs.run_id, runs.config_sha256, runs.implementation_sha256
        FROM analysis.gear_scenarios scenarios
        JOIN analysis.model_runs runs USING (run_id)
        """
    ).fetchall()

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one current complete model run, found {len(rows)}"
        )

    return str(rows[0][0]), str(rows[0][1]), str(rows[0][2])
