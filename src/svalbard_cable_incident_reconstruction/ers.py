"""ERS catch reports provide counts for the article's trawling comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from svalbard_cable_incident_reconstruction.config import ErsConfig
from svalbard_cable_incident_reconstruction.database import connect
from svalbard_cable_incident_reconstruction.sources import acquire_ers_files


def _ers_sql(paths: list[Path]) -> str:
    escaped = [str(path).replace("'", "''") for path in paths]
    path_list = ", ".join(f"'{path}'" for path in escaped)
    return f"""
    CREATE OR REPLACE TABLE raw.ers_dca AS
    SELECT filename AS source_path,
           TRY_CAST(regexp_extract(filename, 'ers-(20[0-9]{{2}})', 1) AS INTEGER)
               AS source_year,
           * EXCLUDE (filename)
    FROM read_csv([{path_list}], delim=';', header=true, all_varchar=true,
                  union_by_name=true, filename=true, nullstr='');

    CREATE OR REPLACE TABLE clean.ers_dca AS
    SELECT TRY_CAST(nullif("Relevant år", '') AS INTEGER) AS relevant_year,
           TRY_STRPTIME(nullif("Starttidspunkt", ''), '%d.%m.%Y %H:%M:%S')
               AS start_timestamp_utc,
           "Hovedområde start (kode)" AS start_main_area_code,
           "Lokasjon start (kode)" AS start_location_code,
           "Redskap - hovedgruppe" AS gear_main_group,
           "Redskap FAO" AS gear_fao,
           "Redskapsspesifikasjon" AS gear_specification,
           "Hovedart FAO" AS main_species_fao
    FROM raw.ers_dca;
    """


def ingest_ers(db_path: Path, config: ErsConfig) -> dict[str, int]:
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")
    paths = acquire_ers_files(config)

    with connect(db_path) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(_ers_sql(paths))
            raw_rows = connection.execute("SELECT count(*) FROM raw.ers_dca").fetchone()
            clean_rows = connection.execute(
                "SELECT count(*) FROM clean.ers_dca"
            ).fetchone()
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    raw_row_count = 0
    if raw_rows:
        raw_row_count = int(raw_rows[0])

    clean_row_count = 0
    if clean_rows:
        clean_row_count = int(clean_rows[0])

    return {"raw_rows": raw_row_count, "catch_rows": clean_row_count}


def calculate_ers_metrics(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, Any]:
    exists = connection.execute(
        """
        SELECT count(*) FROM information_schema.tables
        WHERE table_schema='clean' AND table_name='ers_dca'
        """
    ).fetchone()
    if exists is None or not exists[0]:
        raise RuntimeError("ERS data is missing; run `svalbard-cable reproduce`")

    row = connection.execute(
        """
        SELECT count(*),
               count(*) FILTER (WHERE gear_specification='Dobbeltrål'),
               count(*) FILTER (WHERE gear_specification='Enkeltrål'),
               count(*) FILTER (
                   WHERE gear_specification IS DISTINCT FROM 'Dobbeltrål'
                     AND gear_specification IS DISTINCT FROM 'Enkeltrål'
               ),
               count(*) FILTER (WHERE main_species_fao='Torsk'),
               count(*) FILTER (WHERE main_species_fao='Hyse'),
               count(*) FILTER (WHERE main_species_fao='Dypvannsreke')
        FROM clean.ers_dca
        WHERE (
                start_main_area_code='25' AND start_location_code IN ('04', '05')
              OR start_main_area_code='21' AND start_location_code IN ('25', '26')
              )
          AND relevant_year BETWEEN 2020 AND 2023
          AND month(start_timestamp_utc) BETWEEN 1 AND 4
          AND gear_main_group='Trål'
          AND gear_fao='Bunntrål, otter'
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("ERS metric query returned no result")

    (
        bottom_otter_count,
        double_trawl_count,
        single_trawl_count,
        other_trawl_count,
        cod_count,
        haddock_count,
        shrimp_count,
    ) = row

    return {
        "ers.filter.relevant_years": [2020, 2021, 2022, 2023],
        "ers.filter.months": [1, 2, 3, 4],
        "ers.filter.start_main_area_locations": {
            "25": ["04", "05"],
            "21": ["25", "26"],
        },
        "ers.filter.gear_main_group": "Trål",
        "ers.filter.gear_fao": "Bunntrål, otter",
        "ers.trawl.bottom_otter_catch_row_count": bottom_otter_count,
        "ers.trawl.double_catch_row_count": double_trawl_count,
        "ers.trawl.single_catch_row_count": single_trawl_count,
        "ers.trawl.other_catch_row_count": other_trawl_count,
        "ers.trawl.main_species_cod_catch_row_count": cod_count,
        "ers.trawl.main_species_haddock_catch_row_count": haddock_count,
        "ers.trawl.main_species_shrimp_catch_row_count": shrimp_count,
    }
