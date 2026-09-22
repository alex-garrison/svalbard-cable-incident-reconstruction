"""The command-line interface reproduces the analysis and exports its results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

from svalbard_cable_incident_reconstruction.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    load_env,
)
from svalbard_cable_incident_reconstruction.database import (
    DEFAULT_DB_PATH,
    PROJECT_ROOT,
)
from svalbard_cable_incident_reconstruction.ers import ingest_ers
from svalbard_cable_incident_reconstruction.metrics import (
    calculate_metrics,
    write_publication_metrics,
)
from svalbard_cable_incident_reconstruction.model import run_model
from svalbard_cable_incident_reconstruction.publication import (
    playback_data,
    publication_data,
)
from svalbard_cable_incident_reconstruction.sources import ingest_sources
from svalbard_cable_incident_reconstruction.yagry import ingest_yagry

DEFAULT_REPRODUCTION_REPORT_PATH = PROJECT_ROOT / "outputs/reproduction_results.json"
DEFAULT_PLAYBACK_REPORT_PATH = PROJECT_ROOT / "outputs/playback_data.json"


@contextmanager
def _stage(number: int, name: str, stage_count: int = 6) -> Iterator[None]:
    started = perf_counter()
    print(f"[{number}/{stage_count}] {name}...", file=sys.stderr, flush=True)
    try:
        yield
    except Exception:
        elapsed = perf_counter() - started
        print(
            f"[{number}/{stage_count}] Failed after {elapsed:.1f} s",
            file=sys.stderr,
            flush=True,
        )
        raise
    elapsed = perf_counter() - started
    print(
        f"[{number}/{stage_count}] Complete in {elapsed:.1f} s",
        file=sys.stderr,
        flush=True,
    )


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        partial_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        partial_path.replace(output_path)
    finally:
        partial_path.unlink(missing_ok=True)
    return output_path


def _reproduce(args: argparse.Namespace) -> None:
    started = perf_counter()
    load_env()
    if not os.environ.get("GFW_API_ACCESS_TOKEN"):
        print(
            "Warning: GFW_API_ACCESS_TOKEN is not set; GFW metrics will be omitted.",
            file=sys.stderr,
            flush=True,
        )
    config = load_config(args.config)
    with _stage(1, "Loading source data"):
        ingest_sources(args.db, config)
    with _stage(2, "Loading ERS reports"):
        ingest_ers(args.db, config.ers)
    with _stage(3, "Loading Yagry AIS"):
        ingest_yagry(args.db, config)
    with _stage(4, "Running model"):
        model = run_model(db_path=args.db, config=config)
    with _stage(5, "Calculating metrics"):
        metrics = calculate_metrics(args.db, config)
    with _stage(6, "Saving results"):
        write_publication_metrics(args.db, config, metrics)
        report = publication_data(args.db)
        output_path = _write_report(args.output, report)
    elapsed = perf_counter() - started
    print(
        f"Reproduction complete in {elapsed:.1f} s: {model.run_id}\n"
        f"Report: {output_path}",
        file=sys.stderr,
        flush=True,
    )


def _export_playback(args: argparse.Namespace) -> None:
    started = perf_counter()
    config = load_config(args.config)
    with _stage(1, "Running model", stage_count=2):
        model = run_model(db_path=args.db, config=config, publish_results=False)
    with _stage(2, "Saving playback", stage_count=2):
        report = playback_data(args.db, model.run_id, model.replay)
        output_path = _write_report(args.output, report)
    elapsed = perf_counter() - started
    print(
        f"Playback export complete in {elapsed:.1f} s: {model.run_id}\n"
        f"Report: {output_path}",
        file=sys.stderr,
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="svalbard-cable")
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    result.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    commands = result.add_subparsers(dest="command", required=True)

    reproduce = commands.add_parser("reproduce")
    reproduce.add_argument(
        "--output", type=Path, default=DEFAULT_REPRODUCTION_REPORT_PATH
    )
    reproduce.set_defaults(handler=_reproduce)

    playback = commands.add_parser("export-playback")
    playback.add_argument("--output", type=Path, default=DEFAULT_PLAYBACK_REPORT_PATH)
    playback.set_defaults(handler=_export_playback)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
