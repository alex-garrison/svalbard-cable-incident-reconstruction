"""The configuration defines the model inputs and checks their permitted ranges."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from svalbard_cable_incident_reconstruction.database import PROJECT_ROOT, portable_path


def load_env() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"

# Reject a custom source host.
ALLOWED_AIS_HOST = "kystdatahuset.no"
ALLOWED_CABLE_WMS_HOST = "wms.geonorge.no"


@dataclass(frozen=True)
class ProjectConfig:
    mmsi: int
    cable_id: str
    smooth_start_utc: str
    smooth_end_utc: str
    analysis_start_utc: str
    analysis_end_utc: str
    model_interval_seconds: int


@dataclass(frozen=True)
class AisConfig:
    api_base_url: str
    cache_dir: Path
    selection_start_utc: str
    selection_end_utc: str
    mmsis: tuple[int, ...]
    min_speed_knots: float
    request_delay_seconds: float
    max_retries: int
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class CableConfig:
    wms_url: str
    wms_layer: str
    request_crs: str
    resolution_m: float
    accepted_vector_wkb_sha256: str


@dataclass(frozen=True)
class CableVigilanceConfig:
    mmsi: int
    start_utc: str
    end_utc: str
    maximum_dwell_speed_knots: float
    density_grid_m: float


@dataclass(frozen=True)
class EfsConfig:
    url: str
    sha256: str
    notice_number: int


@dataclass(frozen=True)
class BathymetryConfig:
    path: Path
    opendap_url: str
    west: float
    south: float
    east: float
    north: float
    width: int
    height: int
    array_sha256: str


@dataclass(frozen=True)
class ErsSourceConfig:
    year: int
    url: str
    dca_sha256: str


@dataclass(frozen=True)
class ErsConfig:
    directory: Path
    sources: tuple[ErsSourceConfig, ...]
    request_timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class GearConfig:
    antenna_offset_m: float
    max_warp_length_m: float
    depth_offset_m: float
    clump_along_track_offset_m: float


@dataclass(frozen=True)
class GeometryGridConfig:
    warp_depth_ratios: tuple[float, ...]
    door_spreads_m: tuple[float, ...]


@dataclass(frozen=True)
class FailureIntervalConfig:
    start_utc: str
    end_utc: str


@dataclass(frozen=True)
class TrackSmoothingConfig:
    ais_position_rms_m: float
    max_implied_speed_m_s: float


@dataclass(frozen=True)
class MelkartConfig:
    project: ProjectConfig
    failure_interval: FailureIntervalConfig
    ais: AisConfig
    cable: CableConfig
    cable_vigilance: CableVigilanceConfig
    efs: EfsConfig
    bathymetry: BathymetryConfig
    ers: ErsConfig
    gear: GearConfig
    geometry_grid: GeometryGridConfig
    track_smoothing: TrackSmoothingConfig

    def effective_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # External checks do not define the model run.
        value.pop("cable_vigilance", None)
        value.pop("ers", None)

        for key in ("cache_dir", "request_delay_seconds", "max_retries"):
            value["ais"].pop(key, None)

        for section in ("ais", "cable", "bathymetry"):
            for key, item in list(value[section].items()):
                if isinstance(item, Path):
                    value[section][key] = portable_path(item)

        return value

    def canonical_json(self) -> str:
        return json.dumps(self.effective_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def _path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_ais_config(raw: dict[str, Any]) -> AisConfig:
    bbox_values = raw.get("bbox")
    if bbox_values is not None and len(bbox_values) != 4:
        raise ValueError("ais.bbox must contain west, south, east, north")

    bbox = None
    if bbox_values is not None:
        bbox = (
            float(bbox_values[0]),
            float(bbox_values[1]),
            float(bbox_values[2]),
            float(bbox_values[3]),
        )

    return AisConfig(
        api_base_url=str(raw["api_base_url"]).rstrip("/"),
        cache_dir=_path(raw["cache_dir"]),
        selection_start_utc=str(raw["selection_start_utc"]),
        selection_end_utc=str(raw["selection_end_utc"]),
        mmsis=tuple(int(value) for value in raw.get("mmsis", ())),
        min_speed_knots=float(raw["min_speed_knots"]),
        request_delay_seconds=float(raw["request_delay_seconds"]),
        max_retries=int(raw["max_retries"]),
        bbox=bbox,
    )


def _load_cable_config(raw: dict[str, Any]) -> CableConfig:
    return CableConfig(
        wms_url=str(raw["wms_url"]),
        wms_layer=str(raw["wms_layer"]),
        request_crs=str(raw["request_crs"]),
        resolution_m=float(raw["resolution_m"]),
        accepted_vector_wkb_sha256=str(raw["accepted_vector_wkb_sha256"]),
    )


def _load_cable_vigilance_config(raw: dict[str, Any]) -> CableVigilanceConfig:
    return CableVigilanceConfig(
        mmsi=int(raw["mmsi"]),
        start_utc=str(raw["start_utc"]),
        end_utc=str(raw["end_utc"]),
        maximum_dwell_speed_knots=float(raw["maximum_dwell_speed_knots"]),
        density_grid_m=float(raw["density_grid_m"]),
    )


def _load_bathymetry_config(raw: dict[str, Any]) -> BathymetryConfig:
    return BathymetryConfig(
        path=_path(raw["path"]),
        opendap_url=raw["opendap_url"],
        west=float(raw["west"]),
        south=float(raw["south"]),
        east=float(raw["east"]),
        north=float(raw["north"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        array_sha256=raw["array_sha256"],
    )


def _load_ers_config(raw: dict[str, Any]) -> ErsConfig:
    sources = tuple(
        ErsSourceConfig(
            year=int(source["year"]),
            url=str(source["url"]),
            dca_sha256=str(source["dca_sha256"]),
        )
        for source in raw["sources"]
    )
    return ErsConfig(
        directory=_path(raw["directory"]),
        sources=sources,
        request_timeout_seconds=float(raw["request_timeout_seconds"]),
        max_retries=int(raw["max_retries"]),
    )


def _build_config(raw: dict[str, Any]) -> MelkartConfig:
    return MelkartConfig(
        project=ProjectConfig(**raw["project"]),
        failure_interval=FailureIntervalConfig(**raw["failure_interval"]),
        ais=_load_ais_config(raw["ais"]),
        cable=_load_cable_config(raw["cable"]),
        cable_vigilance=_load_cable_vigilance_config(raw["cable_vigilance"]),
        efs=EfsConfig(**raw["efs"]),
        bathymetry=_load_bathymetry_config(raw["bathymetry"]),
        ers=_load_ers_config(raw["ers"]),
        gear=GearConfig(**raw["gear"]),
        geometry_grid=GeometryGridConfig(
            warp_depth_ratios=tuple(
                float(value) for value in raw["geometry_grid"]["warp_depth_ratios"]
            ),
            door_spreads_m=tuple(
                float(value) for value in raw["geometry_grid"]["door_spreads_m"]
            ),
        ),
        track_smoothing=TrackSmoothingConfig(**raw["track_smoothing"]),
    )


def _validate_times(config: MelkartConfig) -> None:
    timestamp_values = (
        ("project.smooth_start_utc", config.project.smooth_start_utc),
        ("project.smooth_end_utc", config.project.smooth_end_utc),
        ("project.analysis_start_utc", config.project.analysis_start_utc),
        ("project.analysis_end_utc", config.project.analysis_end_utc),
        ("failure_interval.start_utc", config.failure_interval.start_utc),
        ("failure_interval.end_utc", config.failure_interval.end_utc),
        ("ais.selection_start_utc", config.ais.selection_start_utc),
        ("ais.selection_end_utc", config.ais.selection_end_utc),
        ("cable_vigilance.start_utc", config.cable_vigilance.start_utc),
        ("cable_vigilance.end_utc", config.cable_vigilance.end_utc),
    )
    timestamps: dict[str, datetime] = {}

    for name, value in timestamp_values:
        try:
            timestamps[name] = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid {name}: {value!r}") from exc

    failure_duration = (
        timestamps["failure_interval.end_utc"]
        - timestamps["failure_interval.start_utc"]
    )
    if failure_duration != timedelta(minutes=1):
        raise ValueError("failure_interval must span exactly one minute")

    ordered_boundaries = (
        timestamps["ais.selection_start_utc"],
        timestamps["project.smooth_start_utc"],
        timestamps["project.analysis_start_utc"],
        timestamps["failure_interval.start_utc"],
        timestamps["failure_interval.end_utc"],
        timestamps["project.analysis_end_utc"],
        timestamps["project.smooth_end_utc"],
        timestamps["ais.selection_end_utc"],
    )
    if list(ordered_boundaries) != sorted(ordered_boundaries):
        raise ValueError(
            "AIS selection, smoothing, analysis, and failure times are out of order"
        )

    vigilance_start = timestamps["cable_vigilance.start_utc"]
    vigilance_end = timestamps["cable_vigilance.end_utc"]
    if vigilance_start >= vigilance_end:
        raise ValueError("cable_vigilance.end_utc must follow start_utc")

    interval_seconds = config.project.model_interval_seconds
    if interval_seconds <= 0:
        raise ValueError("project.model_interval_seconds must be positive")

    smooth_start = timestamps["project.smooth_start_utc"]
    model_times = (
        timestamps["project.smooth_end_utc"],
        timestamps["project.analysis_start_utc"],
        timestamps["project.analysis_end_utc"],
        timestamps["failure_interval.start_utc"],
        timestamps["failure_interval.end_utc"],
    )

    for model_time in model_times:
        elapsed_seconds = (model_time - smooth_start).total_seconds()
        remainder = elapsed_seconds % interval_seconds

        if remainder:
            raise ValueError(
                "Model times must align with project.model_interval_seconds"
            )


def _validate_ais(config: MelkartConfig) -> None:
    _require(
        config.project.mmsi > 0,
        "project.mmsi must be a positive integer",
    )

    for mmsi in config.ais.mmsis:
        _require(mmsi > 0, "ais.mmsis must contain positive integers")

    ais = config.ais
    _require(bool(ais.mmsis or ais.bbox), "AIS API selection requires mmsis or bbox")
    _require(math.isfinite(ais.min_speed_knots), "ais.min_speed_knots must be finite")
    _require(
        ais.min_speed_knots >= 0,
        "ais.min_speed_knots must be finite and non-negative",
    )
    _require(
        math.isfinite(ais.request_delay_seconds),
        "ais.request_delay_seconds must be finite",
    )
    _require(
        ais.request_delay_seconds >= 0,
        "ais.request_delay_seconds must be finite and non-negative",
    )
    _require(ais.max_retries >= 0, "ais.max_retries must be non-negative")

    ais_host = urlparse(ais.api_base_url).hostname
    _require(
        ais_host == ALLOWED_AIS_HOST,
        (f"ais.api_base_url must be an {ALLOWED_AIS_HOST} URL, got {ais_host!r}"),
    )


def _validate_sources(config: MelkartConfig) -> None:
    cable = config.cable
    _require(
        urlparse(cable.wms_url).hostname == ALLOWED_CABLE_WMS_HOST,
        f"cable.wms_url must use {ALLOWED_CABLE_WMS_HOST}",
    )
    _require(cable.resolution_m > 0, "Invalid cable WMS resolution")
    _require(cable.request_crs == "EPSG:25833", "cable.request_crs must be EPSG:25833")
    _require(
        len(cable.accepted_vector_wkb_sha256) == 64,
        "cable.accepted_vector_wkb_sha256 must be a SHA-256 value",
    )
    _require(len(config.efs.sha256) == 64, "efs.sha256 must be a SHA-256 value")
    _require(config.efs.notice_number > 0, "efs.notice_number must be positive")

    bathymetry = config.bathymetry
    _require(
        bathymetry.west < bathymetry.east,
        "bathymetry.west must be less than bathymetry.east",
    )
    _require(
        bathymetry.south < bathymetry.north,
        "bathymetry.south must be less than bathymetry.north",
    )
    _require(bathymetry.width > 0, "bathymetry.width must be positive")
    _require(bathymetry.height > 0, "bathymetry.height must be positive")

    _validate_ers(config.ers)


def _validate_sha256(name: str, value: str) -> None:
    is_sha256 = len(value) == 64 and set(value) <= set("0123456789abcdef")
    _require(is_sha256, f"{name} must be a lowercase SHA-256 value")


def _validate_ers(config: ErsConfig) -> None:
    _require(bool(config.sources), "ers.sources cannot be empty")
    years = [source.year for source in config.sources]
    _require(len(years) == len(set(years)), "ers.sources cannot repeat a year")
    _require(
        config.request_timeout_seconds > 0,
        "ers.request_timeout_seconds must be positive",
    )
    _require(config.max_retries >= 0, "ers.max_retries must be non-negative")

    for source in config.sources:
        parsed_url = urlparse(source.url)
        _require(
            parsed_url.scheme == "https"
            and parsed_url.hostname == "register.fiskeridir.no",
            f"ers source {source.year} must use the Fiskeridirektoratet HTTPS host",
        )
        _validate_sha256(f"ers source {source.year} dca_sha256", source.dca_sha256)


def _validate_model(config: MelkartConfig) -> None:
    if config.track_smoothing.ais_position_rms_m <= 0:
        raise ValueError("track_smoothing.ais_position_rms_m must be positive")
    if config.track_smoothing.max_implied_speed_m_s <= 0:
        raise ValueError("track_smoothing.max_implied_speed_m_s must be positive")
    if config.gear.antenna_offset_m < 0:
        raise ValueError("gear.antenna_offset_m cannot be negative")

    if config.gear.max_warp_length_m <= 0:
        raise ValueError("gear.max_warp_length_m must be positive")

    vigilance = config.cable_vigilance
    if vigilance.mmsi <= 0:
        raise ValueError("cable_vigilance.mmsi must be positive")
    if vigilance.maximum_dwell_speed_knots < 0:
        raise ValueError("cable_vigilance.maximum_dwell_speed_knots cannot be negative")

    if vigilance.density_grid_m <= 0:
        raise ValueError("cable_vigilance.density_grid_m must be positive")

    _validate_geometry_grid(config.geometry_grid)


def _validate_geometry_grid(geometry: GeometryGridConfig) -> None:
    for name, values in (
        ("warp_depth_ratios", geometry.warp_depth_ratios),
        ("door_spreads_m", geometry.door_spreads_m),
    ):
        if not values:
            raise ValueError(f"geometry_grid.{name} cannot be empty")

        for value in values:
            if value <= 0:
                raise ValueError(f"geometry_grid.{name} must contain positive values")

        if len(values) != len(set(values)):
            raise ValueError(f"geometry_grid.{name} cannot contain duplicates")

    for ratio in geometry.warp_depth_ratios:
        if ratio <= 1:
            raise ValueError("geometry_grid.warp_depth_ratios must be greater than 1")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> MelkartConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    config = _build_config(raw)
    _validate_times(config)
    _validate_ais(config)
    _validate_sources(config)
    _validate_model(config)

    return config
