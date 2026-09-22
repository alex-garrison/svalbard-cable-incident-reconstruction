"""The GEBCO grid supplies a local slope estimate at the modelled crossing centroid."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import rasterio
from pyproj import Geod
from rasterio.io import DatasetReader
from rasterio.windows import Window

from svalbard_cable_incident_reconstruction.config import BathymetryConfig
from svalbard_cable_incident_reconstruction.sources import validate_bathymetry


def _horn_slope(
    elevation: np.ndarray, x_resolution: float, y_resolution: float
) -> np.ndarray:
    eastern_elevations = (
        elevation[:-2, 2:] + 2 * elevation[1:-1, 2:] + elevation[2:, 2:]
    )
    western_elevations = (
        elevation[:-2, :-2] + 2 * elevation[1:-1, :-2] + elevation[2:, :-2]
    )
    southern_elevations = (
        elevation[2:, :-2] + 2 * elevation[2:, 1:-1] + elevation[2:, 2:]
    )
    northern_elevations = (
        elevation[:-2, :-2] + 2 * elevation[:-2, 1:-1] + elevation[:-2, 2:]
    )

    dzdx = (eastern_elevations - western_elevations) / (8 * x_resolution)
    dzdy = (southern_elevations - northern_elevations) / (8 * y_resolution)

    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def _slope_neighbourhood(
    raster: DatasetReader,
    longitude: float,
    latitude: float,
    radius_m: float,
) -> tuple[float, np.ndarray]:
    row, column = raster.index(longitude, latitude)
    cell_longitude, cell_latitude = raster.xy(row, column)
    geod = Geod(ellps="WGS84")
    _, _, x_resolution = geod.inv(
        cell_longitude,
        cell_latitude,
        cell_longitude + abs(float(raster.transform.a)),
        cell_latitude,
    )
    _, _, y_resolution = geod.inv(
        cell_longitude,
        cell_latitude,
        cell_longitude,
        cell_latitude + abs(float(raster.transform.e)),
    )
    radius_cells = max(1, math.ceil(radius_m / min(x_resolution, y_resolution)))
    padding = radius_cells + 1
    window_size = 2 * padding + 1

    window_constructor: Any = Window
    elevations = raster.read(
        1,
        window=window_constructor(
            column - padding,
            row - padding,
            window_size,
            window_size,
        ),
        boundless=True,
        masked=True,
    ).astype(float)
    slopes = _horn_slope(
        elevations.filled(np.nan),
        x_resolution,
        y_resolution,
    )

    # The Horn calculation removes one cell from each edge of the input window.
    centre_index = padding - 1
    row_indexes, column_indexes = np.ogrid[: slopes.shape[0], : slopes.shape[1]]
    x_distance = (column_indexes - centre_index) * x_resolution
    y_distance = (row_indexes - centre_index) * y_resolution
    distance = np.hypot(x_distance, y_distance)
    local_slopes = slopes[(distance <= radius_m) & np.isfinite(slopes)]

    if not len(local_slopes):
        raise RuntimeError(
            f"No valid GEBCO slope cells in the {radius_m:g} m neighbourhood"
        )

    return float(slopes[centre_index, centre_index]), local_slopes


def calculate_slope_metrics(
    longitude: float, latitude: float, config: BathymetryConfig
) -> dict[str, Any]:
    raster_hash = validate_bathymetry(config.path, config)

    radius_m = 250.0
    with rasterio.open(config.path) as raster:
        centre_slope, local_slopes = _slope_neighbourhood(
            raster, longitude, latitude, radius_m
        )

    return {
        "slope.centroid_cell_slope_degrees": centre_slope,
        "slope.neighbourhood_median_slope_degrees": float(np.median(local_slopes)),
        "slope.neighbourhood_min_slope_degrees": float(np.min(local_slopes)),
        "slope.neighbourhood_max_slope_degrees": float(np.max(local_slopes)),
        "slope.neighbourhood_valid_cell_count": len(local_slopes),
        "slope.neighbourhood_radius_m": radius_m,
        "slope.raster_sha256": raster_hash,
    }
