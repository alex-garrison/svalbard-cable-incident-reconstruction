"""The two-dimensional model estimates gear paths and horizontal cable crossings."""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_splprep
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry


class ComponentType(enum.Enum):
    PORT_DOOR = "port_door"
    STARBOARD_DOOR = "starboard_door"
    CENTRE_CLUMP = "centre_clump"


@dataclass(frozen=True)
class GearParameters:
    scenario_id: str
    antenna_offset_m: float = 25.0
    warp_depth_ratio: float = 3.0
    max_warp_length_m: float = 3000.0
    depth_offset_m: float = 0.0
    door_spread_m: float = 1000.0
    clump_along_track_offset_m: float = 0.0


@dataclass
class SmoothedTrack:
    time_s: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    vx: NDArray[np.float64]
    vy: NDArray[np.float64]


@dataclass
class ComponentState:
    component: ComponentType
    time_s: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]


@dataclass(frozen=True)
class HorizontalCrossing:
    component: ComponentType
    crossing_time_s: float
    crossing_x: float
    crossing_y: float


@dataclass
class GeometryResult:
    params: GearParameters
    component_states: list[ComponentState]
    rejected: bool = False
    rejection_reason: str = ""


@dataclass
class _GearCentreSolution:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    direction_x: NDArray[np.float64]
    direction_y: NDArray[np.float64]
    depths: NDArray[np.float64]


DepthFunction = Callable[[float, float], float | None]


def smooth_ais_track(
    times_s: NDArray[np.float64],
    x_m: NDArray[np.float64],
    y_m: NDArray[np.float64],
    *,
    eval_start_s: float,
    eval_end_s: float,
    interval_s: int,
    ais_position_rms_m: float,
) -> SmoothedTrack:
    """A cubic spline estimates the vessel's position and direction at fixed intervals."""
    mask = (times_s >= eval_start_s) & (times_s <= eval_end_s)
    times = times_s[mask]
    positions = np.vstack([x_m[mask], y_m[mask]])
    if len(times) < 4:
        raise RuntimeError(f"Need at least 4 AIS positions, got {len(times)}")
    if np.any(np.diff(times) <= 0):
        raise RuntimeError("AIS representative times must be strictly increasing")

    spline, _ = make_splprep(
        positions,
        u=times,
        k=3,
        s=len(times) * ais_position_rms_m**2,
    )
    evaluated_times = np.arange(eval_start_s, eval_end_s + interval_s, interval_s)
    evaluated = np.asarray(spline(evaluated_times), dtype=np.float64)
    velocity = np.asarray(spline.derivative()(evaluated_times), dtype=np.float64)

    return SmoothedTrack(
        time_s=evaluated_times,
        x=evaluated[0],
        y=evaluated[1],
        vx=velocity[0],
        vy=velocity[1],
    )


def estimate_towing_point(
    track: SmoothedTrack,
    antenna_offset_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The towing point is placed aft of the AIS antenna along the fitted path."""
    speed = np.hypot(track.vx, track.vy)
    if np.any(speed < 0.01):
        raise RuntimeError("Smoothed AIS path is stationary")

    aft_offset_x = track.vx / speed * antenna_offset_m
    aft_offset_y = track.vy / speed * antenna_offset_m
    towing_point_x = track.x - aft_offset_x
    towing_point_y = track.y - aft_offset_y

    return towing_point_x, towing_point_y


def _sample_depths(
    x_m: NDArray[np.float64],
    y_m: NDArray[np.float64],
    depth: DepthFunction,
) -> NDArray[np.float64]:
    values = np.asarray(
        [depth(float(x), float(y)) for x, y in zip(x_m, y_m, strict=True)],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise RuntimeError(
            "GEBCO depth is missing or invalid at the assumed gear centre"
        )
    return values


def _solve_gear_centre(
    current_x: NDArray[np.float64],
    current_y: NDArray[np.float64],
    current_chainage: NDArray[np.float64],
    chainage: NDArray[np.float64],
    tow_x: NDArray[np.float64],
    tow_y: NDArray[np.float64],
    depth: DepthFunction,
    params: GearParameters,
) -> _GearCentreSolution:
    """Water depth and warp length set the gear's lag along the earlier tow path."""
    depths = _sample_depths(current_x, current_y, depth) + params.depth_offset_m
    half_spread = params.door_spread_m / 2.0
    warp = params.warp_depth_ratio * depths

    if np.any(warp > params.max_warp_length_m):
        raise _RejectedGeometry("warp ceiling")

    # Warp, depth, half-spread, and horizontal lag form a right triangle.
    lag_squared = warp**2 - depths**2 - half_spread**2
    if np.any(lag_squared <= 0):
        raise _RejectedGeometry("door spread exceeds horizontal warp reach")

    lag = np.sqrt(lag_squared)
    earlier_chainage = current_chainage - lag
    if np.any(earlier_chainage < chainage[0]):
        raise RuntimeError("AIS smoothing window has insufficient earlier tow path")

    earlier_x = np.interp(earlier_chainage, chainage, tow_x)
    earlier_y = np.interp(earlier_chainage, chainage, tow_y)

    # The chord from the earlier path point lets the gear cut across a vessel turn.
    direction_x = current_x - earlier_x
    direction_y = current_y - earlier_y
    chord = np.hypot(direction_x, direction_y)

    if np.any(chord < 1.0):
        raise RuntimeError("Recent towing path does not define a trailing direction")

    direction_x /= chord
    direction_y /= chord
    centre_x = current_x - lag * direction_x
    centre_y = current_y - lag * direction_y
    return _GearCentreSolution(
        x=centre_x,
        y=centre_y,
        direction_x=direction_x,
        direction_y=direction_y,
        depths=depths,
    )


class _RejectedGeometry(Exception):
    pass


def _point_parts(geometry: BaseGeometry) -> list[Point]:
    if isinstance(geometry, Point):
        return [geometry]

    if geometry.geom_type in {"MultiPoint", "GeometryCollection"}:
        points = []
        for part in geometry.geoms:
            points.extend(_point_parts(part))
        return points

    return []


def _validate_gear_geometry(
    current_x: NDArray[np.float64],
    current_y: NDArray[np.float64],
    depths: NDArray[np.float64],
    port_x: NDArray[np.float64],
    port_y: NDArray[np.float64],
    starboard_x: NDArray[np.float64],
    starboard_y: NDArray[np.float64],
    params: GearParameters,
) -> None:
    for door_x, door_y in ((port_x, port_y), (starboard_x, starboard_y)):
        solved_warp = np.hypot(np.hypot(door_x - current_x, door_y - current_y), depths)
        if not np.allclose(solved_warp, params.warp_depth_ratio * depths, atol=1e-6):
            raise RuntimeError("Accepted door does not satisfy the warp equation")
    if not np.allclose(
        np.hypot(port_x - starboard_x, port_y - starboard_y),
        params.door_spread_m,
        atol=1e-6,
    ):
        raise RuntimeError("Accepted door separation does not match the scenario")


def _component_states(
    centre: _GearCentreSolution,
    times: NDArray[np.float64],
    params: GearParameters,
) -> list[ComponentState]:
    half_spread = params.door_spread_m / 2.0
    normal_x = -centre.direction_y
    normal_y = centre.direction_x

    return [
        ComponentState(
            component=ComponentType.PORT_DOOR,
            time_s=times,
            x=centre.x + half_spread * normal_x,
            y=centre.y + half_spread * normal_y,
        ),
        ComponentState(
            component=ComponentType.STARBOARD_DOOR,
            time_s=times,
            x=centre.x - half_spread * normal_x,
            y=centre.y - half_spread * normal_y,
        ),
        ComponentState(
            component=ComponentType.CENTRE_CLUMP,
            time_s=times,
            x=centre.x + params.clump_along_track_offset_m * centre.direction_x,
            y=centre.y + params.clump_along_track_offset_m * centre.direction_y,
        ),
    ]


def solve_path_lag_geometry(
    track: SmoothedTrack,
    tow_x: NDArray[np.float64],
    tow_y: NDArray[np.float64],
    depth: DepthFunction,
    params: GearParameters,
    *,
    analysis_start_s: float,
    analysis_end_s: float,
) -> GeometryResult:
    """The two doors and centre clump are placed behind the towing point."""
    step_distance = np.hypot(np.diff(tow_x), np.diff(tow_y))
    if np.any(step_distance <= 0):
        raise RuntimeError("Smoothed towing path must move at every time step")
    chainage = np.concatenate(([0.0], np.cumsum(step_distance)))
    used = (track.time_s >= analysis_start_s) & (track.time_s <= analysis_end_s)
    times = track.time_s[used]
    current_x = tow_x[used]
    current_y = tow_y[used]
    current_chainage = chainage[used]

    try:
        centre = _solve_gear_centre(
            current_x,
            current_y,
            current_chainage,
            chainage,
            tow_x,
            tow_y,
            depth,
            params,
        )
    except _RejectedGeometry as rejection:
        return GeometryResult(
            params=params,
            component_states=[],
            rejected=True,
            rejection_reason=str(rejection),
        )

    states = _component_states(centre, times, params)
    port_door, starboard_door, _centre_clump = states

    _validate_gear_geometry(
        current_x,
        current_y,
        centre.depths,
        port_door.x,
        port_door.y,
        starboard_door.x,
        starboard_door.y,
        params,
    )

    return GeometryResult(params=params, component_states=states)


def _crossings_for_segment(
    state: ComponentState,
    index: int,
    cable_line: LineString,
    clip_start: float,
    clip_end: float,
    interval_start_s: float,
    interval_end_s: float,
) -> list[HorizontalCrossing]:
    sample_start = float(state.time_s[index])
    sample_end = float(state.time_s[index + 1])
    sample_duration = max(sample_end - sample_start, np.finfo(np.float64).eps)

    clip_start_fraction = (clip_start - sample_start) / sample_duration
    clip_end_fraction = (clip_end - sample_start) / sample_duration

    sample_delta_x = state.x[index + 1] - state.x[index]
    sample_delta_y = state.y[index + 1] - state.y[index]

    segment_start_x = state.x[index] + clip_start_fraction * sample_delta_x
    segment_start_y = state.y[index] + clip_start_fraction * sample_delta_y
    segment_end_x = state.x[index] + clip_end_fraction * sample_delta_x
    segment_end_y = state.y[index] + clip_end_fraction * sample_delta_y

    segment = LineString(
        [(segment_start_x, segment_start_y), (segment_end_x, segment_end_y)]
    )
    if segment.length == 0 or not segment.intersects(cable_line):
        return []

    points = _point_parts(segment.intersection(cable_line))

    crossings: list[HorizontalCrossing] = []
    for point in points:
        # Distance along the segment gives the crossing time by linear interpolation.
        distance_fraction = segment.project(point) / segment.length
        clipped_duration = clip_end - clip_start
        crossing_time = clip_start + distance_fraction * clipped_duration

        if not interval_start_s <= crossing_time < interval_end_s:
            continue

        crossings.append(
            HorizontalCrossing(
                component=state.component,
                crossing_time_s=crossing_time,
                crossing_x=float(point.x),
                crossing_y=float(point.y),
            )
        )

    return crossings


def measure_horizontal_crossings(
    states: list[ComponentState],
    cable_line: LineString,
    *,
    interval_start_s: float,
    interval_end_s: float,
) -> list[HorizontalCrossing]:
    """Only horizontal intersections within the half-open interval are retained."""
    if interval_end_s <= interval_start_s:
        raise ValueError("Crossing interval end must be after its start")

    results: list[HorizontalCrossing] = []
    for state in states:
        previous_time: float | None = None
        for index in range(len(state.time_s) - 1):
            sample_start = float(state.time_s[index])
            sample_end = float(state.time_s[index + 1])
            clip_start = max(sample_start, interval_start_s)
            clip_end = min(sample_end, interval_end_s)

            if clip_end <= clip_start:
                continue

            for crossing in _crossings_for_segment(
                state,
                index,
                cable_line,
                clip_start,
                clip_end,
                interval_start_s,
                interval_end_s,
            ):
                if (
                    previous_time is not None
                    and abs(crossing.crossing_time_s - previous_time) < 1e-6
                ):
                    continue
                results.append(crossing)
                previous_time = crossing.crossing_time_s

    return results
