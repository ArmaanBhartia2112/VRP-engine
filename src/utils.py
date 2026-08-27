"""Utility functions: haversine distance, route time computation, plotting helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.models import DistanceMatrix, Route, Solution, Stop


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in kilometres."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def haversine_time_seconds(
    lat1: float, lon1: float, lat2: float, lon2: float,
    speed_kmh: float = 40.0,
) -> float:
    """Estimated driving time (seconds) from haversine distance.

    Uses a fixed average urban speed (default 40 km/h) — this is a rough
    approximation used only as a baseline comparison against real road data.
    """
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    return (dist_km / speed_kmh) * 3600.0


# ---------------------------------------------------------------------------
# Route time computation
# ---------------------------------------------------------------------------

def compute_route_time(route_stop_ids: list[int], dm: "DistanceMatrix") -> float:
    """Compute total travel time (seconds) for a route including depot legs.

    Args:
        route_stop_ids: Ordered stop IDs to visit (excludes depot bookends).
        dm: The distance matrix to look up times.

    Returns:
        Total time: depot → first_stop → ... → last_stop → depot.
    """
    if not route_stop_ids:
        return 0.0
    full_path = [0] + list(route_stop_ids) + [0]
    total = 0.0
    for i in range(len(full_path) - 1):
        total += dm.time(full_path[i], full_path[i + 1])
    return total


def compute_route_demand(route_stop_ids: list[int], stops: list["Stop"]) -> int:
    """Sum the demand of all stops on a route."""
    stop_map = {s.id: s for s in stops}
    return sum(stop_map[sid].demand for sid in route_stop_ids)


# ---------------------------------------------------------------------------
# Solution validation
# ---------------------------------------------------------------------------

def validate_solution(
    solution: "Solution",
    stops: list["Stop"],
    dm: "DistanceMatrix",
    num_vehicles: int,
    vehicle_capacity: int,
) -> list[str]:
    """Validate a VRP solution. Returns a list of error strings (empty = valid).

    Checks:
    - Every non-depot stop appears in exactly one route
    - No route exceeds vehicle capacity
    - Route times match the distance matrix (within 1s tolerance)
    - Number of routes ≤ num_vehicles
    """
    errors: list[str] = []

    # Check vehicle count
    if len(solution.routes) > num_vehicles:
        errors.append(
            f"Too many routes: {len(solution.routes)} > {num_vehicles} vehicles"
        )

    # Collect all visited stops
    all_visited: list[int] = []
    non_depot_ids = {s.id for s in stops if s.id != 0}

    for route in solution.routes:
        # Check capacity
        demand = compute_route_demand(route.stop_ids, stops)
        if demand > vehicle_capacity:
            errors.append(
                f"Route v{route.vehicle_id} over capacity: "
                f"{demand} > {vehicle_capacity}"
            )

        # Verify demand matches stored value
        if route.total_demand != demand:
            errors.append(
                f"Route v{route.vehicle_id} demand mismatch: "
                f"stored={route.total_demand}, computed={demand}"
            )

        # Verify route time
        expected_time = compute_route_time(route.stop_ids, dm)
        if abs(route.total_time - expected_time) > 1.0:
            errors.append(
                f"Route v{route.vehicle_id} time mismatch: "
                f"stored={route.total_time:.1f}, computed={expected_time:.1f}"
            )

        all_visited.extend(route.stop_ids)

    # Check all stops visited exactly once
    visited_set = set(all_visited)
    missing = non_depot_ids - visited_set
    if missing:
        errors.append(f"Missing stops: {missing}")

    duplicates = [s for s in all_visited if all_visited.count(s) > 1]
    if duplicates:
        errors.append(f"Duplicate stops: {set(duplicates)}")

    extra = visited_set - non_depot_ids
    if extra:
        errors.append(f"Unknown stops visited: {extra}")

    return errors


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_matrix_heatmap(
    dm: "DistanceMatrix",
    stop_names: list[str] | None = None,
    title: str = "Travel Time Matrix",
    save_path: str | None = None,
) -> None:
    """Plot a heatmap of a distance matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    n = dm.matrix.shape[0]
    # Convert to minutes for readability
    matrix_min = dm.matrix / 60.0
    im = ax.imshow(matrix_min, cmap="YlOrRd", aspect="equal")

    labels = stop_names or [str(i) for i in dm.stop_ids]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix_min[i, j]
            color = "white" if val > matrix_min.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6, color=color)

    ax.set_title(f"{title}\n(values in minutes)", fontsize=12)
    plt.colorbar(im, ax=ax, label="Minutes", shrink=0.8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved heatmap to {save_path}")
    plt.close(fig)


def plot_solution_routes(
    solution: "Solution",
    stops: list["Stop"],
    title: str = "VRP Solution",
    save_path: str | None = None,
) -> None:
    """Plot vehicle routes on a lat/lon scatter plot."""
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot depot
    depot = stops[0]
    ax.scatter(depot.lon, depot.lat, c="black", s=200, marker="s",
               zorder=5, label="Depot")
    ax.annotate("DEPOT", (depot.lon, depot.lat), fontsize=8,
                fontweight="bold", ha="center", va="bottom",
                xytext=(0, 10), textcoords="offset points")

    for idx, route in enumerate(solution.routes):
        color = colors[idx % len(colors)]
        path_ids = route.full_path
        lats = [next(s.lat for s in stops if s.id == sid) for sid in path_ids]
        lons = [next(s.lon for s in stops if s.id == sid) for sid in path_ids]

        ax.plot(lons, lats, "-o", color=color, linewidth=1.5, markersize=5,
                label=f"Vehicle {route.vehicle_id} ({route.total_time/60:.0f} min)")

        # Label stops
        for sid in route.stop_ids:
            s = next(st for st in stops if st.id == sid)
            ax.annotate(f"{s.id}", (s.lon, s.lat), fontsize=7,
                        ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{title}\nTotal: {solution.total_time/60:.1f} min | "
                 f"Method: {solution.method}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved route plot to {save_path}")
    plt.close(fig)
