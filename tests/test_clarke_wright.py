#!/usr/bin/env python3
"""Tests for the Clarke-Wright savings algorithm."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.models import Stop, DistanceMatrix
from src.clarke_wright import clarke_wright
from src.utils import validate_solution


def make_stops_and_matrix(n_stops, demands=None):
    """Create a small test instance with known distances.
    
    Stops are laid out in a line: depot at 0, stop 1 at 1km, stop 2 at 2km, etc.
    Travel time = distance in seconds (for simplicity).
    """
    stops = [Stop(id=0, name="Depot", lat=0.0, lon=0.0, demand=0)]
    for i in range(1, n_stops + 1):
        d = demands[i - 1] if demands else 1
        stops.append(Stop(id=i, name=f"S{i}", lat=0.0, lon=float(i), demand=d))
    
    n = len(stops)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = abs(stops[i].lon - stops[j].lon) * 100  # 100s per unit
    
    dm = DistanceMatrix(
        matrix=matrix,
        stop_ids=[s.id for s in stops],
        source="test",
    )
    return stops, dm


class TestClarkeWrightBasic:
    def test_single_stop(self):
        """Single stop should produce one route."""
        stops, dm = make_stops_and_matrix(1)
        sol = clarke_wright(stops, dm, num_vehicles=3, vehicle_capacity=100)
        assert len(sol.routes) == 1
        assert sol.routes[0].stop_ids == [1]

    def test_all_stops_visited(self):
        """All stops must appear exactly once across all routes."""
        stops, dm = make_stops_and_matrix(6)
        sol = clarke_wright(stops, dm, num_vehicles=3, vehicle_capacity=100)
        all_visited = []
        for r in sol.routes:
            all_visited.extend(r.stop_ids)
        assert sorted(all_visited) == [1, 2, 3, 4, 5, 6]

    def test_validation_passes(self):
        """The solution should pass full validation."""
        stops, dm = make_stops_and_matrix(6)
        sol = clarke_wright(stops, dm, num_vehicles=3, vehicle_capacity=100)
        errors = validate_solution(sol, stops, dm, num_vehicles=3, vehicle_capacity=100)
        assert errors == [], f"Validation errors: {errors}"

    def test_vehicle_count_respected(self):
        """Should not produce more routes than vehicles."""
        stops, dm = make_stops_and_matrix(8)
        sol = clarke_wright(stops, dm, num_vehicles=2, vehicle_capacity=100)
        assert len(sol.routes) <= 2


class TestClarkeWrightCapacity:
    def test_capacity_respected(self):
        """No route should exceed vehicle capacity."""
        # Each stop has demand 5, capacity 12 → at most 2 stops per route
        stops, dm = make_stops_and_matrix(6, demands=[5, 5, 5, 5, 5, 5])
        sol = clarke_wright(stops, dm, num_vehicles=4, vehicle_capacity=12)
        for r in sol.routes:
            assert r.total_demand <= 12, f"Route {r.vehicle_id} over capacity: {r.total_demand}"

    def test_tight_capacity_forces_split(self):
        """With very tight capacity, stops can't all merge into one route."""
        # 4 stops, demand 10 each, capacity 15 → each route holds at most 1 stop
        stops, dm = make_stops_and_matrix(4, demands=[10, 10, 10, 10])
        sol = clarke_wright(stops, dm, num_vehicles=4, vehicle_capacity=15)
        assert len(sol.routes) == 4
        for r in sol.routes:
            assert len(r.stop_ids) == 1

    def test_validation_passes_with_capacity(self):
        """Full validation with capacity constraints."""
        stops, dm = make_stops_and_matrix(8, demands=[3, 7, 2, 8, 4, 6, 1, 5])
        sol = clarke_wright(stops, dm, num_vehicles=3, vehicle_capacity=20)
        errors = validate_solution(sol, stops, dm, num_vehicles=3, vehicle_capacity=20)
        assert errors == [], f"Validation errors: {errors}"


class TestClarkeWrightSavings:
    def test_merges_nearby_stops(self):
        """Stops close together should be merged into the same route."""
        # 4 stops: 1,2 near each other (lon=1.0, 1.1), 3,4 near each other (lon=5.0, 5.1)
        stops = [
            Stop(id=0, name="Depot", lat=0.0, lon=0.0, demand=0),
            Stop(id=1, name="S1", lat=0.0, lon=1.0, demand=1),
            Stop(id=2, name="S2", lat=0.0, lon=1.1, demand=1),
            Stop(id=3, name="S3", lat=0.0, lon=5.0, demand=1),
            Stop(id=4, name="S4", lat=0.0, lon=5.1, demand=1),
        ]
        n = len(stops)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = abs(stops[i].lon - stops[j].lon) * 100
        dm = DistanceMatrix(matrix=matrix, stop_ids=[s.id for s in stops], source="test")
        
        sol = clarke_wright(stops, dm, num_vehicles=2, vehicle_capacity=100)
        
        # Expect 1&2 together, 3&4 together (greatest savings)
        route_sets = [set(r.stop_ids) for r in sol.routes]
        assert {1, 2} in route_sets or any({1, 2}.issubset(s) for s in route_sets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
