#!/usr/bin/env python3
"""Tests for solution validation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.models import Stop, Route, Solution, DistanceMatrix
from src.utils import validate_solution, compute_route_time


def make_test_instance():
    """Create a 4-stop test instance."""
    stops = [
        Stop(id=0, name="Depot", lat=0.0, lon=0.0, demand=0),
        Stop(id=1, name="S1", lat=1.0, lon=0.0, demand=5),
        Stop(id=2, name="S2", lat=0.0, lon=1.0, demand=3),
        Stop(id=3, name="S3", lat=1.0, lon=1.0, demand=7),
        Stop(id=4, name="S4", lat=2.0, lon=0.0, demand=4),
    ]
    n = len(stops)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = stops[i].lat - stops[j].lat
            dy = stops[i].lon - stops[j].lon
            matrix[i, j] = (dx**2 + dy**2) ** 0.5 * 100
    
    dm = DistanceMatrix(matrix=matrix, stop_ids=[s.id for s in stops], source="test")
    return stops, dm


class TestValidSolution:
    def test_valid_solution_passes(self):
        """A correct solution should validate with no errors."""
        stops, dm = make_test_instance()
        route1 = Route(
            vehicle_id=1, stop_ids=[1, 4],
            total_time=compute_route_time([1, 4], dm),
            total_demand=9,
        )
        route2 = Route(
            vehicle_id=2, stop_ids=[2, 3],
            total_time=compute_route_time([2, 3], dm),
            total_demand=10,
        )
        sol = Solution(routes=[route1, route2], method="test")
        sol.recompute_total_time()
        
        errors = validate_solution(sol, stops, dm, num_vehicles=3, vehicle_capacity=20)
        assert errors == []


class TestMissingStops:
    def test_catches_missing_stop(self):
        """Should catch when a stop is not visited."""
        stops, dm = make_test_instance()
        route1 = Route(
            vehicle_id=1, stop_ids=[1, 4],
            total_time=compute_route_time([1, 4], dm),
            total_demand=9,
        )
        # Missing stops 2 and 3
        sol = Solution(routes=[route1], method="test")
        sol.recompute_total_time()
        
        errors = validate_solution(sol, stops, dm, num_vehicles=3, vehicle_capacity=20)
        assert any("Missing" in e for e in errors)


class TestDuplicateStops:
    def test_catches_duplicate_stop(self):
        """Should catch when a stop is visited more than once."""
        stops, dm = make_test_instance()
        route1 = Route(
            vehicle_id=1, stop_ids=[1, 2],
            total_time=compute_route_time([1, 2], dm),
            total_demand=8,
        )
        route2 = Route(
            vehicle_id=2, stop_ids=[2, 3, 4],  # stop 2 also in route1
            total_time=compute_route_time([2, 3, 4], dm),
            total_demand=14,
        )
        sol = Solution(routes=[route1, route2], method="test")
        sol.recompute_total_time()
        
        errors = validate_solution(sol, stops, dm, num_vehicles=3, vehicle_capacity=20)
        assert any("Duplicate" in e for e in errors)


class TestCapacityViolation:
    def test_catches_over_capacity(self):
        """Should catch when a route exceeds vehicle capacity."""
        stops, dm = make_test_instance()
        # All stops on one route: demand = 5+3+7+4 = 19, capacity = 10
        route1 = Route(
            vehicle_id=1, stop_ids=[1, 2, 3, 4],
            total_time=compute_route_time([1, 2, 3, 4], dm),
            total_demand=19,
        )
        sol = Solution(routes=[route1], method="test")
        sol.recompute_total_time()
        
        errors = validate_solution(sol, stops, dm, num_vehicles=3, vehicle_capacity=10)
        assert any("capacity" in e.lower() for e in errors)


class TestVehicleCount:
    def test_catches_too_many_routes(self):
        """Should catch when more routes than vehicles."""
        stops, dm = make_test_instance()
        routes = []
        for i, s in enumerate(stops[1:], start=1):
            routes.append(Route(
                vehicle_id=i, stop_ids=[s.id],
                total_time=compute_route_time([s.id], dm),
                total_demand=s.demand,
            ))
        sol = Solution(routes=routes, method="test")
        sol.recompute_total_time()
        
        # 4 routes but only 2 vehicles allowed
        errors = validate_solution(sol, stops, dm, num_vehicles=2, vehicle_capacity=100)
        assert any("Too many" in e for e in errors)


class TestTimeMismatch:
    def test_catches_wrong_route_time(self):
        """Should catch when stored route time doesn't match matrix."""
        stops, dm = make_test_instance()
        route1 = Route(
            vehicle_id=1, stop_ids=[1, 2, 3, 4],
            total_time=999999.0,  # Wrong time
            total_demand=19,
        )
        sol = Solution(routes=[route1], method="test")
        sol.recompute_total_time()
        
        errors = validate_solution(sol, stops, dm, num_vehicles=3, vehicle_capacity=100)
        assert any("time mismatch" in e.lower() for e in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
