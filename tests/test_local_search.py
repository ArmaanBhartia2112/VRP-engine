#!/usr/bin/env python3
"""Tests for local search operators (2-opt and or-opt)."""

import sys
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.models import Stop, Route, Solution, DistanceMatrix
from src.local_search import two_opt_pass, or_opt_pass, run_local_search
from src.utils import compute_route_time


def make_stops_and_matrix():
    """Create a test instance where 2-opt can improve a crossing route.
    
    Layout (depot at origin, 4 stops forming a square):
        2---3
        |   |
        1---4
        |
        0 (depot)
    
    A route visiting 1→3→2→4 has a crossing; 2-opt should fix to 1→2→3→4.
    """
    stops = [
        Stop(id=0, name="Depot", lat=0.0, lon=0.0, demand=0),
        Stop(id=1, name="S1", lat=1.0, lon=0.0, demand=2),
        Stop(id=2, name="S2", lat=2.0, lon=0.0, demand=2),
        Stop(id=3, name="S3", lat=2.0, lon=1.0, demand=2),
        Stop(id=4, name="S4", lat=1.0, lon=1.0, demand=2),
    ]
    # Euclidean distances as travel times (in seconds)
    n = len(stops)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = stops[i].lat - stops[j].lat
            dy = stops[i].lon - stops[j].lon
            matrix[i, j] = (dx**2 + dy**2) ** 0.5 * 100  # *100 for larger numbers
    
    dm = DistanceMatrix(matrix=matrix, stop_ids=[s.id for s in stops], source="test")
    return stops, dm


class TestTwoOpt:
    def test_improves_crossing_route(self):
        """2-opt should fix a crossing route."""
        stops, dm = make_stops_and_matrix()
        
        # Route with a crossing: 1→3→2→4 (crosses)
        crossing_route = Route(
            vehicle_id=1,
            stop_ids=[1, 3, 2, 4],
            total_time=compute_route_time([1, 3, 2, 4], dm),
            total_demand=8,
        )
        original_time = crossing_route.total_time
        
        routes, improvement = two_opt_pass([crossing_route], dm)
        assert improvement > 0, "2-opt should find an improvement on a crossing route"
        assert routes[0].total_time < original_time

    def test_no_improvement_on_optimal(self):
        """2-opt should not change an already optimal route."""
        stops, dm = make_stops_and_matrix()
        
        # Optimal route: 1→2→3→4 (no crossings)
        optimal_route = Route(
            vehicle_id=1,
            stop_ids=[1, 2, 3, 4],
            total_time=compute_route_time([1, 2, 3, 4], dm),
            total_demand=8,
        )
        original_time = optimal_route.total_time
        
        routes, improvement = two_opt_pass([optimal_route], dm)
        assert improvement < 0.01, "2-opt should not improve an already optimal route"

    def test_never_worsens(self):
        """Solution should never get worse after 2-opt."""
        stops, dm = make_stops_and_matrix()
        route = Route(
            vehicle_id=1,
            stop_ids=[1, 4, 2, 3],
            total_time=compute_route_time([1, 4, 2, 3], dm),
            total_demand=8,
        )
        original_time = route.total_time
        routes, _ = two_opt_pass([route], dm)
        assert routes[0].total_time <= original_time + 0.01


class TestOrOpt:
    def test_moves_stop_to_better_route(self):
        """Or-opt should move a stop closer to another route's cluster."""
        stops, dm = make_stops_and_matrix()
        
        # Route 1 has stops 1,2,4 — but 4 is far from 1,2 and near 3
        # Route 2 has stop 3
        route1 = Route(vehicle_id=1, stop_ids=[1, 2, 4],
                       total_time=compute_route_time([1, 2, 4], dm), total_demand=6)
        route2 = Route(vehicle_id=2, stop_ids=[3],
                       total_time=compute_route_time([3], dm), total_demand=2)
        
        old_total = route1.total_time + route2.total_time
        routes, improvement = or_opt_pass([route1, route2], dm, stops, vehicle_capacity=100)
        new_total = sum(r.total_time for r in routes)
        
        # Should either improve or stay same
        assert new_total <= old_total + 0.01


class TestRunLocalSearch:
    def test_returns_tuple(self):
        """run_local_search should return (solution, log)."""
        stops, dm = make_stops_and_matrix()
        initial = Solution(
            routes=[Route(vehicle_id=1, stop_ids=[1, 3, 2, 4],
                          total_time=compute_route_time([1, 3, 2, 4], dm),
                          total_demand=8)],
            method="test",
        )
        initial.recompute_total_time()
        
        result = run_local_search(initial, dm, stops, max_passes=5, max_time_seconds=5.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        sol, log = result
        assert isinstance(sol, Solution)
        assert isinstance(log, list)

    def test_improves_or_stays_same(self):
        """Final solution should be no worse than initial."""
        stops, dm = make_stops_and_matrix()
        initial = Solution(
            routes=[Route(vehicle_id=1, stop_ids=[1, 3, 2, 4],
                          total_time=compute_route_time([1, 3, 2, 4], dm),
                          total_demand=8)],
            method="test",
        )
        initial.recompute_total_time()
        original_time = initial.total_time
        
        sol, log = run_local_search(
            copy.deepcopy(initial), dm, stops, max_passes=10, max_time_seconds=5.0,
        )
        assert sol.total_time <= original_time + 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
