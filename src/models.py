"""Data models for the VRP engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


@dataclass
class Stop:
    """A delivery stop (or depot when id == 0)."""
    id: int
    name: str
    lat: float
    lon: float
    demand: int = 0  # load units; depot has demand 0

    def __repr__(self) -> str:
        return f"Stop({self.id}, {self.name!r}, demand={self.demand})"


@dataclass
class Vehicle:
    """A delivery vehicle."""
    id: int
    capacity: int

    def __repr__(self) -> str:
        return f"Vehicle({self.id}, cap={self.capacity})"


@dataclass
class Route:
    """An ordered route for one vehicle, starting and ending at the depot."""
    vehicle_id: int
    stop_ids: list[int]          # ordered visit sequence (excludes depot bookends)
    total_time: float = 0.0      # seconds
    total_demand: int = 0        # sum of demands of stops on this route

    @property
    def full_path(self) -> list[int]:
        """Return the full path including depot (id=0) at start and end."""
        return [0] + self.stop_ids + [0]

    def __repr__(self) -> str:
        return (f"Route(v{self.vehicle_id}: "
                f"0→{'→'.join(str(s) for s in self.stop_ids)}→0, "
                f"time={self.total_time:.0f}s, load={self.total_demand})")


@dataclass
class Solution:
    """A complete VRP solution: a set of routes covering all stops."""
    routes: list[Route]
    total_time: float = 0.0      # seconds — sum of all route times
    method: str = ""             # e.g. "clarke_wright", "clarke_wright+2opt+oropt"
    runtime_seconds: float = 0.0 # wall-clock time to compute this solution
    metadata: dict = field(default_factory=dict)

    def recompute_total_time(self) -> None:
        self.total_time = sum(r.total_time for r in self.routes)

    def __repr__(self) -> str:
        return (f"Solution({self.method}, {len(self.routes)} routes, "
                f"total={self.total_time:.0f}s, runtime={self.runtime_seconds:.2f}s)")


@dataclass
class DistanceMatrix:
    """An NxN travel-time matrix between stops."""
    matrix: np.ndarray           # shape (N, N), values in seconds
    stop_ids: list[int]          # ordered stop IDs corresponding to rows/columns
    source: str = "unknown"      # "google_maps" or "haversine"
    traffic_model: str = ""      # e.g. "TRAFFIC_UNAWARE"
    timestamp: str = ""          # ISO-8601 when the matrix was computed

    def time(self, from_id: int, to_id: int) -> float:
        """Look up travel time between two stop IDs."""
        i = self.stop_ids.index(from_id)
        j = self.stop_ids.index(to_id)
        return float(self.matrix[i, j])

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "matrix": self.matrix.tolist(),
            "stop_ids": self.stop_ids,
            "source": self.source,
            "traffic_model": self.traffic_model,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DistanceMatrix:
        """Deserialize from a dict."""
        return cls(
            matrix=np.array(d["matrix"], dtype=float),
            stop_ids=d["stop_ids"],
            source=d.get("source", "unknown"),
            traffic_model=d.get("traffic_model", ""),
            timestamp=d.get("timestamp", ""),
        )

    def __repr__(self) -> str:
        n = len(self.stop_ids)
        return f"DistanceMatrix({n}x{n}, source={self.source!r})"
