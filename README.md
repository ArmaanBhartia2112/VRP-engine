# VRP Engine — Multi-Vehicle Route Optimization

A Capacitated Vehicle Routing Problem (CVRP) solver that uses real Google Maps
travel-time data, implements Clarke-Wright + local search heuristics, and includes
rigorous evaluation against exact-optimal solutions.


## Architecture

```
src
models.py          # Data classes: Stop, Vehicle, Route, Solution, DistanceMatrix
config.py          # API key, constants, paths
distance_matrix.py # Google Maps Routes API + haversine matrix computation
cache.py           # Disk-based matrix caching (JSON)
clarke_wright.py   # Clarke-Wright parallel savings algorithm
local_search.py    # 2-opt (intra-route) + or-opt (inter-route) improvement
exact_solver.py    # OR-Tools CVRP solver + brute-force for tiny instances
utils.py           # Haversine, route time computation, validation, plotting
```

## Problem Formulation

**Input**: N delivery stops with (lat, lon, demand), K vehicles with max capacity, 1 depot  
**Output**: Assignment of stops to vehicles + visit order per vehicle  
**Objective**: Minimize total travel time across all vehicles  

**Constraints**:
- Every stop visited exactly once
- Each route starts and ends at the depot
- Total demand per route ≤ vehicle capacity
- Number of routes ≤ K vehicles

See [docs/evaluation_report.md](docs/evaluation_report.md) for the full evaluation.

## API Usage

Uses the Google Maps **Routes API** (`computeRouteMatrix` endpoint):
- Traffic model: `TRAFFIC_UNAWARE` (static road distances)
- Requires `GOOGLE_MAPS_API_KEY` environment variable
- Results are cached to avoid repeat API calls

