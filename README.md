# VRP Engine — Multi-Vehicle Route Optimization

A Capacitated Vehicle Routing Problem (CVRP) solver that uses real Google Maps
travel-time data, implements Clarke-Wright + local search heuristics, and includes
rigorous evaluation against exact-optimal solutions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Google Maps API key (optional — haversine fallback works without it)
export GOOGLE_MAPS_API_KEY="your_key_here"

# Run each phase
python scripts/run_phase1.py    # Generate travel-time matrices
python scripts/run_phase2.py    # Clarke-Wright initial solution
python scripts/run_phase3.py    # Local search improvement
python scripts/run_phase4.py    # Exact-solver comparison (takes ~10 min)
python scripts/run_evaluation.py  # Full evaluation pipeline + plots

# Run tests
python -m pytest tests/ -v
```

## Architecture

```
src/
├── models.py          # Data classes: Stop, Vehicle, Route, Solution, DistanceMatrix
├── config.py          # API key, constants, paths
├── distance_matrix.py # Google Maps Routes API + haversine matrix computation
├── cache.py           # Disk-based matrix caching (JSON)
├── clarke_wright.py   # Clarke-Wright parallel savings algorithm
├── local_search.py    # 2-opt (intra-route) + or-opt (inter-route) improvement
├── exact_solver.py    # OR-Tools CVRP solver + brute-force for tiny instances
└── utils.py           # Haversine, route time computation, validation, plotting
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

## Solver Pipeline

1. **Distance Matrix** (Phase 1): Google Maps Routes API for real driving times,
   with haversine fallback. Cached to disk.
2. **Clarke-Wright** (Phase 2): Parallel savings algorithm constructs initial feasible routes.
3. **Local Search** (Phase 3): 2-opt reversal + or-opt relocation improve routes iteratively.
4. **Exact Solver** (Phase 4): OR-Tools CVRP solver provides ground-truth for benchmarking.

## Key Results

See [docs/evaluation_report.md](docs/evaluation_report.md) for the full evaluation.

## API Usage

Uses the Google Maps **Routes API** (`computeRouteMatrix` endpoint):
- Traffic model: `TRAFFIC_UNAWARE` (static road distances)
- Requires `GOOGLE_MAPS_API_KEY` environment variable
- Results are cached to avoid repeat API calls

## License

MIT