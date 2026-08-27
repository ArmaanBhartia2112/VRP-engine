#!/usr/bin/env python3
"""Phase 2: Run Clarke-Wright savings algorithm and validate the solution."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Stop
from src.config import STOP_SETS_DIR, PLOTS_DIR, DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY
from src.cache import get_or_compute
from src.utils import validate_solution, plot_solution_routes
from src.clarke_wright import clarke_wright


def main():
    with open(STOP_SETS_DIR / "mumbai_10.json", "r") as f:
        data = json.load(f)
    stops = [Stop(**d) for d in data["stops"]]

    print("Loading/computing haversine matrix...")
    dm = get_or_compute(stops, source='haversine')
    print(f"Matrix: {dm.matrix.shape[0]}x{dm.matrix.shape[1]}")

    print(f"\nRunning Clarke-Wright (K={DEFAULT_NUM_VEHICLES}, capacity={DEFAULT_VEHICLE_CAPACITY})...")
    solution = clarke_wright(stops, dm)

    print(f"\n{'Vehicle':<8} {'Route':<40} {'Time (min)':<12} {'Demand':<8}")
    print("-" * 70)
    for r in solution.routes:
        route_str = "0→" + "→".join(str(s) for s in r.stop_ids) + "→0"
        print(f"V{r.vehicle_id:<7} {route_str:<40} {r.total_time/60:<12.1f} {r.total_demand:<8}")
    print(f"\nTotal time: {solution.total_time/60:.1f} min ({solution.total_time:.0f}s)")
    print(f"Runtime: {solution.runtime_seconds:.4f}s")

    errors = validate_solution(
        solution, stops, dm,
        num_vehicles=DEFAULT_NUM_VEHICLES,
        vehicle_capacity=DEFAULT_VEHICLE_CAPACITY,
    )
    if not errors:
        print("\n✓ Validation: PASS")
    else:
        print(f"\n✗ Validation FAIL:")
        for e in errors:
            print(f"  - {e}")

    plot_solution_routes(
        solution, stops,
        title="Phase 2: Clarke-Wright Routes",
        save_path=str(PLOTS_DIR / "phase2_routes.png"),
    )


if __name__ == "__main__":
    main()
