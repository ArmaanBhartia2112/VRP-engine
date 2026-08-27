#!/usr/bin/env python3
"""Phase 3: Run local search improvement on Clarke-Wright solution."""

import sys
import json
import copy
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Stop
from src.config import (
    STOP_SETS_DIR, PLOTS_DIR, METRICS_DIR,
    DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY,
)
from src.cache import get_or_compute
from src.utils import validate_solution, plot_solution_routes
from src.clarke_wright import clarke_wright
from src.local_search import run_local_search


def main():
    with open(STOP_SETS_DIR / "mumbai_10.json", "r") as f:
        data = json.load(f)
    stops = [Stop(**d) for d in data["stops"]]

    dm = get_or_compute(stops, source='haversine')

    print("Running Clarke-Wright...")
    initial_solution = clarke_wright(stops, dm)
    print(f"Initial cost: {initial_solution.total_time/60:.1f} min ({initial_solution.total_time:.0f}s)")

    print("\nRunning local search (2-opt + or-opt)...")
    sol_to_opt = copy.deepcopy(initial_solution)
    final_solution, log = run_local_search(sol_to_opt, dm, stops)

    if log:
        print(f"\n{'Pass':<6} {'Type':<8} {'Before (s)':<12} {'After (s)':<12} {'Delta (s)':<12} {'Elapsed':<10}")
        print("-" * 62)
        for entry in log:
            print(
                f"{entry['pass']:<6} {entry['type']:<8} "
                f"{entry['cost_before']:<12.1f} {entry['cost_after']:<12.1f} "
                f"{entry['delta']:<12.1f} {entry['elapsed_seconds']:<10.3f}"
            )
    else:
        print("  No improving moves found — Clarke-Wright solution is already locally optimal.")

    improvement = initial_solution.total_time - final_solution.total_time
    pct = (improvement / initial_solution.total_time * 100) if initial_solution.total_time > 0 else 0
    print(f"\nInitial cost:  {initial_solution.total_time/60:.1f} min")
    print(f"Final cost:    {final_solution.total_time/60:.1f} min")
    print(f"Improvement:   {improvement/60:.1f} min ({pct:.1f}%)")

    # Validate
    errors = validate_solution(
        final_solution, stops, dm,
        num_vehicles=DEFAULT_NUM_VEHICLES,
        vehicle_capacity=DEFAULT_VEHICLE_CAPACITY,
    )
    if not errors:
        print("\n✓ Validation: PASS")
    else:
        print(f"\n✗ Validation FAIL:")
        for e in errors:
            print(f"  - {e}")

    # Final routes
    print(f"\n{'Vehicle':<8} {'Route':<40} {'Time (min)':<12} {'Demand':<8}")
    print("-" * 70)
    for r in final_solution.routes:
        route_str = "0→" + "→".join(str(s) for s in r.stop_ids) + "→0"
        print(f"V{r.vehicle_id:<7} {route_str:<40} {r.total_time/60:<12.1f} {r.total_demand:<8}")

    # Save plots
    plot_solution_routes(
        final_solution, stops,
        title="Phase 3: Optimized Routes (CW + 2-opt + or-opt)",
        save_path=str(PLOTS_DIR / "phase3_routes.png"),
    )

    # Improvement curve
    costs = [initial_solution.total_time] + [e["cost_after"] for e in log]
    times = [0.0] + [e["elapsed_seconds"] for e in log]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, [c / 60 for c in costs], marker='o', linewidth=2)
    ax.set_xlabel("Elapsed Time (seconds)")
    ax.set_ylabel("Solution Cost (minutes)")
    ax.set_title("Local Search Improvement Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "phase3_improvement_curve.png"), dpi=150)
    print(f"\n  Saved improvement curve to {PLOTS_DIR / 'phase3_improvement_curve.png'}")
    plt.close(fig)

    # Save log
    with open(METRICS_DIR / "local_search_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved improvement log to {METRICS_DIR / 'local_search_log.json'}")


if __name__ == "__main__":
    main()
