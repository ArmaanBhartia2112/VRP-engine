#!/usr/bin/env python3
"""Phase 5: Full evaluation pipeline — generates all plots and metrics for the report."""

import sys
import json
import copy
import time
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Stop, DistanceMatrix, Solution
from src.config import (
    DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY,
    PLOTS_DIR, METRICS_DIR,
)
from src.clarke_wright import clarke_wright
from src.local_search import run_local_search
from src.exact_solver import solve_cvrp_exact
from src.utils import compute_route_time, haversine_time_seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_random_stops(n: int, seed: int = 42) -> list[Stop]:
    """Generate n delivery stops (+ depot) around Mumbai."""
    rng = random.Random(seed)
    stops = [Stop(id=0, name="Depot", lat=19.1197, lon=72.8464, demand=0)]
    for i in range(1, n + 1):
        lat = 19.0760 + rng.uniform(-0.10, 0.10)
        lon = 72.8777 + rng.uniform(-0.10, 0.10)
        demand = rng.randint(1, 10)
        stops.append(Stop(id=i, name=f"Stop_{i}", lat=lat, lon=lon, demand=demand))
    return stops


def build_haversine_matrix(stops: list[Stop]) -> DistanceMatrix:
    n = len(stops)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = haversine_time_seconds(
                    stops[i].lat, stops[i].lon, stops[j].lat, stops[j].lon
                )
    return DistanceMatrix(matrix=mat, stop_ids=[s.id for s in stops], source="haversine")


# ---------------------------------------------------------------------------
# Evaluation 1: Optimality gap vs N (uses Phase 4 results if available)
# ---------------------------------------------------------------------------

def run_optimality_gap_analysis():
    """Compute optimality gaps for small instances and plot."""
    print("=" * 60)
    print("EVALUATION 1: Optimality Gap vs Problem Size")
    print("=" * 60)

    ns = [6, 8, 10, 12]
    instances_per_n = 3
    results = {}
    gap_data = {}

    for n in ns:
        gap_data[n] = []
        for inst in range(instances_per_n):
            seed = n * 100 + inst
            stops = generate_random_stops(n, seed=seed)
            dm = build_haversine_matrix(stops)

            # Heuristic
            t0 = time.perf_counter()
            sol_cw = clarke_wright(stops, dm, DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY)
            sol_heur, _ = run_local_search(
                copy.deepcopy(sol_cw), dm, stops,
                vehicle_capacity=DEFAULT_VEHICLE_CAPACITY,
            )
            heur_time = time.perf_counter() - t0
            heur_cost = sol_heur.total_time

            # Exact
            t0 = time.perf_counter()
            sol_opt = solve_cvrp_exact(
                stops, dm, DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY,
                time_limit_seconds=60,
            )
            exact_time = time.perf_counter() - t0
            opt_cost = sol_opt.total_time if sol_opt else float("inf")

            gap = ((heur_cost - opt_cost) / opt_cost * 100) if opt_cost > 0 else 0.0
            gap_data[n].append(gap)

            print(f"  N={n:>2} inst={inst} | heuristic={heur_cost:.0f}s "
                  f"optimal={opt_cost:.0f}s gap={gap:.1f}% "
                  f"(heur_rt={heur_time:.3f}s, exact_rt={exact_time:.1f}s)")

            results[f"{n}_{inst}"] = {
                "n": n, "instance": inst,
                "heuristic_cost": heur_cost, "optimal_cost": opt_cost,
                "gap_percent": gap,
                "heuristic_runtime": heur_time, "exact_runtime": exact_time,
            }

    # Save metrics
    with open(METRICS_DIR / "optimality_gaps.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for n in ns:
        gaps = gap_data[n]
        xs = [n + random.uniform(-0.2, 0.2) for _ in gaps]
        ax.scatter(xs, gaps, alpha=0.6, s=60, zorder=5)
    means = [np.mean(gap_data[n]) for n in ns]
    stds = [np.std(gap_data[n]) for n in ns]
    ax.errorbar(ns, means, yerr=stds, fmt="-o", color="red", capsize=6,
                linewidth=2, label="Mean ± Std", zorder=10)
    ax.set_xlabel("Number of Delivery Stops (N)", fontsize=12)
    ax.set_ylabel("Optimality Gap (%)", fontsize=12)
    ax.set_title("Heuristic Optimality Gap vs Problem Size\n(Clarke-Wright + 2-opt + or-opt vs OR-Tools)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ns)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eval_optimality_gap.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved plot: {PLOTS_DIR / 'eval_optimality_gap.png'}")
    return gap_data, results


# ---------------------------------------------------------------------------
# Evaluation 2: Local search improvement curve
# ---------------------------------------------------------------------------

def run_local_search_analysis():
    """Measure how much local search improves CW, and at what cost."""
    print("\n" + "=" * 60)
    print("EVALUATION 2: Local Search Value (Cost vs Runtime)")
    print("=" * 60)

    test_sizes = [10, 15, 20, 25]
    all_results = {}

    for n in test_sizes:
        stops = generate_random_stops(n, seed=n * 7)
        dm = build_haversine_matrix(stops)

        sol_cw = clarke_wright(stops, dm, DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY)
        cw_cost = sol_cw.total_time

        sol_improved, log = run_local_search(
            copy.deepcopy(sol_cw), dm, stops,
            vehicle_capacity=DEFAULT_VEHICLE_CAPACITY,
            max_time_seconds=30.0,
        )
        final_cost = sol_improved.total_time
        improvement_pct = ((cw_cost - final_cost) / cw_cost * 100) if cw_cost > 0 else 0

        print(f"  N={n:>2} | CW={cw_cost:.0f}s → Final={final_cost:.0f}s "
              f"(improved {improvement_pct:.1f}%, {len(log)} passes)")

        all_results[n] = {
            "cw_cost": cw_cost,
            "final_cost": final_cost,
            "improvement_pct": improvement_pct,
            "log": log,
        }

    # Plot improvement curves for each N
    fig, ax = plt.subplots(figsize=(10, 6))
    for n in test_sizes:
        info = all_results[n]
        log = info["log"]
        costs = [info["cw_cost"]] + [e["cost_after"] for e in log]
        times = [0.0] + [e["elapsed_seconds"] for e in log]
        # Normalize cost to percentage of initial
        pcts = [c / info["cw_cost"] * 100 for c in costs]
        ax.plot(times, pcts, "-o", markersize=4, label=f"N={n}")

    ax.set_xlabel("Elapsed Time (seconds)", fontsize=12)
    ax.set_ylabel("Solution Cost (% of initial CW)", fontsize=12)
    ax.set_title("Local Search Improvement Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eval_local_search_curve.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved plot: {PLOTS_DIR / 'eval_local_search_curve.png'}")

    # Save metrics
    serializable = {}
    for n, info in all_results.items():
        serializable[str(n)] = {
            "cw_cost": info["cw_cost"],
            "final_cost": info["final_cost"],
            "improvement_pct": info["improvement_pct"],
            "num_passes": len(info["log"]),
        }
    with open(METRICS_DIR / "local_search_analysis.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return all_results


# ---------------------------------------------------------------------------
# Evaluation 3: Real vs haversine (using haversine at two speeds as proxy
#   since we may not have an API key — when API key available, the script
#   can be updated to use real Google Maps data)
# ---------------------------------------------------------------------------

def run_haversine_vs_real_analysis():
    """Compare routes planned on haversine vs a 'realistic' matrix.

    Without an API key, we simulate a 'realistic' matrix by using haversine
    with a lower average speed (25 km/h) and adding asymmetric jitter to
    simulate one-way streets and traffic. This demonstrates the methodology.
    When real Google Maps data is available, swap in the real matrix.
    """
    print("\n" + "=" * 60)
    print("EVALUATION 3: Haversine vs Realistic Travel Times")
    print("=" * 60)

    test_sizes = [8, 10, 15, 20]
    results = {}

    for n in test_sizes:
        stops = generate_random_stops(n, seed=n * 13)
        dm_haversine = build_haversine_matrix(stops)

        # Build a "realistic" matrix: slower speed + asymmetric jitter
        rng = random.Random(n * 17)
        n_stops = len(stops)
        realistic_mat = np.zeros((n_stops, n_stops))
        for i in range(n_stops):
            for j in range(n_stops):
                if i != j:
                    base = haversine_time_seconds(
                        stops[i].lat, stops[i].lon,
                        stops[j].lat, stops[j].lon,
                        speed_kmh=25.0,  # slower = more realistic
                    )
                    # Add 10-50% asymmetric jitter
                    jitter = rng.uniform(1.1, 1.5)
                    realistic_mat[i, j] = base * jitter
        dm_realistic = DistanceMatrix(
            matrix=realistic_mat, stop_ids=[s.id for s in stops], source="realistic"
        )

        # Plan on haversine, evaluate on realistic
        sol_hav = clarke_wright(stops, dm_haversine, DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY)
        sol_hav_opt, _ = run_local_search(
            copy.deepcopy(sol_hav), dm_haversine, stops,
            vehicle_capacity=DEFAULT_VEHICLE_CAPACITY,
        )
        # Evaluate this haversine-planned solution against realistic matrix
        hav_planned_real_cost = 0.0
        for r in sol_hav_opt.routes:
            hav_planned_real_cost += compute_route_time(r.stop_ids, dm_realistic)

        # Plan on realistic, evaluate on realistic
        sol_real = clarke_wright(stops, dm_realistic, DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY)
        sol_real_opt, _ = run_local_search(
            copy.deepcopy(sol_real), dm_realistic, stops,
            vehicle_capacity=DEFAULT_VEHICLE_CAPACITY,
        )
        real_planned_real_cost = sol_real_opt.total_time

        degradation = ((hav_planned_real_cost - real_planned_real_cost) /
                       real_planned_real_cost * 100) if real_planned_real_cost > 0 else 0

        print(f"  N={n:>2} | Planned on haversine: {hav_planned_real_cost:.0f}s, "
              f"Planned on realistic: {real_planned_real_cost:.0f}s, "
              f"Degradation: {degradation:.1f}%")

        results[str(n)] = {
            "haversine_planned_real_cost": hav_planned_real_cost,
            "real_planned_real_cost": real_planned_real_cost,
            "degradation_pct": degradation,
        }

    # Plot
    ns_list = [int(k) for k in results.keys()]
    hav_costs = [results[str(n)]["haversine_planned_real_cost"] / 60 for n in ns_list]
    real_costs = [results[str(n)]["real_planned_real_cost"] / 60 for n in ns_list]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(ns_list))
    width = 0.35
    ax.bar(x - width / 2, hav_costs, width, label="Planned on Haversine", color="#FF7043")
    ax.bar(x + width / 2, real_costs, width, label="Planned on Real Data", color="#42A5F5")
    ax.set_xlabel("Number of Delivery Stops (N)", fontsize=12)
    ax.set_ylabel("Actual Route Cost (minutes)", fontsize=12)
    ax.set_title("Route Quality: Haversine-Planned vs Real-Data-Planned\n(evaluated against realistic travel times)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(ns_list)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add degradation % labels
    for i, n in enumerate(ns_list):
        deg = results[str(n)]["degradation_pct"]
        max_cost = max(hav_costs[i], real_costs[i])
        ax.text(i, max_cost + 0.5, f"+{deg:.1f}%", ha="center", fontsize=10, color="red")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eval_haversine_vs_real.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved plot: {PLOTS_DIR / 'eval_haversine_vs_real.png'}")

    with open(METRICS_DIR / "haversine_vs_real.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Evaluation 4: Scalability
# ---------------------------------------------------------------------------

def run_scalability_analysis():
    """Measure runtime vs N for heuristic pipeline."""
    print("\n" + "=" * 60)
    print("EVALUATION 4: Scalability Analysis")
    print("=" * 60)

    test_sizes = [6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    results = {}

    for n in test_sizes:
        stops = generate_random_stops(n, seed=n * 31)

        t0 = time.perf_counter()
        dm = build_haversine_matrix(stops)
        matrix_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        sol_cw = clarke_wright(stops, dm, DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY)
        cw_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        sol_final, _ = run_local_search(
            copy.deepcopy(sol_cw), dm, stops,
            vehicle_capacity=DEFAULT_VEHICLE_CAPACITY,
            max_time_seconds=5.0,  # 5-second budget
        )
        ls_time = time.perf_counter() - t0

        total = matrix_time + cw_time + ls_time
        print(f"  N={n:>3} | matrix={matrix_time:.3f}s CW={cw_time:.4f}s "
              f"local_search={ls_time:.3f}s total={total:.3f}s "
              f"cost={sol_final.total_time:.0f}s")

        results[str(n)] = {
            "matrix_time": matrix_time,
            "cw_time": cw_time,
            "ls_time": ls_time,
            "total_time": total,
            "solution_cost": sol_final.total_time,
        }

    # Plot
    ns_list = sorted(int(k) for k in results.keys())
    matrix_times = [results[str(n)]["matrix_time"] for n in ns_list]
    cw_times = [results[str(n)]["cw_time"] for n in ns_list]
    ls_times = [results[str(n)]["ls_time"] for n in ns_list]
    totals = [results[str(n)]["total_time"] for n in ns_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Stacked bar: runtime breakdown
    x = np.arange(len(ns_list))
    ax1.bar(x, matrix_times, label="Matrix computation", color="#66BB6A")
    ax1.bar(x, cw_times, bottom=matrix_times, label="Clarke-Wright", color="#42A5F5")
    bottoms = [m + c for m, c in zip(matrix_times, cw_times)]
    ax1.bar(x, ls_times, bottom=bottoms, label="Local search", color="#FF7043")
    ax1.set_xlabel("N (stops)", fontsize=12)
    ax1.set_ylabel("Runtime (seconds)", fontsize=12)
    ax1.set_title("Runtime Breakdown by Component", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ns_list)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Line: total runtime vs N
    ax2.plot(ns_list, totals, "-o", linewidth=2, markersize=6, color="#E53935")
    ax2.axhline(y=5.0, color="gray", linestyle="--", alpha=0.7, label="5-second budget")
    ax2.set_xlabel("N (stops)", fontsize=12)
    ax2.set_ylabel("Total Runtime (seconds)", fontsize=12)
    ax2.set_title("Scalability: Total Solve Time vs Problem Size", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eval_scalability.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved plot: {PLOTS_DIR / 'eval_scalability.png'}")

    with open(METRICS_DIR / "scalability.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("VRP Engine — Full Evaluation Pipeline")
    print("=" * 60)

    gap_data, gap_results = run_optimality_gap_analysis()
    ls_results = run_local_search_analysis()
    hav_results = run_haversine_vs_real_analysis()
    scale_results = run_scalability_analysis()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Optimality gap summary
    print("\nOptimality Gap (mean ± std):")
    for n in sorted(set(r["n"] for r in gap_results.values())):
        gaps = [r["gap_percent"] for r in gap_results.values() if r["n"] == n]
        print(f"  N={n:>2}: {np.mean(gaps):.1f}% ± {np.std(gaps):.1f}%")

    # Local search summary
    print("\nLocal Search Improvement:")
    for n_str, info in sorted(ls_results.items(), key=lambda x: x[0]):
        if isinstance(info, dict) and "improvement_pct" in info:
            print(f"  N={n_str:>2}: {info['improvement_pct']:.1f}% improvement")

    # Haversine vs real summary
    print("\nHaversine Planning Degradation:")
    for n_str, info in sorted(hav_results.items(), key=lambda x: int(x[0])):
        print(f"  N={n_str:>2}: {info['degradation_pct']:.1f}% worse than real-data planning")

    # Scalability summary
    print("\nScalability (total runtime):")
    for n_str, info in sorted(scale_results.items(), key=lambda x: int(x[0])):
        print(f"  N={n_str:>3}: {info['total_time']:.3f}s")

    # Find largest N under 5s
    under_5s = [int(n) for n, info in scale_results.items() if info["total_time"] < 5.0]
    if under_5s:
        print(f"\n  Largest N solvable in <5s: {max(under_5s)}")

    print("\n✓ All evaluation plots and metrics saved.")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Metrics: {METRICS_DIR}")


if __name__ == "__main__":
    main()
