import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from src.clarke_wright import clarke_wright
from src.local_search import run_local_search
from src.exact_solver import solve_cvrp_exact, solve_cvrp_bruteforce
from src.utils import compute_route_time, compute_route_demand, haversine_time_seconds
from src.models import Stop, DistanceMatrix, Solution
from src.config import (
    DEFAULT_NUM_VEHICLES,
    DEFAULT_VEHICLE_CAPACITY,
    METRICS_DIR,
    PLOTS_DIR
)

def generate_random_stops(n: int) -> list[Stop]:
    """Generate n stops around Mumbai."""
    stops = []
    # Depot
    stops.append(Stop(id=0, name="Depot", lat=19.0760, lon=72.8777, demand=0))
    for i in range(1, n):
        lat = 19.0760 + random.uniform(-0.1, 0.1)
        lon = 72.8777 + random.uniform(-0.1, 0.1)
        demand = random.randint(1, 10)
        stops.append(Stop(id=i, name=f"Stop_{i}", lat=lat, lon=lon, demand=demand))
    return stops

def create_haversine_matrix(stops: list[Stop]) -> DistanceMatrix:
    """Create a Haversine distance matrix for given stops."""
    n = len(stops)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = haversine_time_seconds(
                    stops[i].lat, stops[i].lon,
                    stops[j].lat, stops[j].lon
                )
    return DistanceMatrix(
        matrix=mat,
        stop_ids=[s.id for s in stops],
        source="haversine"
    )

def main():
    random.seed(42)
    ns = [6, 8, 10, 12]
    instances_per_n = 3
    
    results = {}
    
    print(f"{'N':>3} | {'Inst':>4} | {'Heur Cost':>10} | {'Opt Cost':>10} | {'Gap %':>7} | {'Match Brute':>11}")
    print("-" * 65)
    
    gap_data = {}
    
    for n in ns:
        gap_data[n] = []
        for inst in range(instances_per_n):
            stops = generate_random_stops(n)
            dm = create_haversine_matrix(stops)
            
            # Heuristic
            sol_cw = clarke_wright(stops, dm, DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY)
            sol_heur, _ = run_local_search(sol_cw, dm, stops) if sol_cw else (None, [])
            heur_cost = sol_heur.total_time if sol_heur else float('inf')
            
            # Exact OR-Tools
            sol_opt = solve_cvrp_exact(stops, dm, DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY)
            opt_cost = sol_opt.total_time if sol_opt else float('inf')
            
            # Brute force
            match_brute = "N/A"
            if n <= 8:
                sol_brute = solve_cvrp_bruteforce(stops, dm, DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY)
                brute_cost = sol_brute.total_time if sol_brute else float('inf')
                if abs(brute_cost - opt_cost) < 1e-4:
                    match_brute = "Yes"
                else:
                    match_brute = "No"
            
            if opt_cost < float('inf') and heur_cost < float('inf') and opt_cost > 0:
                gap = (heur_cost - opt_cost) / opt_cost * 100
            else:
                gap = 0.0
                
            gap_data[n].append(gap)
            
            print(f"{n:>3} | {inst:>4} | {heur_cost:>10.2f} | {opt_cost:>10.2f} | {gap:>7.2f} | {match_brute:>11}")
            
            results_key = f"{n}_{inst}"
            results[results_key] = {
                "n": n,
                "instance": inst,
                "heuristic_cost": heur_cost,
                "optimal_cost": opt_cost,
                "gap_percent": gap,
                "match_brute": match_brute
            }
            
    with open(METRICS_DIR / "optimality_gaps.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Plotting
    ns_plot = []
    gaps_plot = []
    means = []
    stds = []
    
    for n in ns:
        gaps = gap_data[n]
        ns_plot.extend([n] * len(gaps))
        gaps_plot.extend(gaps)
        means.append(np.mean(gaps))
        stds.append(np.std(gaps))
        
    plt.figure(figsize=(8, 6))
    plt.scatter(ns_plot, gaps_plot, alpha=0.5, label="Instances")
    plt.errorbar(ns, means, yerr=stds, fmt='-o', color='red', capsize=5, label="Mean ± Std")
    plt.xlabel("Number of Stops (N)")
    plt.ylabel("Optimality Gap (%)")
    plt.title("Heuristic Optimality Gap vs Instance Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "phase4_optimality_gap.png")
    
    print("\nSummary Statistics:")
    for n, m, s in zip(ns, means, stds):
        print(f"N={n}: Mean Gap = {m:.2f}%, Std Dev = {s:.2f}%")

if __name__ == "__main__":
    main()
