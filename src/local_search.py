import time
from src.models import Route, Solution, DistanceMatrix, Stop
from src.config import DEFAULT_VEHICLE_CAPACITY, MAX_LOCAL_SEARCH_PASSES, MAX_LOCAL_SEARCH_TIME_SECONDS
from src.utils import compute_route_time

def two_opt_pass(routes: list[Route], dm: DistanceMatrix) -> tuple[list[Route], float]:
    total_improvement = 0.0
    for r in routes:
        improved = True
        while improved:
            improved = False
            
            n = len(r.stop_ids)
            if n < 2:
                break
                
            current_time = compute_route_time(r.stop_ids, dm)
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_seq = r.stop_ids[:i] + list(reversed(r.stop_ids[i:j+1])) + r.stop_ids[j+1:]
                    new_time = compute_route_time(new_seq, dm)
                    
                    if new_time < current_time - 0.001:
                        r.stop_ids = new_seq
                        r.total_time = new_time
                        total_improvement += (current_time - new_time)
                        current_time = new_time
                        improved = True
                        break
                if improved:
                    break
    return routes, total_improvement

def or_opt_pass(
    routes: list[Route], dm: DistanceMatrix, stops: list[Stop],
    vehicle_capacity: int = DEFAULT_VEHICLE_CAPACITY,
) -> tuple[list[Route], float]:
    total_improvement = 0.0
    improved = True
    stop_map = {s.id: s for s in stops}
    
    while improved:
        improved = False
        
        for r_idx_from in range(len(routes)):
            r_from = routes[r_idx_from]
            n_from = len(r_from.stop_ids)
            if n_from == 0:
                continue
                
            for chain_len in [1, 2]:
                for i in range(n_from - chain_len + 1):
                    chain = r_from.stop_ids[i:i+chain_len]
                    chain_demand = sum(stop_map[sid].demand for sid in chain)
                    
                    for r_idx_to in range(len(routes)):
                        r_to = routes[r_idx_to]
                        
                        if r_idx_from != r_idx_to and r_to.total_demand + chain_demand > vehicle_capacity:
                            continue
                            
                        n_to = len(r_to.stop_ids)
                        
                        for j in range(n_to + 1):
                            if r_idx_from == r_idx_to and (i <= j <= i + chain_len):
                                continue
                                
                            new_from_seq = r_from.stop_ids[:i] + r_from.stop_ids[i+chain_len:]
                            
                            if r_idx_from == r_idx_to:
                                if j > i:
                                    j_adj = j - chain_len
                                else:
                                    j_adj = j
                                new_to_seq = new_from_seq[:j_adj] + chain + new_from_seq[j_adj:]
                            else:
                                new_to_seq = r_to.stop_ids[:j] + chain + r_to.stop_ids[j:]
                                
                            time_from = r_from.total_time
                            time_to = r_to.total_time
                            
                            new_time_from = compute_route_time(new_from_seq, dm) if r_idx_from != r_idx_to else 0
                            new_time_to = compute_route_time(new_to_seq, dm)
                            
                            old_cost = time_from + time_to if r_idx_from != r_idx_to else time_from
                            new_cost = new_time_from + new_time_to if r_idx_from != r_idx_to else new_time_to
                            
                            if new_cost < old_cost - 0.001:
                                if r_idx_from != r_idx_to:
                                    r_from.stop_ids = new_from_seq
                                    r_from.total_time = new_time_from
                                    r_from.total_demand -= chain_demand
                                    
                                    r_to.stop_ids = new_to_seq
                                    r_to.total_time = new_time_to
                                    r_to.total_demand += chain_demand
                                else:
                                    r_from.stop_ids = new_to_seq
                                    r_from.total_time = new_cost
                                    
                                total_improvement += (old_cost - new_cost)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return routes, total_improvement


def run_local_search(
    solution: Solution,
    dm: DistanceMatrix,
    stops: list[Stop],
    vehicle_capacity: int = DEFAULT_VEHICLE_CAPACITY,
    max_passes: int = MAX_LOCAL_SEARCH_PASSES,
    max_time_seconds: float = MAX_LOCAL_SEARCH_TIME_SECONDS,
) -> tuple[Solution, list[dict]]:
    start_time = time.time()
    log = []
    
    routes = solution.routes
    
    for pass_idx in range(1, max_passes + 1):
        if time.time() - start_time > max_time_seconds:
            break
            
        old_cost = sum(r.total_time for r in routes)
        
        # 2-opt pass
        routes, two_opt_imp = two_opt_pass(routes, dm)
        if two_opt_imp > 0:
            new_cost = sum(r.total_time for r in routes)
            log.append({
                "pass": pass_idx,
                "type": "2-opt",
                "cost_before": old_cost,
                "cost_after": new_cost,
                "delta": two_opt_imp,
                "elapsed_seconds": time.time() - start_time
            })
            old_cost = new_cost
            
        # or-opt pass
        routes, or_opt_imp = or_opt_pass(routes, dm, stops, vehicle_capacity)
        if or_opt_imp > 0:
            new_cost = sum(r.total_time for r in routes)
            log.append({
                "pass": pass_idx,
                "type": "or-opt",
                "cost_before": old_cost,
                "cost_after": new_cost,
                "delta": or_opt_imp,
                "elapsed_seconds": time.time() - start_time
            })
            
        if two_opt_imp < 0.001 and or_opt_imp < 0.001:
            break
            
    solution.method = (solution.method or "") + "+2opt+oropt"
    solution.recompute_total_time()
    solution.runtime_seconds += time.time() - start_time
    
    return solution, log
