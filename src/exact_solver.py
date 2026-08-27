import time
import itertools
from typing import Optional

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from src.models import Stop, DistanceMatrix, Route, Solution
from src.config import DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY, EXACT_SOLVER_TIME_LIMIT_SECONDS
from src.utils import compute_route_time

def solve_cvrp_exact(
    stops: list[Stop],
    dm: DistanceMatrix,
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    vehicle_capacity: int = DEFAULT_VEHICLE_CAPACITY,
    time_limit_seconds: int = EXACT_SOLVER_TIME_LIMIT_SECONDS,
) -> Solution | None:
    """Solve CVRP using OR-Tools' routing solver.
    
    Returns Solution with method='ortools_exact', or None if no solution found.
    """
    start_time = time.perf_counter()
    num_stops = len(stops)
    
    manager = pywrapcp.RoutingIndexManager(num_stops, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    stop_ids = [s.id for s in stops]

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        from_id = stop_ids[from_node]
        to_id = stop_ids[to_node]
        return int(round(dm.time(from_id, to_id) * 100))

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return stops[from_node].demand

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  
        [vehicle_capacity] * num_vehicles,
        True,
        'Capacity'
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit_seconds
    search_parameters.log_search = False

    assignment = routing.SolveWithParameters(search_parameters)
    if not assignment:
        return None

    routes = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route_stop_ids = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:
                route_stop_ids.append(stop_ids[node_index])
            index = assignment.Value(routing.NextVar(index))
            
        if route_stop_ids:
            route_time = compute_route_time(route_stop_ids, dm)
            route_demand = sum(s.demand for s in stops if s.id in route_stop_ids)
            routes.append(Route(
                vehicle_id=vehicle_id,
                stop_ids=route_stop_ids,
                total_time=route_time,
                total_demand=route_demand
            ))
            
    sol = Solution(
        routes=routes,
        total_time=sum(r.total_time for r in routes),
        method="ortools_exact",
        runtime_seconds=time.perf_counter() - start_time,
        metadata={"status": routing.status()}
    )
    return sol


def solve_cvrp_bruteforce(
    stops: list[Stop],
    dm: DistanceMatrix,
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    vehicle_capacity: int = DEFAULT_VEHICLE_CAPACITY,
) -> Solution | None:
    """Brute-force optimal CVRP for very small instances (≤ 8 stops).
    
    Enumerates all possible partitions of stops into num_vehicles groups,
    and all permutations within each group. Returns the best feasible solution.
    Only use for N ≤ 8 — complexity is astronomical beyond that.
    """
    start_time = time.perf_counter()
    non_depots = [s for s in stops if s.id != 0]
    
    subset_best_cost = {}
    for r in range(1, len(non_depots) + 1):
        for subset in itertools.combinations(non_depots, r):
            if sum(s.demand for s in subset) > vehicle_capacity:
                continue
                
            subset_ids = frozenset(s.id for s in subset)
            best_cost = float('inf')
            best_perm = None
            
            for perm in itertools.permutations(s.id for s in subset):
                cost = compute_route_time(list(perm), dm)
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm
                    
            subset_best_cost[subset_ids] = (best_cost, best_perm)

    def generate_partitions(items, k):
        if not items:
            yield [[] for _ in range(k)]
            return
        first = items[0]
        for p in generate_partitions(items[1:], k):
            for i in range(k):
                new_p = [list(sub) for sub in p]
                new_p[i].append(first)
                yield new_p

    best_time = float('inf')
    best_routes = []
    
    for partition in generate_partitions(non_depots, num_vehicles):
        partition_cost = 0
        partition_routes = []
        valid = True
        
        for v_id, group in enumerate(partition):
            if not group:
                continue
            
            subset_ids = frozenset(s.id for s in group)
            if subset_ids not in subset_best_cost:
                valid = False
                break
                
            cost, perm = subset_best_cost[subset_ids]
            partition_cost += cost
            partition_routes.append((len(partition_routes), list(perm), cost, sum(s.demand for s in group)))
            
        if valid and partition_cost < best_time:
            best_time = partition_cost
            best_routes = partition_routes

    if best_time == float('inf'):
        return None
        
    routes = []
    for i, (v_id, stop_ids, cost, demand) in enumerate(best_routes):
        routes.append(Route(
            vehicle_id=i,
            stop_ids=stop_ids,
            total_time=cost,
            total_demand=demand
        ))
        
    sol = Solution(
        routes=routes,
        total_time=best_time,
        method="bruteforce",
        runtime_seconds=time.perf_counter() - start_time
    )
    return sol
