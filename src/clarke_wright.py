import time
from src.models import Stop, Route, Solution, DistanceMatrix
from src.config import DEFAULT_NUM_VEHICLES, DEFAULT_VEHICLE_CAPACITY
from src.utils import compute_route_time

def clarke_wright(
    stops: list[Stop],
    dm: DistanceMatrix,
    num_vehicles: int = DEFAULT_NUM_VEHICLES,
    vehicle_capacity: int = DEFAULT_VEHICLE_CAPACITY,
) -> Solution:
    start_time = time.time()
    
    non_depot_stops = [s for s in stops if s.id != 0]
    
    # 1. Initial solution: one route per non-depot stop
    routes = []
    route_of_stop = {}
    for i, s in enumerate(non_depot_stops):
        r = Route(vehicle_id=i+1, stop_ids=[s.id], total_time=0.0, total_demand=s.demand)
        routes.append(r)
        route_of_stop[s.id] = r
        
    # 2. Compute savings list
    savings = []
    for i in range(len(non_depot_stops)):
        for j in range(i + 1, len(non_depot_stops)):
            s1 = non_depot_stops[i].id
            s2 = non_depot_stops[j].id
            sav = dm.time(0, s1) + dm.time(0, s2) - dm.time(s1, s2)
            savings.append((sav, s1, s2))
            
    # 3. Sort savings descending
    savings.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Merge routes greedily
    for sav, s1, s2 in savings:
        r1 = route_of_stop[s1]
        r2 = route_of_stop[s2]
        
        if r1 is r2:
            continue
            
        s1_exterior = (r1.stop_ids[0] == s1 or r1.stop_ids[-1] == s1)
        s2_exterior = (r2.stop_ids[0] == s2 or r2.stop_ids[-1] == s2)
        
        if not (s1_exterior and s2_exterior):
            continue
            
        if r1.total_demand + r2.total_demand > vehicle_capacity:
            continue
            
        if r1.stop_ids[-1] == s1:
            r1_seq = list(r1.stop_ids)
        else: # r1.stop_ids[0] == s1
            r1_seq = list(reversed(r1.stop_ids))
            
        if r2.stop_ids[0] == s2:
            r2_seq = list(r2.stop_ids)
        else: # r2.stop_ids[-1] == s2
            r2_seq = list(reversed(r2.stop_ids))
            
        new_seq = r1_seq + r2_seq
        
        r1.stop_ids = new_seq
        r1.total_demand += r2.total_demand
        
        for s in r2_seq:
            route_of_stop[s] = r1
            
        routes.remove(r2)
        
    # 5. Handle excess routes
    while len(routes) > num_vehicles:
        for r in routes:
            r.total_time = compute_route_time(r.stop_ids, dm)
            
        routes.sort(key=lambda x: x.total_time)
        
        r1 = routes[0]
        r2 = routes[1]
        
        r1.stop_ids.extend(r2.stop_ids)
        r1.total_demand += r2.total_demand
        
        for s in r2.stop_ids:
            route_of_stop[s] = r1
            
        routes.remove(r2)
        
    # Re-assign vehicle IDs and compute times
    for idx, r in enumerate(routes):
        r.vehicle_id = idx + 1
        r.total_time = compute_route_time(r.stop_ids, dm)
        
    solution = Solution(
        routes=routes,
        method="clarke_wright",
        runtime_seconds=time.time() - start_time
    )
    solution.recompute_total_time()
    return solution
