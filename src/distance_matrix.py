import datetime
import json
import urllib.request
import urllib.error

import numpy as np

from src.models import DistanceMatrix, Stop
from src.utils import haversine_time_seconds
from src import config


def compute_google_matrix(stops: list[Stop]) -> DistanceMatrix:
    """Computes a distance matrix using the Google Maps Routes API."""
    api_key = config.get_api_key()
    n = len(stops)
    matrix = np.zeros((n, n))
    
    chunk_size = 25
    
    for origin_start in range(0, n, chunk_size):
        origin_end = min(origin_start + chunk_size, n)
        origins = stops[origin_start:origin_end]
        
        for dest_start in range(0, n, chunk_size):
            dest_end = min(dest_start + chunk_size, n)
            destinations = stops[dest_start:dest_end]
            
            origin_payload = [
                {"waypoint": {"location": {"latLng": {"latitude": s.lat, "longitude": s.lon}}}}
                for s in origins
            ]
            dest_payload = [
                {"waypoint": {"location": {"latLng": {"latitude": s.lat, "longitude": s.lon}}}}
                for s in destinations
            ]
            
            payload = {
                "origins": origin_payload,
                "destinations": dest_payload,
                "travelMode": "DRIVE",
                "routingPreference": "TRAFFIC_UNAWARE"
            }
            
            req = urllib.request.Request(
                config.ROUTES_API_ENDPOINT,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    "X-Goog-Api-Key": api_key,
                    "X-Goog-FieldMask": config.ROUTES_API_FIELD_MASK,
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            
            try:
                with urllib.request.urlopen(req) as response:
                    response_data = json.loads(response.read().decode('utf-8'))
                    
                    for element in response_data:
                        o_idx = element.get('originIndex', 0)
                        d_idx = element.get('destinationIndex', 0)
                        
                        global_o = origin_start + o_idx
                        global_d = dest_start + d_idx
                        
                        condition = element.get('condition', '')
                        if condition == 'ROUTE_NOT_FOUND' or 'duration' not in element:
                            if global_o == global_d:
                                matrix[global_o, global_d] = 0.0
                            else:
                                o_stop = stops[global_o]
                                d_stop = stops[global_d]
                                matrix[global_o, global_d] = haversine_time_seconds(
                                    o_stop.lat, o_stop.lon, d_stop.lat, d_stop.lon
                                )
                        else:
                            duration_str = element['duration']
                            duration_sec = float(duration_str.rstrip('s'))
                            matrix[global_o, global_d] = duration_sec
            except Exception as e:
                raise RuntimeError(f"Google API request failed: {e}")

    np.fill_diagonal(matrix, 0)
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return DistanceMatrix(
        matrix=matrix,
        stop_ids=[s.id for s in stops],
        source="google_maps",
        traffic_model="TRAFFIC_UNAWARE",
        timestamp=timestamp
    )


def compute_haversine_matrix(stops: list[Stop]) -> DistanceMatrix:
    """Computes a distance matrix using haversine distances."""
    n = len(stops)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = haversine_time_seconds(
                    stops[i].lat, stops[i].lon, stops[j].lat, stops[j].lon
                )
                
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return DistanceMatrix(
        matrix=matrix,
        stop_ids=[s.id for s in stops],
        source="haversine",
        traffic_model="",
        timestamp=timestamp
    )
