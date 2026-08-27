import hashlib
import json

from src.models import DistanceMatrix, Stop
from src.distance_matrix import compute_google_matrix, compute_haversine_matrix
from src import config


def get_cache_key(stops: list[Stop], source: str) -> str:
    """Generates a cache key based on sorted stop coordinates and source label."""
    coords = sorted([(s.lat, s.lon) for s in stops])
    coords_str = json.dumps(coords)
    key_input = f"{coords_str}_{source}"
    return hashlib.sha256(key_input.encode('utf-8')).hexdigest()


def load_cached(stops: list[Stop], source: str) -> DistanceMatrix | None:
    """Loads a distance matrix from cache if it exists."""
    key = get_cache_key(stops, source)
    filename = key[:16] + '.json'
    filepath = config.CACHE_DIR / filename
    
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return DistanceMatrix.from_dict(data)
        except Exception:
            return None
    return None


def save_to_cache(stops: list[Stop], source: str, dm: DistanceMatrix) -> None:
    """Saves a distance matrix to the cache."""
    key = get_cache_key(stops, source)
    filename = key[:16] + '.json'
    filepath = config.CACHE_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dm.to_dict(), f, indent=2)


def get_or_compute(stops: list[Stop], source: str = 'google_maps') -> DistanceMatrix:
    """Returns a cached matrix or computes a new one and caches it."""
    dm = load_cached(stops, source)
    if dm is not None:
        return dm
        
    if source == 'google_maps':
        dm = compute_google_matrix(stops)
    elif source == 'haversine':
        dm = compute_haversine_matrix(stops)
    else:
        raise ValueError(f"Unknown source {source}")
        
    save_to_cache(stops, source, dm)
    return dm
