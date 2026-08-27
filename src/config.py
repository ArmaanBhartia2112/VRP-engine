"""Configuration and constants for the VRP engine."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


def get_api_key() -> str:
    """Get the Google Maps API key from environment."""
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY not set. "
            "Set it as an environment variable or in a .env file in the project root."
        )
    return key


# --- API Configuration ---
ROUTES_API_ENDPOINT = (
    "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
)
ROUTES_API_FIELD_MASK = "originIndex,destinationIndex,duration,condition,status"
MAX_MATRIX_ELEMENTS = 625  # Routes API limit: 25 origins × 25 destinations

# --- Default VRP Parameters ---
DEFAULT_NUM_VEHICLES = 3
DEFAULT_VEHICLE_CAPACITY = 40
DEPOT_INDEX = 0  # Stop index 0 is always the depot

# --- Local Search Defaults ---
MAX_LOCAL_SEARCH_PASSES = 1000
MAX_LOCAL_SEARCH_TIME_SECONDS = 30.0

# --- Exact Solver Defaults ---
EXACT_SOLVER_TIME_LIMIT_SECONDS = 60

# --- Paths ---
PROJECT_ROOT = _project_root
DATA_DIR = PROJECT_ROOT / "data"
STOP_SETS_DIR = DATA_DIR / "stop_sets"
CACHE_DIR = DATA_DIR / "matrix_cache"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"

# Ensure directories exist
for d in [STOP_SETS_DIR, CACHE_DIR, PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
