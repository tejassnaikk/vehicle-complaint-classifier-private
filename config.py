"""Central config: paths, seed, sample sizes. Class list is set after Phase 2."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data" / "raw"  # internal disk, gitignored
INTERIM_DIR = ROOT / "data" / "interim"
CLEAN_DIR = ROOT / "data" / "clean"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Applied to top_level BEFORE the multi-component rule (NHTSA renamed these categories over time).
MERGE_MAP = {
    "ENGINE AND ENGINE COOLING": "ENGINE",  # new NHTSA name; both in use side by side from 2020
    "FUEL SYSTEM": "FUEL/PROPULSION SYSTEM",  # old name for the same system
}
# VISIBILITY is NOT merged into VISIBILITY/WIPER: both coexist 2020-2026 with different content
# (sunroof, windshield, glass, mirrors vs a single wiper-style label), so VISIBILITY is dropped.

CLASSES = [
    "ENGINE", "POWER TRAIN", "ELECTRICAL SYSTEM", "STEERING", "SERVICE BRAKES", "AIR BAGS",
    "STRUCTURE", "FUEL/PROPULSION SYSTEM", "VISIBILITY/WIPER", "FORWARD COLLISION AVOIDANCE",
]

# Temporal windows (partial Nov 2026 excluded): train < VAL_START <= val < TEST_START <= test <= TEST_END
VAL_START = "2025-11-01"
TEST_START = "2026-05-01"
TEST_END = "2026-10-31"

TRAIN_CAP = 5000
EVAL_CAP = 800

# Plain-English names for the 10 classes (used in figures and on the website)
PLAIN_NAMES = {
    "ENGINE": "Engine and cooling", "POWER TRAIN": "Transmission and drive parts", "ELECTRICAL SYSTEM": "Electrical system",
    "STEERING": "Steering", "SERVICE BRAKES": "Brakes", "AIR BAGS": "Air bags", "STRUCTURE": "Body and frame",
    "FUEL/PROPULSION SYSTEM": "Fuel system", "VISIBILITY/WIPER": "Windshield, glass and wipers",
    "FORWARD COLLISION AVOIDANCE": "Automatic braking and collision warning",
}
