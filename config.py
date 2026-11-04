"""Central config: paths, seed, sample sizes. Class list is set after Phase 2."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data" / "raw"  # internal disk, gitignored
INTERIM_DIR = ROOT / "data" / "interim"
CLEAN_DIR = ROOT / "data" / "clean"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CLASSES = []  # chosen by the user after Phase 2

TRAIN_CAP = 5000
EVAL_CAP = 800
