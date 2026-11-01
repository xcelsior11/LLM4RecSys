from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
SUB_DIR = ROOT / "submissions"

VIEW_LOG = DATA_DIR / "dataset11-30.csv"
SCHED_TRAIN = DATA_DIR / "export_arh_11-20-final.csv"
SCHED_VAL = DATA_DIR / "export_arh_21-30-final.csv"
SCHED_TEST = DATA_DIR / "export_arh_31-42-final.csv"
TEMPLATE = DATA_DIR / "submission.csv"

# A match is a watch if the overlap is at least half of the programme duration.
WATCH_FRAC = 0.5
MIN_OVERLAP_SEC = 1
N_RECS = 5

# Simple SGD matrix factorization (same idea as HW5, implicit 0/1 watches).
N_FACTORS = 16
N_EPOCHS = 8
LEARNING_RATE = 0.05
REG = 0.02
SEED = 42
