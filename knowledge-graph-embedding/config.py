"""
config.py
---------
Central configuration for the Knowledge Graph Embedding & Link Prediction project.

All tuneable hyper-parameters and directory paths live here so that every other
module can import from a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (absolute so imports work from any working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR        = PROJECT_ROOT / "data"
MODELS_DIR      = PROJECT_ROOT / "models"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR  / "checkpoints"
RESULTS_DIR     = OUTPUTS_DIR  / "results"
SRC_DIR         = PROJECT_ROOT / "src"

# Ensure runtime directories exist (safe no-op if already present)
for _dir in (DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "FB15k-237"          # PyKEEN dataset identifier

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME       = "TransE"
EMBEDDING_DIM    = 128              # Size of entity / relation embeddings
SCORING_FN_NORM  = 1                # L1 norm used in TransE score function

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
NUM_EPOCHS      = 100
BATCH_SIZE      = 512
LEARNING_RATE   = 1e-3
RANDOM_SEED     = 42

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_BATCH_SIZE = 64
TOP_K           = 10               # Hits@K to report

# ---------------------------------------------------------------------------
# Device  (CPU-first; change to "cuda" if a GPU is available)
# ---------------------------------------------------------------------------
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Checkpoint filename
# ---------------------------------------------------------------------------
CHECKPOINT_FILE = CHECKPOINTS_DIR / f"{MODEL_NAME.lower()}_trained.pkl"
