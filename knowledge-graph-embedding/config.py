"""
config.py — Central configuration for KGE Link Prediction project.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR        = PROJECT_ROOT / "data"
MODELS_DIR      = PROJECT_ROOT / "models"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR  / "checkpoints"
RESULTS_DIR     = OUTPUTS_DIR  / "results"
SRC_DIR         = PROJECT_ROOT / "src"

for _dir in (DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "FB15k-237"
MODEL_NAME   = "TransE"
EMBEDDING_DIM    = 50
SCORING_FN_NORM  = 1
NUM_EPOCHS      = 10
BATCH_SIZE      = 512
LEARNING_RATE   = 0.01
RANDOM_SEED     = 42
EVAL_BATCH_SIZE = 64
TOP_K           = 10
DEVICE          = "cpu"
CHECKPOINT_FILE        = CHECKPOINTS_DIR / f"{MODEL_NAME.lower()}_trained.pkl"
TRAINING_METADATA_FILE = CHECKPOINTS_DIR / "training_metadata.json"
RESULTS_FILE           = RESULTS_DIR     / "metrics.json"
