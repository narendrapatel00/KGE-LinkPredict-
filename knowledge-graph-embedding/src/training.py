"""
training.py
-----------
TransE model training pipeline using PyKEEN.

Responsibilities
----------------
- Instantiate and configure the PyKEEN pipeline for TransE on FB15k-237.
- Run the training loop with configurable hyper-parameters.
- Save the trained pipeline (model + metadata) to outputs/checkpoints/.
- Load previously saved checkpoints from disk via a lightweight wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path so `config` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from config import CHECKPOINTS_DIR, RESULTS_DIR
except Exception as exc:
    logger.warning("Could not import config: %s — using fallback paths", exc)
    _root = Path(__file__).parent.parent
    CHECKPOINTS_DIR = _root / "outputs" / "checkpoints"
    RESULTS_DIR = _root / "outputs" / "results"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_METADATA_FILE = CHECKPOINTS_DIR / "training_metadata.json"

# Mapping from friendly metric names (used by PipelineResult.get_metric)
# to the nested keys stored in PyKEEN's results.json["metrics"]["both"]["realistic"].
_RESULTS_JSON_KEY_MAP = {
    "hits@1": "hits_at_1",
    "hits@3": "hits_at_3",
    "hits@10": "hits_at_10",
    "mean_rank": "arithmetic_mean_rank",
    "mean_reciprocal_rank": "inverse_harmonic_mean_rank",
}


# ---------------------------------------------------------------------------
# Loaded checkpoint wrapper
# ---------------------------------------------------------------------------

@dataclass
class LoadedCheckpoint:
    """Lightweight wrapper around a checkpoint loaded from disk.

    Exposes the same ``.model``, ``.training``, and ``.get_metric(key)``
    interface as ``pykeen.pipeline.PipelineResult`` so that
    :mod:`src.evaluation` and :mod:`src.prediction` work identically on
    both freshly trained results and checkpoints restored from disk.
    """

    model: Any
    training: Any
    metrics_dict: dict = field(default_factory=dict)
    _evaluated_metrics: Optional[dict] = field(default=None, repr=False)

    def get_metric(self, key: str) -> float:
        """Return a metric value by friendly key (e.g. ``'hits@10'``)."""
        # 1. Try pre-loaded metrics from results.json
        if self.metrics_dict:
            both_realistic = (
                self.metrics_dict.get("metrics", {})
                .get("both", {})
                .get("realistic", {})
            )
            mapped_key = _RESULTS_JSON_KEY_MAP.get(key, key)
            if mapped_key in both_realistic and both_realistic[mapped_key] is not None:
                return float(both_realistic[mapped_key])
            if key in both_realistic and both_realistic[key] is not None:
                return float(both_realistic[key])

        # 2. Fallback: evaluate on FB15k-237 test split if results.json was absent
        if self._evaluated_metrics is None:
            self._evaluated_metrics = self._evaluate_fallback()

        if key in self._evaluated_metrics and self._evaluated_metrics[key] is not None:
            return float(self._evaluated_metrics[key])

        raise KeyError(f"Metric {key!r} not available in loaded checkpoint.")

    def _evaluate_fallback(self) -> dict:
        """Run RankBasedEvaluator on FB15k-237 test split as a fallback."""
        try:
            from pykeen.evaluation import RankBasedEvaluator
            from src.dataset import load_dataset

            dataset = load_dataset()
            evaluator = RankBasedEvaluator()
            eval_results = evaluator.evaluate(
                model=self.model,
                mapped_triples=dataset.testing.mapped_triples,
                additional_filter_triples=[
                    dataset.training.mapped_triples,
                    dataset.validation.mapped_triples,
                ],
            )
            return {
                k: float(eval_results.get_metric(k))
                for k in _RESULTS_JSON_KEY_MAP
            }
        except Exception as exc:
            logger.warning("Fallback evaluation failed: %s", exc)
            return {}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    num_epochs: int = 10,
    embedding_dim: int = 50,
    batch_size: int = 512,
    learning_rate: float = 0.01,
    random_seed: int = 42,
    device: str = "cpu",
):
    """Train a TransE model on FB15k-237 and persist the checkpoint.

    Parameters
    ----------
    num_epochs : int
        Number of full training epochs (passed as ``epochs`` to PyKEEN).
    embedding_dim : int
        Dimensionality of entity and relation embeddings.
    batch_size : int
        Mini-batch size for the stochastic local closed-world assumption sampler.
    learning_rate : float
        Initial learning rate for Adam.
    random_seed : int
        Global random seed for reproducibility.
    device : str
        One of ``'cpu'`` or ``'cuda'``.

    Returns
    -------
    pykeen.pipeline.PipelineResult
        The PyKEEN result object containing the trained model and metrics.
    """
    from pykeen.pipeline import pipeline

    logger.info(
        "Starting TransE training: epochs=%d, dim=%d, batch=%d, lr=%g",
        num_epochs, embedding_dim, batch_size, learning_rate,
    )

    result = pipeline(
        model="TransE",
        dataset="FB15k237",
        epochs=num_epochs,
        training_kwargs=dict(batch_size=batch_size),
        optimizer_kwargs=dict(lr=learning_rate),
        model_kwargs=dict(embedding_dim=embedding_dim),
        random_seed=random_seed,
        device=device,
    )

    # Persist the full pipeline result (model + triples factories + results.json)
    result.save_to_directory(str(CHECKPOINTS_DIR))
    logger.info("Checkpoint saved to %s", CHECKPOINTS_DIR)

    # Save human-readable metadata alongside the checkpoint
    metadata = {
        "model_name": "TransE",
        "dataset": "FB15k237",
        "num_epochs": num_epochs,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Training metadata saved to %s", _METADATA_FILE)
    except Exception as exc:
        logger.warning("Could not save training metadata: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint() -> Optional[LoadedCheckpoint]:
    """Load a previously saved PyKEEN checkpoint from ``CHECKPOINTS_DIR``.

    Loads ``trained_model.pkl`` via ``torch.load``, ``training_triples/`` via
    ``TriplesFactory.from_path_binary`` (falling back to
    ``load_dataset().training`` if absent), and ``results.json`` if present.

    Returns
    -------
    LoadedCheckpoint or None
        Wrapper exposing ``.model``, ``.training``, and ``.get_metric(key)``,
        or ``None`` if no checkpoint is found or loading fails.
    """
    model_pkl = CHECKPOINTS_DIR / "trained_model.pkl"
    if not model_pkl.exists():
        logger.info("No checkpoint found at %s", model_pkl)
        return None

    try:
        import torch

        model = torch.load(str(model_pkl), map_location="cpu", weights_only=False)
        model.eval()
    except Exception as exc:
        logger.warning("Failed to load model from %s: %s", model_pkl, exc)
        return None

    # Load training TriplesFactory from binary directory if available
    training_factory = None
    triples_dir = CHECKPOINTS_DIR / "training_triples"
    if triples_dir.exists():
        try:
            from pykeen.triples import TriplesFactory
            training_factory = TriplesFactory.from_path_binary(str(triples_dir))
        except Exception as exc:
            logger.warning("Failed to load training_triples from %s: %s", triples_dir, exc)

    if training_factory is None:
        try:
            from src.dataset import load_dataset
            training_factory = load_dataset().training
        except Exception as exc:
            logger.warning("Fallback load_dataset().training failed: %s", exc)
            return None

    # Load results.json if present
    metrics_dict: dict = {}
    results_json = CHECKPOINTS_DIR / "results.json"
    if results_json.exists():
        try:
            metrics_dict = json.loads(results_json.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", results_json, exc)

    logger.info("Loaded checkpoint from %s", CHECKPOINTS_DIR)
    return LoadedCheckpoint(
        model=model,
        training=training_factory,
        metrics_dict=metrics_dict,
    )


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def get_training_metadata() -> Optional[dict]:
    """Read the training metadata JSON saved alongside the checkpoint.

    Returns
    -------
    dict or None
        Parsed metadata dictionary, or ``None`` if the file does not exist.
    """
    if not _METADATA_FILE.exists():
        return None
    try:
        return json.loads(_METADATA_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read training metadata: %s", exc)
        return None


def is_model_trained() -> bool:
    """Return True if a trained model checkpoint exists on disk."""
    if not CHECKPOINTS_DIR.exists():
        return False
    return (CHECKPOINTS_DIR / "trained_model.pkl").exists()
