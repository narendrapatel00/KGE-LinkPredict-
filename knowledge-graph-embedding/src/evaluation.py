"""
evaluation.py
-------------
Metric computation for the trained TransE model.

Responsibilities
----------------
- Extract standard KGE evaluation metrics from a PipelineResult:
    * Mean Rank (MR)
    * Mean Reciprocal Rank (MRR)
    * Hits@1, Hits@3, Hits@10
- Persist results to outputs/results/ as JSON.
- Load persisted results from disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path so `config` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from config import RESULTS_DIR
except Exception as exc:
    logger.warning("Could not import config: %s — using fallback path", exc)
    RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_METRICS_FILE = RESULTS_DIR / "metrics.json"

# Metric keys recognised by PipelineResult.get_metric()
_METRIC_KEYS = ["hits@1", "hits@3", "hits@10", "mean_rank", "mean_reciprocal_rank"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(pipeline_result) -> dict:
    """Extract evaluation metrics from a trained PipelineResult.

    Parameters
    ----------
    pipeline_result : pykeen.pipeline.PipelineResult
        The result object from a completed PyKEEN training run.

    Returns
    -------
    dict
        Keys: ``hits@1``, ``hits@3``, ``hits@10``, ``mean_rank``,
        ``mean_reciprocal_rank``.  Values are floats or ``None`` if a metric
        could not be retrieved.
    """
    metrics: dict = {}
    for key in _METRIC_KEYS:
        try:
            value = pipeline_result.get_metric(key)
            metrics[key] = float(value) if value is not None else None
        except Exception as exc:
            logger.warning("Could not retrieve metric '%s': %s", key, exc)
            metrics[key] = None
    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(metrics: dict, filepath=None) -> None:
    """Persist an evaluation metrics dictionary to a JSON file.

    Parameters
    ----------
    metrics : dict
        The metrics dictionary returned by :func:`evaluate_model`.
    filepath : str or Path, optional
        Destination path.  Defaults to ``outputs/results/metrics.json``.
    """
    dest = Path(filepath) if filepath else _DEFAULT_METRICS_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(metrics)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()

    try:
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Evaluation results saved to %s", dest)
    except Exception as exc:
        logger.error("save_results failed: %s", exc)


def load_results(filepath=None) -> Optional[dict]:
    """Load previously saved evaluation metrics from a JSON file.

    Parameters
    ----------
    filepath : str or Path, optional
        Source path.  Defaults to ``outputs/results/metrics.json``.

    Returns
    -------
    dict or None
        Parsed metrics dictionary, or ``None`` if the file does not exist
        or cannot be parsed.
    """
    src = Path(filepath) if filepath else _DEFAULT_METRICS_FILE
    if not src.exists():
        logger.info("No results file found at %s", src)
        return None
    try:
        return json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("load_results failed: %s", exc)
        return None
