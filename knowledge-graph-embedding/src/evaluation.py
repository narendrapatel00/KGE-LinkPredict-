"""
evaluation.py
-------------
Metric computation for the trained TransE model.

Responsibilities
----------------
- Compute standard KGE evaluation metrics on the test split:
    * Mean Rank (MR)
    * Mean Reciprocal Rank (MRR)
    * Hits@1, Hits@3, Hits@10
- Support both raw and filtered evaluation settings.
- Persist results to outputs/results/ as JSON.

Notes
-----
Actual implementation is deferred to Phase 2.
"""

from __future__ import annotations


def evaluate_model(pipeline_result) -> dict:
    """Evaluate a trained model and return a metrics dictionary.

    Parameters
    ----------
    pipeline_result : pykeen.pipeline.PipelineResult
        The result object from a completed PyKEEN training run.

    Returns
    -------
    dict
        Dictionary with keys: mean_rank, mean_reciprocal_rank,
        hits_at_1, hits_at_3, hits_at_10.

    Raises
    ------
    NotImplementedError
        Until Phase 2 implementation is complete.
    """
    raise NotImplementedError("Model evaluation will be implemented in Phase 2.")


def save_results(metrics: dict, filepath=None) -> None:
    """Persist an evaluation metrics dictionary to a JSON file.

    Parameters
    ----------
    metrics : dict
        The metrics dictionary returned by :func:evaluate_model.
    filepath : str or Path, optional
        Destination path.  Defaults to outputs/results/metrics.json.

    Raises
    ------
    NotImplementedError
        Until Phase 2 implementation is complete.
    """
    raise NotImplementedError("Result saving will be implemented in Phase 2.")
