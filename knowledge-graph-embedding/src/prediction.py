"""
prediction.py
-------------
Link-prediction inference helpers.

Responsibilities
----------------
- Given a trained TransE pipeline result (or LoadedCheckpoint), predict the
  most likely head or tail entities for a (head, relation, ?) or
  (?, relation, tail) query.
- Return ranked candidate entities with their scores.
- Provide a convenience formatter for displaying results as a DataFrame.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path so `config` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation helper
# ---------------------------------------------------------------------------

def _validate_prediction_inputs(
    pipeline_result,
    relation: str,
    entity: str,
    top_k: int,
) -> bool:
    """Validate that pipeline_result, relation, entity, and top_k are usable."""
    if pipeline_result is None:
        logger.warning("Prediction called with pipeline_result=None")
        return False
    if not hasattr(pipeline_result, "model") or not hasattr(pipeline_result, "training"):
        logger.warning("pipeline_result missing .model or .training attribute")
        return False
    if pipeline_result.model is None or pipeline_result.training is None:
        logger.warning("pipeline_result.model or .training is None")
        return False
    if not isinstance(relation, str) or not relation.strip():
        logger.warning("Invalid relation: %r", relation)
        return False
    if not isinstance(entity, str) or not entity.strip():
        logger.warning("Invalid entity: %r", entity)
        return False
    if not isinstance(top_k, int) or top_k <= 0:
        logger.warning("Invalid top_k: %r", top_k)
        return False

    tf = pipeline_result.training
    if entity not in tf.entity_to_id:
        logger.warning("Entity %r not found in training TriplesFactory", entity)
        return False
    if relation not in tf.relation_to_id:
        logger.warning("Relation %r not found in training TriplesFactory", relation)
        return False

    return True


# ---------------------------------------------------------------------------
# Tail prediction
# ---------------------------------------------------------------------------

def predict_tail(
    pipeline_result,
    head: str,
    relation: str,
    top_k: int = 10,
) -> list[dict]:
    """Predict the top-K tail entities for a (head, relation, ?) query.

    Parameters
    ----------
    pipeline_result : pykeen.pipeline.PipelineResult or LoadedCheckpoint
        A trained pipeline result or loaded checkpoint exposing ``.model``
        and ``.training``.
    head : str
        Head entity label (must exist in the training TriplesFactory).
    relation : str
        Relation label (must exist in the training TriplesFactory).
    top_k : int, optional
        Number of top candidates to return (default: 10).

    Returns
    -------
    list[dict]
        Ranked list of dicts with keys ``entity`` (str), ``score`` (float),
        and ``rank`` (int), sorted descending by score (higher / closer to 0
        is better for PyKEEN TransE). Returns an empty list on invalid input
        or error.
    """
    if not _validate_prediction_inputs(pipeline_result, relation, head, top_k):
        return []

    try:
        from pykeen.predict import predict_target

        # predict_target returns a TargetPredictions object; .df is a DataFrame
        # with columns ['tail_id', 'score', 'tail_label'] sorted descending by score
        df: pd.DataFrame = predict_target(
            model=pipeline_result.model,
            head=head,
            relation=relation,
            triples_factory=pipeline_result.training,
        ).df

        df = df.sort_values(by="score", ascending=False).head(top_k).reset_index(drop=True)

        results = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            entity_label = str(row.get("tail_label") or row.get("tail_id", ""))
            score = float(row.get("score", 0.0))
            results.append({"entity": entity_label, "score": score, "rank": rank})

        return results

    except Exception as exc:
        logger.error("predict_tail failed (head=%r, relation=%r): %s", head, relation, exc)
        return []


# ---------------------------------------------------------------------------
# Head prediction
# ---------------------------------------------------------------------------

def predict_head(
    pipeline_result,
    relation: str,
    tail: str,
    top_k: int = 10,
) -> list[dict]:
    """Predict the top-K head entities for a (?, relation, tail) query.

    Parameters
    ----------
    pipeline_result : pykeen.pipeline.PipelineResult or LoadedCheckpoint
        A trained pipeline result or loaded checkpoint exposing ``.model``
        and ``.training``.
    relation : str
        Relation label (must exist in the training TriplesFactory).
    tail : str
        Tail entity label (must exist in the training TriplesFactory).
    top_k : int, optional
        Number of top candidates to return (default: 10).

    Returns
    -------
    list[dict]
        Ranked list of dicts with keys ``entity`` (str), ``score`` (float),
        and ``rank`` (int), sorted descending by score (higher / closer to 0
        is better for PyKEEN TransE). Returns an empty list on invalid input
        or error.
    """
    if not _validate_prediction_inputs(pipeline_result, relation, tail, top_k):
        return []

    try:
        from pykeen.predict import predict_target

        # predict_target returns a TargetPredictions object; .df is a DataFrame
        # with columns ['head_id', 'score', 'head_label'] sorted descending by score
        df: pd.DataFrame = predict_target(
            model=pipeline_result.model,
            relation=relation,
            tail=tail,
            triples_factory=pipeline_result.training,
        ).df

        df = df.sort_values(by="score", ascending=False).head(top_k).reset_index(drop=True)

        results = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            entity_label = str(row.get("head_label") or row.get("head_id", ""))
            score = float(row.get("score", 0.0))
            results.append({"entity": entity_label, "score": score, "rank": rank})

        return results

    except Exception as exc:
        logger.error("predict_head failed (relation=%r, tail=%r): %s", relation, tail, exc)
        return []


# ---------------------------------------------------------------------------
# DataFrame formatter
# ---------------------------------------------------------------------------

def format_predictions_df(predictions: list[dict]) -> pd.DataFrame:
    """Convert the raw prediction list into a display-ready DataFrame.

    Parameters
    ----------
    predictions : list[dict]
        Output of :func:`predict_tail` or :func:`predict_head`.

    Returns
    -------
    pd.DataFrame
        Columns: Rank, Entity, Score.  Returns an empty DataFrame if the
        input list is empty or malformed.
    """
    if not predictions:
        return pd.DataFrame(columns=["Rank", "Entity", "Score"])

    try:
        rows = [
            {
                "Rank": p.get("rank", i + 1),
                "Entity": p.get("entity", ""),
                "Score": round(float(p.get("score", 0.0)), 6),
            }
            for i, p in enumerate(predictions)
        ]
        return pd.DataFrame(rows, columns=["Rank", "Entity", "Score"])
    except Exception as exc:
        logger.error("format_predictions_df failed: %s", exc)
        return pd.DataFrame(columns=["Rank", "Entity", "Score"])
