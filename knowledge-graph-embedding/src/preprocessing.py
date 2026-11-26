"""
preprocessing.py
----------------
Data cleaning and transformation utilities for the KGE pipeline.

Responsibilities
----------------
- Validate triple integrity (detect malformed or duplicate triples).
- Map entity and relation strings to integer indices.
- Provide helpers for inspecting the graph structure before training.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path so `config` can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triple validation
# ---------------------------------------------------------------------------

def validate_triples(triples_tensor: np.ndarray) -> dict:
    """Validate a triples numpy array and return statistics.

    Parameters
    ----------
    triples_tensor : np.ndarray of shape (N, 3)
        Each row is a (head_id, relation_id, tail_id) triple encoded as integers.

    Returns
    -------
    dict
        Keys:
          num_triples        – total number of triples
          has_duplicates     – True if any duplicate rows exist
          duplicate_count    – number of exact duplicate rows
          num_unique_entities – count of unique entity IDs (across head + tail cols)
          num_unique_relations – count of unique relation IDs
          is_valid           – True if the array is non-empty and has correct shape
    """
    try:
        arr = np.asarray(triples_tensor)

        if arr.ndim != 2 or arr.shape[1] != 3:
            return {
                "num_triples": 0,
                "has_duplicates": False,
                "duplicate_count": 0,
                "num_unique_entities": 0,
                "num_unique_relations": 0,
                "is_valid": False,
            }

        num_triples = arr.shape[0]

        # Duplicates: convert rows to tuples and compare counts
        unique_rows = np.unique(arr, axis=0)
        num_unique_rows = unique_rows.shape[0]
        duplicate_count = num_triples - num_unique_rows
        has_duplicates = duplicate_count > 0

        # Unique entities (head col=0, tail col=2)
        entity_ids = np.concatenate([arr[:, 0], arr[:, 2]])
        num_unique_entities = int(np.unique(entity_ids).shape[0])

        # Unique relations (col=1)
        num_unique_relations = int(np.unique(arr[:, 1]).shape[0])

        is_valid = num_triples > 0

        return {
            "num_triples": int(num_triples),
            "has_duplicates": bool(has_duplicates),
            "duplicate_count": int(duplicate_count),
            "num_unique_entities": num_unique_entities,
            "num_unique_relations": num_unique_relations,
            "is_valid": is_valid,
        }
    except Exception as exc:
        logger.error("validate_triples failed: %s", exc)
        return {
            "num_triples": 0,
            "has_duplicates": False,
            "duplicate_count": 0,
            "num_unique_entities": 0,
            "num_unique_relations": 0,
            "is_valid": False,
        }


# ---------------------------------------------------------------------------
# Entity / relation index builders
# ---------------------------------------------------------------------------

def build_entity_index(dataset) -> dict[str, int]:
    """Build a mapping from entity string labels to integer indices.

    Parameters
    ----------
    dataset : pykeen.datasets.FB15k237
        A loaded PyKEEN dataset object.

    Returns
    -------
    dict[str, int]
        Mapping of entity label -> PyKEEN integer id, derived from the
        training TriplesFactory.
    """
    try:
        return dict(dataset.training.entity_to_id)
    except Exception as exc:
        logger.error("build_entity_index failed: %s", exc)
        return {}


def build_relation_index(dataset) -> dict[str, int]:
    """Build a mapping from relation string labels to integer indices.

    Parameters
    ----------
    dataset : pykeen.datasets.FB15k237
        A loaded PyKEEN dataset object.

    Returns
    -------
    dict[str, int]
        Mapping of relation label -> PyKEEN integer id, derived from the
        training TriplesFactory.
    """
    try:
        return dict(dataset.training.relation_to_id)
    except Exception as exc:
        logger.error("build_relation_index failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Per-split triple statistics
# ---------------------------------------------------------------------------

def get_triple_stats(dataset) -> dict[str, Any]:
    """Return per-split validation statistics for all three dataset splits.

    Parameters
    ----------
    dataset : pykeen.datasets.FB15k237

    Returns
    -------
    dict
        Top-level keys: 'training', 'validation', 'testing'.
        Each value is the dict returned by :func:`validate_triples` for that
        split's mapped_triples tensor.
    """
    result: dict[str, Any] = {}
    splits = {
        "training": dataset.training,
        "validation": dataset.validation,
        "testing": dataset.testing,
    }
    for name, factory in splits.items():
        try:
            tensor = factory.mapped_triples.numpy()
            result[name] = validate_triples(tensor)
        except Exception as exc:
            logger.error("get_triple_stats failed for split '%s': %s", name, exc)
            result[name] = {
                "num_triples": 0,
                "has_duplicates": False,
                "duplicate_count": 0,
                "num_unique_entities": 0,
                "num_unique_relations": 0,
                "is_valid": False,
            }
    return result
