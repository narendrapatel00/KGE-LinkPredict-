"""
dataset.py
----------
Loads and exposes the FB15k-237 knowledge graph dataset via PyKEEN.

Public API
----------
load_dataset()      -> pykeen.datasets.FB15k237
get_dataset_stats() -> dict
get_sample_triples() -> pd.DataFrame
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset():
    """Load the FB15k-237 dataset via PyKEEN.

    PyKEEN automatically downloads and caches the dataset on first call.
    Subsequent calls are instantaneous (served from cache).

    Returns
    -------
    pykeen.datasets.FB15k237
        Dataset object with .training, .validation, .testing triple-factories.

    Raises
    ------
    RuntimeError
        If download or cache read fails.
    """
    try:
        from pykeen.datasets import FB15k237
        logger.info("Loading FB15k-237 dataset …")
        dataset = FB15k237()
        logger.info("FB15k-237 loaded successfully.")
        return dataset
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load FB15k-237 dataset: {exc}\n"
            "Check your internet connection or PyKEEN installation."
        ) from exc


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def get_dataset_stats(dataset) -> dict:
    """Return a summary dictionary of key dataset statistics.

    Parameters
    ----------
    dataset : pykeen.datasets.FB15k237
        A loaded PyKEEN dataset object.

    Returns
    -------
    dict
        Keys: num_entities, num_relations,
              num_training_triples, num_validation_triples, num_test_triples,
              total_triples.
    """
    return {
        "num_entities": dataset.num_entities,
        "num_relations": dataset.num_relations,
        "num_training_triples": dataset.training.num_triples,
        "num_validation_triples": dataset.validation.num_triples,
        "num_test_triples": dataset.testing.num_triples,
        "total_triples": (
            dataset.training.num_triples
            + dataset.validation.num_triples
            + dataset.testing.num_triples
        ),
    }


# ---------------------------------------------------------------------------
# Sample triples helper
# ---------------------------------------------------------------------------

def get_sample_triples(dataset, split: str = "training", n: int = 50) -> pd.DataFrame:
    """Return a sample of labelled triples as a DataFrame.

    Parameters
    ----------
    dataset : pykeen.datasets.FB15k237
    split   : one of "training", "validation", "testing"
    n       : number of rows to return

    Returns
    -------
    pd.DataFrame with columns [head, relation, tail]
    """
    factory_map = {
        "training": dataset.training,
        "validation": dataset.validation,
        "testing": dataset.testing,
    }
    if split not in factory_map:
        raise ValueError(f"split must be one of {list(factory_map)}, got {split!r}")

    factory = factory_map[split]

    # PyKEEN stores triples as integer tensors; convert to labels via mappings
    entity_id_to_label = {v: k for k, v in factory.entity_to_id.items()}
    relation_id_to_label = {v: k for k, v in factory.relation_to_id.items()}

    # factory.mapped_triples is shape (N, 3): [head_id, relation_id, tail_id]
    tensor = factory.mapped_triples.numpy()
    sample = tensor[:n]

    rows = []
    for h, r, t in sample:
        rows.append({
            "head": entity_id_to_label.get(int(h), str(h)),
            "relation": relation_id_to_label.get(int(r), str(r)),
            "tail": entity_id_to_label.get(int(t), str(t)),
        })
    return pd.DataFrame(rows, columns=["head", "relation", "tail"])


# ---------------------------------------------------------------------------
# Entity / relation label lists (useful for dropdowns)
# ---------------------------------------------------------------------------

def get_entity_labels(dataset) -> list[str]:
    """Return sorted list of all entity labels."""
    return sorted(dataset.training.entity_to_id.keys())


def get_relation_labels(dataset) -> list[str]:
    """Return sorted list of all relation labels."""
    return sorted(dataset.training.relation_to_id.keys())
