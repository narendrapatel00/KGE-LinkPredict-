"""
dataset.py
----------
Handles loading and accessing the FB15k-237 knowledge graph dataset via PyKEEN.

Responsibilities
----------------
- Fetch / cache the FB15k-237 dataset using PyKEEN's dataset registry.
- Expose train, validation, and test triple-factories to other modules.
- Provide basic dataset statistics (entity count, relation count, triple counts).

Notes
-----
Actual implementation is deferred to Phase 2.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Placeholder — to be implemented in Phase 2
# ---------------------------------------------------------------------------


def load_dataset():
    """Load the FB15k-237 dataset via PyKEEN.

    Returns
    -------
    pykeen.datasets.FB15k237
        The PyKEEN dataset object containing training, validation, and
        test triple-factories.

    Raises
    ------
    NotImplementedError
        Until Phase 2 implementation is complete.
    """
    raise NotImplementedError("Dataset loading will be implemented in Phase 2.")


def get_dataset_stats(dataset) -> dict:
    """Return a summary of key dataset statistics.

    Parameters
    ----------
    dataset : pykeen.datasets.FB15k237
        A loaded PyKEEN dataset object.

    Returns
    -------
    dict
        Dictionary with keys: num_entities, num_relations,
        num_training_triples, num_validation_triples, num_test_triples.

    Raises
    ------
    NotImplementedError
        Until Phase 2 implementation is complete.
    """
    raise NotImplementedError("Dataset stats will be implemented in Phase 2.")
