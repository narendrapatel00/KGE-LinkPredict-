"""
prediction.py
-------------
Link-prediction inference helpers.

Responsibilities
----------------
- Given a trained TransE model, predict the most likely head or tail entities
  for a (?, relation, tail) or (head, relation, ?) query.
- Return ranked candidate entities with their scores.
- Provide a convenience wrapper for batch predictions.

Notes
-----
Actual implementation is deferred to Phase 3.
"""

from __future__ import annotations


def predict_tail(model, head: str, relation: str, top_k: int = 10) -> list[dict]:
    """Predict the top-K tail entities for a given (head, relation, ?) query.

    Parameters
    ----------
    model :
        A trained PyKEEN TransE model.
    head : str
        The head entity label.
    relation : str
        The relation label.
    top_k : int, optional
        Number of top candidates to return (default: 10).

    Returns
    -------
    list[dict]
        Ranked list of dicts with keys `entity` and `score`.

    Raises
    ------
    NotImplementedError
        Until Phase 3 implementation is complete.
    """
    raise NotImplementedError("Tail prediction will be implemented in Phase 3.")


def predict_head(model, relation: str, tail: str, top_k: int = 10) -> list[dict]:
    """Predict the top-K head entities for a given (?, relation, tail) query.

    Parameters
    ----------
    model :
        A trained PyKEEN TransE model.
    relation : str
        The relation label.
    tail : str
        The tail entity label.
    top_k : int, optional
        Number of top candidates to return (default: 10).

    Returns
    -------
    list[dict]
        Ranked list of dicts with keys `entity` and `score`.

    Raises
    ------
    NotImplementedError
        Until Phase 3 implementation is complete.
    """
    raise NotImplementedError("Head prediction will be implemented in Phase 3.")
