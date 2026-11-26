"""tests/test_preprocessing.py — Unit tests for preprocessing utilities."""
import pytest
import numpy as np

def test_validate_triples_basic():
    from src.preprocessing import validate_triples
    t = np.array([[0,0,1],[1,0,2],[2,1,0]])
    result = validate_triples(t)
    assert result['num_triples'] == 3
    assert result['is_valid'] == True

def test_validate_triples_with_duplicates():
    from src.preprocessing import validate_triples
    t = np.array([[0,0,1],[0,0,1],[1,0,2]])
    result = validate_triples(t)
    assert result['has_duplicates'] == True
    assert result['duplicate_count'] >= 1

def test_validate_triples_empty():
    from src.preprocessing import validate_triples
    t = np.array([]).reshape(0, 3)
    result = validate_triples(t)
    assert result['num_triples'] == 0
