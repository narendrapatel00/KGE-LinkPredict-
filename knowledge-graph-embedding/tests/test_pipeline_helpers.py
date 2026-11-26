"""tests/test_pipeline_helpers.py — Fast unit tests for dataset, prediction, and evaluation helpers."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def synthetic_dataset_and_pipeline():
    """Create a tiny synthetic TriplesFactory and train TransE for 2 epochs."""
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    triples = np.array([
        ["alice", "knows", "bob"],
        ["bob", "knows", "carol"],
        ["carol", "works_at", "acme"],
        ["alice", "works_at", "acme"],
        ["dave", "knows", "alice"],
        ["dave", "works_at", "beta"],
        ["bob", "lives_in", "nyc"],
        ["carol", "lives_in", "nyc"],
    ])
    tf = TriplesFactory.from_labeled_triples(triples)
    dummy_dataset = SimpleNamespace(
        num_entities=tf.num_entities,
        num_relations=tf.num_relations,
        training=tf,
        validation=tf,
        testing=tf,
    )
    result = pipeline(
        training=tf,
        validation=tf,
        testing=tf,
        model="TransE",
        model_kwargs={"embedding_dim": 8},
        epochs=2,
        training_kwargs={"batch_size": 4},
        random_seed=42,
        device="cpu",
    )
    return dummy_dataset, result


def test_dataset_stats(synthetic_dataset_and_pipeline):
    from src.dataset import get_dataset_stats

    dummy_dataset, _ = synthetic_dataset_and_pipeline
    stats = get_dataset_stats(dummy_dataset)
    assert stats["num_entities"] == dummy_dataset.num_entities
    assert stats["num_relations"] == dummy_dataset.num_relations
    assert stats["num_training_triples"] == 8
    assert stats["num_validation_triples"] == 8
    assert stats["num_test_triples"] == 8
    assert stats["total_triples"] == 24


def test_dataset_sample_triples_and_labels(synthetic_dataset_and_pipeline):
    from src.dataset import get_sample_triples, get_entity_labels, get_relation_labels

    dummy_dataset, _ = synthetic_dataset_and_pipeline
    df = get_sample_triples(dummy_dataset, split="training", n=3)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["head", "relation", "tail"]
    assert len(df) == 3

    with pytest.raises(ValueError):
        get_sample_triples(dummy_dataset, split="invalid_split", n=3)

    entities = get_entity_labels(dummy_dataset)
    relations = get_relation_labels(dummy_dataset)
    assert entities == sorted(entities)
    assert "alice" in entities
    assert relations == sorted(relations)
    assert "knows" in relations


def test_predict_tail_and_head(synthetic_dataset_and_pipeline):
    from src.prediction import predict_tail, predict_head, format_predictions_df

    _, result = synthetic_dataset_and_pipeline

    tail_preds = predict_tail(result, head="alice", relation="knows", top_k=3)
    assert isinstance(tail_preds, list)
    assert len(tail_preds) == 3
    for i, item in enumerate(tail_preds, start=1):
        assert set(item.keys()) == {"entity", "score", "rank"}
        assert item["rank"] == i
        assert isinstance(item["entity"], str) and item["entity"] != ""
        assert isinstance(item["score"], float)
    # Scores should be in descending order (higher / closer to 0 is better for TransE)
    scores = [p["score"] for p in tail_preds]
    assert scores == sorted(scores, reverse=True)

    head_preds = predict_head(result, relation="knows", tail="bob", top_k=3)
    assert isinstance(head_preds, list)
    assert len(head_preds) == 3
    for i, item in enumerate(head_preds, start=1):
        assert set(item.keys()) == {"entity", "score", "rank"}
        assert item["rank"] == i

    df = format_predictions_df(tail_preds)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Rank", "Entity", "Score"]
    assert len(df) == 3

    empty_df = format_predictions_df([])
    assert isinstance(empty_df, pd.DataFrame)
    assert list(empty_df.columns) == ["Rank", "Entity", "Score"]
    assert len(empty_df) == 0


def test_predict_invalid_inputs_return_empty(synthetic_dataset_and_pipeline):
    from src.prediction import predict_tail, predict_head

    _, result = synthetic_dataset_and_pipeline
    assert predict_tail(None, head="alice", relation="knows", top_k=3) == []
    assert predict_head(None, relation="knows", tail="bob", top_k=3) == []
    assert predict_tail(result, head="nonexistent_entity", relation="knows", top_k=3) == []
    assert predict_head(result, relation="nonexistent_rel", tail="bob", top_k=3) == []


def test_evaluate_save_and_load_results(synthetic_dataset_and_pipeline, tmp_path):
    from src.evaluation import evaluate_model, save_results, load_results

    _, result = synthetic_dataset_and_pipeline
    metrics = evaluate_model(result)
    expected_keys = {"hits@1", "hits@3", "hits@10", "mean_rank", "mean_reciprocal_rank"}
    assert set(metrics.keys()) == expected_keys
    for k in expected_keys:
        assert isinstance(metrics[k], float)

    out_file = tmp_path / "metrics.json"
    save_results(metrics, filepath=out_file)
    assert out_file.exists()

    loaded = load_results(filepath=out_file)
    assert isinstance(loaded, dict)
    for k in expected_keys:
        assert loaded[k] == pytest.approx(metrics[k])
    assert "timestamp" in loaded

    assert load_results(filepath=tmp_path / "does_not_exist.json") is None
