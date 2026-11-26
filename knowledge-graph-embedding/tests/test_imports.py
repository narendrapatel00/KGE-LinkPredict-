"""tests/test_imports.py — Test that all project modules import without errors."""
import pytest

def test_config_imports():
    import config
    assert hasattr(config, 'PROJECT_ROOT')
    assert hasattr(config, 'DATASET_NAME')
    assert config.DATASET_NAME == 'FB15k-237'
    assert config.MODEL_NAME == 'TransE'
    assert config.NUM_EPOCHS == 10

def test_src_dataset_imports():
    from src import dataset
    assert callable(dataset.load_dataset)
    assert callable(dataset.get_dataset_stats)
    assert callable(dataset.get_sample_triples)
    assert callable(dataset.get_entity_labels)
    assert callable(dataset.get_relation_labels)

def test_src_preprocessing_imports():
    from src import preprocessing
    assert callable(preprocessing.validate_triples)
    assert callable(preprocessing.build_entity_index)
    assert callable(preprocessing.build_relation_index)

def test_src_training_imports():
    from src import training
    assert callable(training.train_model)
    assert callable(training.load_checkpoint)
    assert callable(training.is_model_trained)

def test_src_prediction_imports():
    from src import prediction
    assert callable(prediction.predict_tail)
    assert callable(prediction.predict_head)

def test_src_evaluation_imports():
    from src import evaluation
    assert callable(evaluation.evaluate_model)
    assert callable(evaluation.save_results)
    assert callable(evaluation.load_results)

def test_pykeen_core_imports():
    from pykeen.datasets import FB15k237
    from pykeen.pipeline import pipeline
    from pykeen.predict import predict_target

def test_preprocessing_validate_triples():
    import numpy as np
    from src.preprocessing import validate_triples
    triples = np.array([[0,0,1],[1,0,2],[0,1,2]])
    result = validate_triples(triples)
    assert isinstance(result, dict)
    assert 'num_triples' in result
    assert result['num_triples'] == 3
    assert 'is_valid' in result

def test_load_checkpoint_returns_none_when_not_trained():
    from src.training import load_checkpoint, is_model_trained
    # If no model trained yet, load_checkpoint should return None (not raise)
    if not is_model_trained():
        result = load_checkpoint()
        assert result is None

def test_load_results_returns_none_when_not_evaluated():
    from src.evaluation import load_results
    import config
    # If results file does not exist, should return None
    if not config.RESULTS_FILE.exists():
        result = load_results()
        assert result is None
