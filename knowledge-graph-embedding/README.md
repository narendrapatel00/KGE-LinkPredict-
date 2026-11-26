# Knowledge Graph Embedding — Link Prediction

A Knowledge Graph Embedding (KGE) project that trains a **TransE** model on the **FB15k-237** dataset and exposes an interactive **Streamlit** dashboard for training, evaluation, and link prediction.

---

## Overview

Knowledge graphs represent facts as (head, relation, tail) triples. Link prediction asks: given a head entity and a relation, which tail entity completes the triple? This project uses the **TransE** model (Bordes et al., 2013) via **PyKEEN** to learn low-dimensional embeddings and answer such queries.

The Streamlit dashboard provides:
- **Dataset** page — explore FB15k-237 statistics and sample triples
- **Training** page — train the model (runs PyKEEN pipeline on CPU)
- **Evaluation** page — view Hits@K and Mean Reciprocal Rank metrics
- **Prediction** page — query the trained model for top-K link predictions

---

## Dataset: FB15k-237

| Property | Value |
|---|---|
| Entities | 14,541 |
| Relations | 237 |
| Training triples | ~272,115 |
| Validation triples | ~17,535 |
| Test triples | ~20,466 |
| **Total triples** | **~310,116** |

FB15k-237 is a challenging subset of Freebase with inverse relations removed. PyKEEN downloads it automatically on first use.

---

## Project Structure

```
knowledge-graph-embedding/
├── app.py                    # Streamlit entry point
├── config.py                 # Central configuration (paths, hyperparameters)
├── src/
│   ├── __init__.py
│   ├── dataset.py            # Dataset loading and statistics
│   ├── preprocessing.py      # Triple validation and indexing utilities
│   ├── training.py           # Model training and checkpoint management
│   ├── prediction.py         # Link prediction queries
│   └── evaluation.py         # Metric evaluation and results persistence
├── pages/
│   ├── 1_Dataset.py
│   ├── 2_Training.py
│   ├── 3_Evaluation.py
│   └── 4_Prediction.py
├── tests/
│   ├── test_imports.py
│   └── test_preprocessing.py
├── data/                     # (auto-created) Raw dataset cache
├── models/                   # (auto-created) Saved model files
├── outputs/
│   ├── checkpoints/          # PyKEEN pipeline checkpoint directory
│   └── results/              # Evaluation metrics (metrics.json)
└── .venv/                    # Python virtual environment
```

---

## Technologies

| Component | Version |
|---|---|
| Python | 3.10 |
| PyKEEN | 1.10.1 |
| PyTorch | 2.2.2+cpu |
| Streamlit | 1.35.0 |
| NumPy | 1.26.4 |
| Pandas | 2.2.2 |
| class_resolver | 0.3.10 (pinned) |

> **Important:** `class_resolver==0.3.10` is required for PyKEEN 1.10.1 compatibility. It is already pinned in this environment.

---

## Installation (Windows)

The virtual environment is already created. Simply activate it:

```powershell
cd knowledge-graph-embedding
.\.venv\Scripts\Activate.ps1
```

If you need to recreate the environment from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pykeen==1.10.1 streamlit==1.35.0 class_resolver==0.3.10
```

> **Note:** `class_resolver==0.3.10` must be installed explicitly — newer versions break PyKEEN 1.10.1 imports.

---

## Running the Dashboard

```powershell
# From knowledge-graph-embedding directory, with .venv activated:
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your browser.

---

## Training the Model

1. Launch the dashboard (`streamlit run app.py`)
2. Navigate to the **Training** page
3. Adjust hyperparameters if desired (defaults: 10 epochs, embedding dim 50)
4. Click **Train Model**

Training runs the PyKEEN `pipeline` with TransE on FB15k-237. With default settings (10 epochs, CPU), expect **15–30 minutes** depending on hardware.

The trained model is saved to `outputs/checkpoints/` and loaded automatically by the Evaluation and Prediction pages.

Default hyperparameters (`config.py`):

| Parameter | Value |
|---|---|
| `NUM_EPOCHS` | 10 |
| `EMBEDDING_DIM` | 50 |
| `BATCH_SIZE` | 512 |
| `LEARNING_RATE` | 0.01 |
| `DEVICE` | cpu |

---

## Evaluation

After training, go to the **Evaluation** page to compute:
- **Hits@1**, **Hits@3**, **Hits@10**
- **Mean Rank (MR)**
- **Mean Reciprocal Rank (MRR)**

Results are saved to `outputs/results/metrics.json` and displayed as a table and bar chart.

---

## Link Prediction

On the **Prediction** page:
1. Select a **head entity** and a **relation** from the dropdowns
2. Click **Predict** to retrieve the top-K predicted tail entities with scores

You can also predict the head entity given a relation and tail.

---

## Running Tests

```powershell
# Syntax check
.\.venv\Scripts\python.exe -m py_compile config.py tests/test_imports.py tests/test_preprocessing.py

# Unit tests (no dataset download, no training)
.\.venv\Scripts\python.exe -m pytest tests/test_imports.py tests/test_preprocessing.py -v
```

---

## Limitations

- **CPU-only**: Training is significantly slower than GPU. 10 epochs on FB15k-237 takes 15–30 minutes on a typical laptop CPU.
- **Large dataset**: FB15k-237 (~310K triples) requires a one-time download (~10 MB) on first use.
- **Memory**: Loading the full dataset and model into memory requires ~1–2 GB RAM.
- **Single model**: Only TransE is currently implemented.

---

## Future Improvements

- **GPU support**: Add CUDA device selection for faster training
- **More models**: Add RotatE, ComplEx, DistMult, ConvE via PyKEEN
- **Hyperparameter search**: Integrate PyKEEN's HPO pipeline
- **REST API**: Expose prediction endpoints via FastAPI
- **Batch predictions**: Support CSV upload for bulk link prediction
- **Explainability**: Add embedding visualization (t-SNE / UMAP)

---

## References

- Bordes, A. et al. (2013). *Translating Embeddings for Modeling Multi-relational Data.* NeurIPS.
- Ali, M. et al. (2021). *PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings.* JMLR.
- Toutanova, K. & Chen, D. (2015). *Observed versus Latent Features for Knowledge Base and Text Inference.* ACL.
