# KGE-LinkPredict: Knowledge Graph Embedding & Link Prediction System

A Machine Learning-based Knowledge Graph Embedding (KGE) and Link Prediction system built using the **FB15k-237** benchmark dataset and the **TransE** translation-based embedding algorithm, complete with a modular architecture and an interactive **Streamlit** dashboard.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Technology Stack](#-technology-stack)
- [Dataset Information](#-dataset-information)
- [TransE Algorithm Explanation](#-transe-algorithm-explanation)
- [Project Structure](#-project-structure)
- [Installation Instructions (Windows)](#-installation-instructions-windows)
- [How to Run](#-how-to-run)
- [Current Development Status](#-current-development-status)
- [License](#-license)

---

## 📖 Project Overview

Knowledge Graphs (KGs) represent real-world facts structured as directed graphs composed of `(head_entity, relation, tail_entity)` or `(h, r, t)` triples. However, real-world knowledge graphs are inherently incomplete.

**KGE-LinkPredict** implements Knowledge Graph Embedding techniques to map symbolic entities and relations into dense, low-dimensional continuous vector spaces. By projecting triples into vector space, the system can:
- Predict missing links: compute the likelihood of missing facts `(h, r, ?)` or `(?, r, t)`.
- Discover hidden relationships across structured entities.
- Provide a CPU-optimized pipeline accessible for academic exploration and rapid prototyping.

---

## 🎯 Objectives

1. **Embedding Representation**: Train low-dimensional continuous vector embeddings for entities and relations using the TransE model.
2. **Benchmark Evaluation**: Leverage the FB15k-237 dataset to benchmark link prediction performance without inverse-relation test leakage.
3. **Rigorous Metrics**: Measure model performance using standard evaluation metrics:
   - Mean Rank (MR)
   - Mean Reciprocal Rank (MRR)
   - Hits@1, Hits@3, and Hits@10 (both raw and filtered settings)
4. **Interactive Dashboard**: Build an intuitive Streamlit interface to explore dataset statistics, run link prediction queries interactively, and visualize embeddings.
5. **Clean Architecture**: Maintain modular code separating configuration, data loading, training, prediction, and testing.

---

## 🛠️ Technology Stack

| Layer | Component | Version / Specification |
|---|---|---|
| **Core KGE Framework** | [PyKEEN](https://github.com/pykeen/pykeen) | 1.10.1 (TransE pipeline & evaluation) |
| **Deep Learning** | [PyTorch](https://pytorch.org/) | 2.2.2 (CPU-first execution) |
| **Data Handling** | NumPy, Pandas, Scikit-Learn | NumPy 1.26.4, Pandas 2.2.2, Scikit-Learn 1.4.2 |
| **Interactive UI** | [Streamlit](https://streamlit.io/) | 1.35.0 |
| **Visualization** | Plotly & Matplotlib | Plotly 5.22.0, Matplotlib 3.9.0 |
| **Testing & QA** | Pytest & pytest-cov | Pytest 8.2.2, pytest-cov 5.0.0 |
| **Environment** | Python | 3.11 (Windows PowerShell compatible) |

---

## 📊 Dataset Information

This project uses the **FB15k-237** benchmark dataset, a curated subset of Freebase:

- **Entities**: 14,541
- **Relations**: 237
- **Total Triples**: 310,116
  - **Train split**: 272,115 triples
  - **Validation split**: 17,535 triples
  - **Test split**: 20,466 triples

### Why FB15k-237 instead of FB15k?
The original FB15k dataset suffered from test-set leakage caused by inverse relations (e.g., `has_child` vs `child_of`). Models could achieve near-perfect evaluation scores simply by memorizing inverse relation patterns rather than learning semantic representations. Toutanova and Chen (2015) constructed FB15k-237 by removing inverse and redundant relations, making it a robust and realistic benchmark for link prediction.

*Dataset files are fetched and cached dynamically by PyKEEN at runtime.*

---

## 🧠 TransE Algorithm Explanation

**TransE** (*Translating Embeddings for Modeling Multi-relational Data*, Bordes et al., NeurIPS 2013) is a foundational energy-based KGE method.

### 1. Geometric Intuition
TransE models relations as geometric translations in the embedding space $\mathbb{R}^d$. For a valid fact triple $(h, r, t)$:
$$\mathbf{h} + \mathbf{r} \approx \mathbf{t}$$

The tail entity $\mathbf{t}$ should be located close to the head entity $\mathbf{h}$ translated by the relation vector $\mathbf{r}$.

### 2. Scoring Function
The plausibility of any triple $(h, r, t)$ is measured by an energy (distance) function:
$$f(h, r, t) = - \|\mathbf{h} + \mathbf{r} - \mathbf{t}\|_{L_1 \text{ or } L_2}$$
- Triples with small translation distance receive high scores (close to $0$).
- Implausible or corrupted triples receive large negative scores.

### 3. Loss Function
TransE is trained using a margin-based pairwise ranking loss over positive triples $\mathcal{S}$ and corrupted/negative triples $\mathcal{S}'$:
$$\mathcal{L} = \sum_{(h, r, t) \in \mathcal{S}} \sum_{(h', r, t') \in \mathcal{S}'} \max\left(0, \gamma + f(h', r, t') - f(h, r, t)\right)$$
where $\gamma > 0$ is a margin hyperparameter separating positive from negative triples. Entity embeddings are constrained to unit length ($\|\mathbf{e}\|_2 = 1$) during training to prevent trivial solutions.

---

## 📁 Project Structure

```text
KGE-LinkPredict/
│
├── .gitignore                          # Root Git exclusion rules
├── README.md                           # Main project documentation
│
└── knowledge-graph-embedding/          # Core application module
    ├── app.py                          # Streamlit interactive dashboard entry point
    ├── config.py                       # Central hyper-parameters and path definitions
    ├── requirements.txt                # Python package dependencies
    ├── .gitignore                      # Module-specific ignore rules
    ├── README.md                       # Module documentation
    │
    ├── src/                            # Source code package
    │   ├── __init__.py
    │   ├── dataset.py                  # FB15k-237 loading and statistics helpers
    │   ├── preprocessing.py            # Data validation and indexing utilities
    │   ├── training.py                 # TransE training pipeline via PyKEEN
    │   ├── prediction.py               # Link prediction inference engine
    │   └── evaluation.py               # Metric calculation (MRR, MR, Hits@K)
    │
    ├── tests/                          # Automated test suite
    │   ├── __init__.py                 # Test package configuration
    │   ├── test_imports.py             # Import and dependency sanity tests
    │   ├── test_pipeline_helpers.py    # Unit tests for dataset, prediction, and evaluation
    │   └── test_preprocessing.py       # Unit tests for triple validation and indexing
    │
    ├── data/                           # Local dataset storage (git-ignored)
    ├── models/                         # Serialized model artifacts (git-ignored)
    └── outputs/                        # Runtime outputs
        ├── checkpoints/                # Model training checkpoints (git-ignored)
        └── results/                    # Metric summaries and logs (git-ignored)
```

---

## 💻 Installation Instructions (Windows)

### Prerequisites
- **Windows 10 / 11**
- **Python 3.11** (Ensure *"Add python.exe to PATH"* is checked during installation)
- **Git for Windows**

### 1. Clone the Repository
Open PowerShell and clone the project:
```powershell
git clone https://github.com/narendrapatel00/KGE-LinkPredict-.git
cd KGE-LinkPredict\knowledge-graph-embedding
```

### 2. Create a Virtual Environment
```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
> *Note: If PowerShell displays an execution policy error, enable script execution for your user:*
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run

> Ensure your virtual environment is activated before running the following commands:

### Run the Streamlit Dashboard
```powershell
streamlit run app.py
```

### Train the TransE Model *(Phase 2 Pipeline)*
```powershell
python -m src.training
```

### Evaluate Model Performance *(Phase 2 Pipeline)*
```powershell
python -m src.evaluation
```

### Run Automated Tests
```powershell
pytest tests/ -v
```

---

## 🚦 Current Development Status

| Phase | Milestone | Status | Description |
|---|---|---|---|
| **Phase 1** | Project Architecture & Setup | **Completed** | Modular directory layout, central configuration, requirements, and skeleton pipelines. |
| **Phase 2** | Model Training & Evaluation | **Completed** | Full PyKEEN TransE training pipeline on FB15k-237, checkpoint saving/loading, and Hits@K/MR/MRR metric computation. |
| **Phase 3** | Interactive Dashboard & Predictions | **Completed** | Rich multi-page Streamlit dashboard (`app.py`) with dataset explorer, model trainer, evaluation charts, and link prediction interface. |
| **Phase 4** | Testing & Quality Assurance | **Completed** | Comprehensive test suite covering imports, preprocessing validation, and pipeline helpers (`tests/`). |

---

## 📄 License
This project is developed for educational and academic research purposes.
