"""
Knowledge Graph Intelligence — Link Prediction System
Streamlit frontend dashboard for Knowledge Graph Embedding with TransE on FB15k-237.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Graceful backend imports
# ---------------------------------------------------------------------------
_import_errors = []

try:
    from src.dataset import (
        load_dataset,
        get_dataset_stats,
        get_sample_triples,
        get_entity_labels,
        get_relation_labels,
    )
    _dataset_ok = True
except Exception as _e:
    _dataset_ok = False
    _import_errors.append(f"src.dataset: {_e}")

try:
    from src.preprocessing import (
        validate_triples,
        build_entity_index,
        build_relation_index,
    )
    _preprocessing_ok = True
except Exception as _e:
    _preprocessing_ok = False
    _import_errors.append(f"src.preprocessing: {_e}")

try:
    from src.training import (
        train_model,
        load_checkpoint,
        get_training_metadata,
        is_model_trained,
    )
    _training_ok = True
except Exception as _e:
    _training_ok = False
    _import_errors.append(f"src.training: {_e}")

try:
    from src.prediction import (
        predict_tail,
        predict_head,
        format_predictions_df,
    )
    _prediction_ok = True
except Exception as _e:
    _prediction_ok = False
    _import_errors.append(f"src.prediction: {_e}")

try:
    from src.evaluation import evaluate_model, save_results, load_results
    _evaluation_ok = True
except Exception as _e:
    _evaluation_ok = False
    _import_errors.append(f"src.evaluation: {_e}")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KG Intelligence — Link Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Knowledge Graph Intelligence — Link Prediction System"

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading FB15k-237 dataset…")
def _load_dataset_cached():
    if not _dataset_ok:
        return None
    try:
        return load_dataset()
    except Exception as e:
        st.error(f"Dataset load error: {e}")
        return None


@st.cache_data(show_spinner=False)
def _get_stats_cached(_ds):
    if _ds is None or not _dataset_ok:
        return {}
    try:
        return get_dataset_stats(_ds)
    except Exception as e:
        return {}


@st.cache_data(show_spinner=False)
def _get_entity_labels_cached(_ds):
    if _ds is None or not _dataset_ok:
        return []
    try:
        return sorted(get_entity_labels(_ds))
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _get_relation_labels_cached(_ds):
    if _ds is None or not _dataset_ok:
        return []
    try:
        return sorted(get_relation_labels(_ds))
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _get_sample_triples_cached(_ds, split, n):
    if _ds is None or not _dataset_ok:
        return pd.DataFrame()
    try:
        return get_sample_triples(_ds, split, n)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None
if "pipeline_result" not in st.session_state:
    st.session_state["pipeline_result"] = None
if "eval_results" not in st.session_state:
    st.session_state["eval_results"] = None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 KG Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "🗂️ Dataset Explorer",
            "🏋️ Model Training",
            "📈 Model Evaluation",
            "🔮 Link Prediction",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Show import warnings in sidebar
    if _import_errors:
        with st.expander("⚠️ Module warnings", expanded=False):
            for err in _import_errors:
                st.warning(err)

# ---------------------------------------------------------------------------
# Helper: show_metric_row
# ---------------------------------------------------------------------------

def show_metric_row(labels_values: list[tuple]):
    """Display a row of metric cards using st.columns."""
    cols = st.columns(len(labels_values))
    for col, (label, value) in zip(cols, labels_values):
        col.metric(label, value)


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == "📊 Overview":
    st.title(f"🧠 {APP_TITLE}")
    st.markdown("*A TransE-based knowledge graph embedding system for link prediction on FB15k-237.*")

    # Load dataset quietly
    ds = _load_dataset_cached()
    st.session_state["dataset"] = ds
    stats = _get_stats_cached(ds)

    # --- Top metric cards: entities / relations / triples ---
    st.subheader("📦 Dataset at a Glance")
    if stats:
        show_metric_row([
            ("Total Entities",   f"{stats.get('num_entities', '—'):,}" if isinstance(stats.get('num_entities'), int) else "—"),
            ("Total Relations",  f"{stats.get('num_relations', '—'):,}" if isinstance(stats.get('num_relations'), int) else "—"),
            ("Total Triples",    f"{stats.get('total_triples', '—'):,}" if isinstance(stats.get('total_triples'), int) else "—"),
        ])
        st.markdown("")
        show_metric_row([
            ("Training Triples",   f"{stats.get('num_training_triples', '—'):,}" if isinstance(stats.get('num_training_triples'), int) else "—"),
            ("Validation Triples", f"{stats.get('num_validation_triples', '—'):,}" if isinstance(stats.get('num_validation_triples'), int) else "—"),
            ("Test Triples",       f"{stats.get('num_test_triples', '—'):,}" if isinstance(stats.get('num_test_triples'), int) else "—"),
        ])
    else:
        st.info("Dataset stats unavailable — check src/dataset.py imports.")

    st.markdown("---")

    # --- Model status ---
    st.subheader("🤖 Model Status")
    if not _training_ok:
        st.error("Training module unavailable. Check import errors.")
    else:
        try:
            trained = is_model_trained()
        except Exception as e:
            trained = False
            st.error(f"Could not check model status: {e}")

        if trained:
            st.success("✅ A trained TransE model is available.")
            meta = None
            try:
                meta = get_training_metadata()
            except Exception:
                pass
            if meta:
                mcols = st.columns(4)
                mcols[0].metric("Epochs",        meta.get("num_epochs", "—"))
                mcols[1].metric("Embedding Dim", meta.get("embedding_dim", "—"))
                mcols[2].metric("Batch Size",    meta.get("batch_size", "—"))
                mcols[3].metric("Learning Rate", meta.get("learning_rate", "—"))
                if "timestamp" in meta:
                    st.caption(f"Trained at: {meta['timestamp']}")
        else:
            st.warning("⚠️ No trained model found. Go to **Model Training** to train one.")

    st.markdown("---")

    # --- Workflow expander ---
    with st.expander("📌 Knowledge Graph Workflow", expanded=True):
        st.markdown(
            """
            ```
            FB15k-237 Dataset
                    │
                    ▼
            Data Loading & Stats  (src/dataset.py)
                    │
                    ▼
            Preprocessing & Validation  (src/preprocessing.py)
                    │
                    ▼
            Model Training — TransE  (src/training.py)
            (embed entities & relations in ℝᵈ, minimise ‖h + r − t‖)
                    │
                    ▼
            Evaluation — MRR, Hits@K  (src/evaluation.py)
                    │
                    ▼
            Link Prediction  (src/prediction.py)
            (given h, r → predict t;  or given r, t → predict h)
            ```

            **TransE** trains entity embeddings **h**, **t** and relation embedding **r** so
            that the scoring function **‖h + r − t‖** is minimised for true triples.
            At inference time we rank all candidate entities by this score.

            | Step | Module | Description |
            |------|--------|-------------|
            | 1 | `dataset.py` | Load FB15k-237, expose splits |
            | 2 | `preprocessing.py` | Validate triples, build indices |
            | 3 | `training.py` | Train TransE via PyKEEN pipeline |
            | 4 | `evaluation.py` | Compute MRR & Hits@K |
            | 5 | `prediction.py` | Rank entities for head/tail queries |
            """
        )


# ===========================================================================
# PAGE 2 — DATASET EXPLORER
# ===========================================================================
elif page == "🗂️ Dataset Explorer":
    st.title("🗂️ Dataset Explorer")
    st.markdown("Explore the **FB15k-237** knowledge graph dataset.")

    ds = _load_dataset_cached()
    st.session_state["dataset"] = ds
    stats = _get_stats_cached(ds)

    if not stats:
        st.error("Could not load dataset statistics.")
        st.stop()

    # Metric cards
    st.subheader("Dataset Statistics")
    show_metric_row([
        ("Entities",           f"{stats.get('num_entities', 0):,}"),
        ("Relations",          f"{stats.get('num_relations', 0):,}"),
        ("Total Triples",      f"{stats.get('total_triples', 0):,}"),
    ])
    st.markdown("")
    show_metric_row([
        ("Training Triples",   f"{stats.get('num_training_triples', 0):,}"),
        ("Validation Triples", f"{stats.get('num_validation_triples', 0):,}"),
        ("Test Triples",       f"{stats.get('num_test_triples', 0):,}"),
    ])

    st.markdown("---")

    # Split selector & search
    col_left, col_right = st.columns([1, 3])
    with col_left:
        split = st.selectbox("Split", ["training", "validation", "testing"])
        n_rows = st.slider("Rows to fetch", 20, 500, 100, step=10)

    with col_right:
        search_query = st.text_input("🔍 Search (entity or relation)", placeholder="e.g. /m/06w2sn5")

    # Load triples
    df = _get_sample_triples_cached(ds, split, n_rows)

    if df is not None and not df.empty:
        # Filter by search
        if search_query.strip():
            mask = (
                df["head"].str.contains(search_query, case=False, na=False)
                | df["relation"].str.contains(search_query, case=False, na=False)
                | df["tail"].str.contains(search_query, case=False, na=False)
            )
            df = df[mask]
        st.markdown(f"**Showing {len(df)} triples from the *{split}* split**")
        st.dataframe(df, use_container_width=True, height=320)
    else:
        st.info("No triples loaded — check dataset module.")

    st.markdown("---")

    # Plotly bar chart: split sizes
    st.subheader("Split Size Comparison")
    split_data = {
        "Split": ["Training", "Validation", "Test"],
        "Triples": [
            stats.get("num_training_triples", 0),
            stats.get("num_validation_triples", 0),
            stats.get("num_test_triples", 0),
        ],
    }
    fig = px.bar(
        pd.DataFrame(split_data),
        x="Split",
        y="Triples",
        color="Split",
        color_discrete_sequence=px.colors.sequential.Blues_r[:3],
        title="Triples per Split",
        text="Triples",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(showlegend=False, height=380, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 3 — MODEL TRAINING
# ===========================================================================
elif page == "🏋️ Model Training":
    st.title("🏋️ Model Training")
    st.markdown("Configure and train a **TransE** model on FB15k-237 using PyKEEN.")

    if not _training_ok:
        st.error("Training module failed to import. Cannot train.")
        st.stop()

    # --- Existing model status ---
    try:
        already_trained = is_model_trained()
        meta = get_training_metadata() if already_trained else None
    except Exception as e:
        already_trained = False
        meta = None
        st.warning(f"Could not check training status: {e}")

    if already_trained and meta:
        st.success("✅ A trained model already exists.")
        with st.expander("Current model metadata", expanded=True):
            mc = st.columns(4)
            mc[0].metric("Model",         meta.get("model_name", "TransE"))
            mc[1].metric("Dataset",       meta.get("dataset", "FB15k-237"))
            mc[2].metric("Epochs",        meta.get("num_epochs", "—"))
            mc[3].metric("Embedding Dim", meta.get("embedding_dim", "—"))
            mc2 = st.columns(4)
            mc2[0].metric("Batch Size",    meta.get("batch_size", "—"))
            mc2[1].metric("Learning Rate", meta.get("learning_rate", "—"))
            if "timestamp" in meta:
                st.caption(f"Trained at: {meta['timestamp']}")
        st.markdown("---")
        st.markdown("You can re-train below. **This will overwrite the existing checkpoint.**")

    # --- Configuration ---
    st.subheader("⚙️ Training Configuration")
    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        num_epochs = st.slider("Epochs", min_value=1, max_value=50, value=10, step=1)
        embedding_dim = st.selectbox("Embedding Dimension", [32, 50, 64, 100, 128], index=1)

    with cfg_col2:
        batch_size = st.selectbox("Batch Size", [256, 512, 1024], index=1)
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
        )

    st.info(
        "ℹ️ Training will block the app for the duration. "
        "For FB15k-237 with 10 epochs, expect ~5–15 minutes depending on your hardware."
    )

    # --- Start training button ---
    btn_label = "🔁 Re-Train Model" if already_trained else "🚀 Start Training"
    if st.button(btn_label, type="primary"):
        with st.spinner("Training TransE model… this may take several minutes."):
            try:
                result = train_model(
                    num_epochs=num_epochs,
                    embedding_dim=embedding_dim,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    random_seed=42,
                    device="cpu",
                )
                st.session_state["pipeline_result"] = result
                st.success("✅ Training complete! Model saved to `outputs/checkpoints/`.")
                # Show updated metadata
                try:
                    new_meta = get_training_metadata()
                    if new_meta:
                        st.json(new_meta)
                except Exception:
                    pass
            except Exception as e:
                st.error(f"❌ Training failed: {e}")


# ===========================================================================
# PAGE 4 — MODEL EVALUATION
# ===========================================================================
elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")
    st.markdown(
        "Evaluate the trained TransE model using standard knowledge graph metrics: "
        "**MRR** and **Hits@K**."
    )

    if not _training_ok or not _evaluation_ok:
        st.error("Training or evaluation module unavailable.")
        st.stop()

    try:
        trained = is_model_trained()
    except Exception as e:
        trained = False
        st.error(f"Cannot check training status: {e}")

    if not trained:
        st.warning("⚠️ No trained model found. Please train a model first.")
        st.stop()

    # --- Load saved results ---
    results = None
    try:
        results = load_results()
        if results:
            st.session_state["eval_results"] = results
    except Exception as e:
        st.warning(f"Could not load saved results: {e}")

    if results:
        st.success("✅ Evaluation results loaded.")
        st.subheader("📊 Metrics")

        mrr   = results.get("mean_reciprocal_rank", results.get("mrr", None))
        h1    = results.get("hits@1",  results.get("hits_at_1",  None))
        h3    = results.get("hits@3",  results.get("hits_at_3",  None))
        h10   = results.get("hits@10", results.get("hits_at_10", None))
        mr    = results.get("mean_rank", None)

        m_cols = st.columns(4)
        m_cols[0].metric("MRR",     f"{mrr:.4f}"  if mrr  is not None else "—")
        m_cols[1].metric("Hits@1",  f"{h1:.4f}"   if h1   is not None else "—")
        m_cols[2].metric("Hits@3",  f"{h3:.4f}"   if h3   is not None else "—")
        m_cols[3].metric("Hits@10", f"{h10:.4f}"  if h10  is not None else "—")

        if mr is not None:
            st.metric("Mean Rank", f"{mr:.1f}")

        st.caption(
            "📌 **Lower Mean Rank is better.** "
            "**Higher Hits@K is better** (fraction of test triples where the correct entity ranks in top K)."
        )

        # --- Hits@K bar chart ---
        hits_data = {
            "Metric": [],
            "Value": [],
        }
        for k, v in [("Hits@1", h1), ("Hits@3", h3), ("Hits@10", h10)]:
            if v is not None:
                hits_data["Metric"].append(k)
                hits_data["Value"].append(v)

        if hits_data["Metric"]:
            fig = px.bar(
                pd.DataFrame(hits_data),
                x="Metric",
                y="Value",
                color="Value",
                color_continuous_scale="Blues",
                title="Hits@K Evaluation Metrics",
                text="Value",
                range_y=[0, 1],
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(
                showlegend=False,
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation results found. Run evaluation below.")

    st.markdown("---")

    # --- Run evaluation button ---
    if st.button("▶️ Run Evaluation", type="primary"):
        with st.spinner("Loading checkpoint and running evaluation… this may take a few minutes."):
            try:
                pipeline_result = load_checkpoint()
                if pipeline_result is None:
                    st.error("Could not load model checkpoint.")
                else:
                    metrics = evaluate_model(pipeline_result)
                    save_results(metrics)
                    st.session_state["eval_results"] = metrics
                    st.success("✅ Evaluation complete! Reload the page to view updated metrics.")
                    st.json(metrics)
            except Exception as e:
                st.error(f"❌ Evaluation failed: {e}")


# ===========================================================================
# PAGE 5 — LINK PREDICTION
# ===========================================================================
elif page == "🔮 Link Prediction":
    st.title("🔮 Link Prediction")
    st.markdown("Predict missing entities in a knowledge graph triple using the trained TransE model.")

    if not _training_ok or not _prediction_ok:
        st.error("Training or prediction module unavailable.")
        st.stop()

    # Check model availability
    try:
        trained = is_model_trained()
    except Exception:
        trained = False

    if not trained:
        st.warning("⚠️ No trained model found. Please train a model first.")
        st.stop()

    # --- Load model (cached in session state) ---
    if st.session_state.get("pipeline_result") is None:
        with st.spinner("Loading model checkpoint…"):
            try:
                st.session_state["pipeline_result"] = load_checkpoint()
            except Exception as e:
                st.error(f"Could not load checkpoint: {e}")
                st.stop()

    pipeline_result = st.session_state.get("pipeline_result")
    if pipeline_result is None:
        st.error("Model checkpoint could not be loaded.")
        st.stop()

    # --- Load dataset for dropdowns ---
    ds = _load_dataset_cached()
    entity_labels   = _get_entity_labels_cached(ds)
    relation_labels = _get_relation_labels_cached(ds)

    if not entity_labels or not relation_labels:
        st.error("Could not load entity/relation labels from dataset.")
        st.stop()

    # --- Prediction type ---
    pred_type = st.radio(
        "Prediction Type",
        ["Tail Prediction (head, relation, ?)", "Head Prediction (?, relation, tail)"],
        horizontal=True,
    )
    is_tail_pred = pred_type.startswith("Tail")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # Entity search helper
    def entity_selector(label: str, key: str):
        search = st.text_input(f"Search {label}", key=f"{key}_search", placeholder="Type to filter…")
        candidates = [e for e in entity_labels if search.lower() in e.lower()] if search.strip() else entity_labels[:500]
        if not candidates:
            candidates = entity_labels[:500]
        chosen = st.selectbox(f"Select {label}", candidates, key=f"{key}_select")
        return chosen

    with col_a:
        if is_tail_pred:
            head_entity = entity_selector("Head Entity", "head")
        else:
            tail_entity = entity_selector("Tail Entity", "tail")

    with col_b:
        relation = st.selectbox("Relation", relation_labels, key="relation_select")

    top_k = st.select_slider("Top-K results", options=[5, 10, 15, 20], value=10)

    st.markdown("---")

    if st.button("🔮 Predict", type="primary"):
        with st.spinner("Running prediction…"):
            try:
                if is_tail_pred:
                    raw_preds = predict_tail(pipeline_result, head_entity, relation, top_k)
                else:
                    raw_preds = predict_head(pipeline_result, relation, tail_entity, top_k)

                pred_df = format_predictions_df(raw_preds)

                if pred_df is not None and not pred_df.empty:
                    # Normalise column names for display
                    display_df = pred_df.copy()
                    # Ensure columns: Rank, Entity, Score
                    if "Rank" not in display_df.columns:
                        display_df.insert(0, "Rank", range(1, len(display_df) + 1))

                    st.subheader("🏆 Prediction Results")
                    st.dataframe(display_df, use_container_width=True)

                    # Plotly horizontal bar chart
                    entity_col  = "Entity" if "Entity" in display_df.columns else display_df.columns[1]
                    score_col   = "Score"  if "Score"  in display_df.columns else display_df.columns[2]

                    fig = go.Figure(go.Bar(
                        x=display_df[score_col],
                        y=display_df[entity_col],
                        orientation="h",
                        marker=dict(
                            color=display_df[score_col],
                            colorscale="Blues",
                            reversescale=False,
                        ),
                    ))
                    fig.update_layout(
                        title="Entity Scores (higher / closer to 0 = better rank)",
                        xaxis_title="Score",
                        yaxis_title="Entity",
                        height=max(300, 30 * len(display_df)),
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(
                        "📌 **Scores are raw PyKEEN TransE model scores (negative distance; higher / closer to 0 is better). "
                        "These are not probabilities.**"
                    )
                else:
                    st.warning("No predictions returned.")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")


# ===========================================================================
# PAGE 6 — ABOUT
# ===========================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ## 🎯 Project Objective

        This application demonstrates **Knowledge Graph Embedding (KGE)** for the
        **link prediction** task on the **FB15k-237** benchmark dataset.

        The goal is to train a **TransE** model that learns low-dimensional vector
        representations for entities and relations in a knowledge graph, then uses
        these embeddings to predict missing links.
        """
    )

    st.markdown("---")

    # Technologies table
    st.subheader("🛠️ Technologies")
    tech_rows = [
        {"Library": "PyKEEN",    "Version": "1.10.1",    "Role": "Knowledge graph embedding framework"},
        {"Library": "PyTorch",   "Version": "2.2.2+cpu", "Role": "Deep learning backend"},
        {"Library": "Streamlit", "Version": "1.35.0",    "Role": "Interactive web dashboard"},
        {"Library": "Plotly",    "Version": "5.22.0",    "Role": "Interactive visualisations"},
        {"Library": "Pandas",    "Version": "2.2.2",     "Role": "Data manipulation"},
        {"Library": "NumPy",     "Version": "1.26.4",    "Role": "Numerical computing"},
        {"Library": "FB15k-237", "Version": "Dataset",   "Role": "14,541 entities, 237 relations, ~310K triples"},
    ]
    st.dataframe(pd.DataFrame(tech_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # TransE explanation
    st.subheader("📐 TransE Model")
    st.markdown(
        r"""
        **TransE** (Bordes et al., 2013) is a translational embedding model.

        **Key idea:** learn embeddings **h**, **r**, **t** ∈ ℝᵈ such that for every
        true triple (h, r, t):

        $$\mathbf{h} + \mathbf{r} \approx \mathbf{t}$$

        **Training:** Minimise margin-based ranking loss:
        $$\mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h',r,t') \in S'} \left[\gamma + d(\mathbf{h}+\mathbf{r}, \mathbf{t}) - d(\mathbf{h}'+\mathbf{r}, \mathbf{t}')\right]_+$$

        where *d* is the L1 or L2 distance, γ is the margin, and *S′* is the set of
        corrupted (negative) triples.

        **Inference:** Given (h, r, ?), rank all candidate entities *t* by
        d(**h** + **r**, **t**) ascending.
        """
    )

    st.markdown("---")

    # Workflow diagram
    st.subheader("🗺️ Pipeline Overview")
    try:
        st.graphviz_chart("""
        digraph KGE {
            rankdir=LR;
            node [shape=box, style=filled, fillcolor="#E8F4F8", fontname="Helvetica"];
            edge [fontname="Helvetica", fontsize=10];

            A [label="FB15k-237\\nDataset", fillcolor="#B3D9F2"];
            B [label="Data Loading\\n(dataset.py)"];
            C [label="Preprocessing\\n(preprocessing.py)"];
            D [label="TransE Training\\n(training.py)", fillcolor="#FFE0B2"];
            E [label="Evaluation\\nMRR & Hits@K\\n(evaluation.py)", fillcolor="#C8E6C9"];
            F [label="Link Prediction\\n(prediction.py)", fillcolor="#F3E5F5"];

            A -> B -> C -> D -> E -> F;
        }
        """)
    except Exception:
        st.markdown(
            "**Data** → **Loading** → **Preprocessing** → "
            "**TransE Training** → **Evaluation** → **Link Prediction**"
        )

    st.markdown("---")

    # Future improvements
    st.subheader("🚀 Future Improvements")
    st.markdown(
        """
        - **Additional models**: RotatE, DistMult, ComplEx, RESCAL
        - **GPU training**: Enable CUDA for faster training on large datasets
        - **Hyperparameter search**: Automated optimisation via Optuna or Ray Tune
        - **Richer visualisations**: Interactive graph explorer (PyVis / D3.js)
        - **Filtered evaluation**: Implement filtered MRR/Hits@K to remove false negatives
        - **Relation-aware analytics**: Per-relation prediction performance breakdown
        - **Explainability**: Highlight subgraph evidence for predictions
        - **Export**: Download embeddings and predictions as CSV/JSON
        """
    )

    st.markdown("---")
    st.caption(
        "Built with ❤️ using [PyKEEN](https://pykeen.readthedocs.io/) and "
        "[Streamlit](https://streamlit.io/). "
        "Dataset: [FB15k-237](https://www.microsoft.com/en-us/download/details.aspx?id=52312)."
    )
