"""
app.py
------
Streamlit dashboard entry-point for the Knowledge Graph Embedding &
Link Prediction project.

Usage
-----
    streamlit run app.py

Pages (planned)
---------------
1. Overview    : Dataset statistics and project description.
2. Training    : Trigger / monitor model training and view loss curves.
3. Evaluation  : Display MRR, Hits@K, and Mean Rank metrics.
4. Prediction  : Interactive link-prediction query interface.

Notes
-----
Full implementation is deferred to Phase 3.  This file currently renders
a placeholder landing page so that streamlit run app.py succeeds.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KGE Link Prediction",
    page_icon="🔗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Landing page (Phase 1 placeholder)
# ---------------------------------------------------------------------------
st.title("🔗 Knowledge Graph Embedding & Link Prediction")
st.markdown(
    """
    **Dataset:** FB15k-237 &nbsp;|&nbsp;
    **Model:** TransE &nbsp;|&nbsp;
    **Framework:** PyKEEN + PyTorch
    """
)

st.info(
    "🚧 **Dashboard under construction.** "
    "Full functionality will be available after Phase 2 (training) and "
    "Phase 3 (dashboard) are complete.",
    icon="🚧",
)

st.markdown("---")
st.markdown("### Planned Sections")
sections = {
    "📊 Overview":    "Dataset statistics and project summary.",
    "🏋️ Training":   "Run TransE training and monitor loss curves.",
    "📈 Evaluation":  "MRR, Hits@1 / @3 / @10, Mean Rank.",
    "🔮 Prediction":  "Interactive (head, relation, ?) / (?, relation, tail) queries.",
}
for title, desc in sections.items():
    st.markdown(f"**{title}** — {desc}")
