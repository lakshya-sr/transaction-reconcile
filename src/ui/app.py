#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for Multi-Source Financial Reconciliation Engine.

Features:
  - KPI Metrics (Deterministic vs AI matches, Precision/Recall, Volume stats)
  - Tab 1: Interactive PyVis Graph (Solid Grey for Deterministic, Dashed Purple for AI)
  - Tab 2: AI Inference Insights (Plotly Confidence Histogram, Truncated UTR / NLP recovery table)
  - Tab 3: Full Filterable Audit Ledger
  - Tab 4: True Suspense Ledger (Unreconciled & Partially Matched Records)
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components

from src.core.config import (
    DB_PATH,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_BANK,
    TABLE_ERP_GW_PRED,
    TABLE_GW_BANK_PRED,
    TABLE_ERP_GW_TRUE,
    TABLE_GW_BANK_TRUE,
)
from src.core.database import get_connection

# Set page configuration
st.set_page_config(
    page_title="Financial Reconciliation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_data(db_path_str: str) -> Dict[str, pd.DataFrame]:
    """Load all tables from SQLite database."""
    db_path = Path(db_path_str)
    if not db_path.exists():
        return {}

    conn = get_connection(db_path)
    data = {}
    try:
        tables = [
            (TABLE_ERP, "erp"),
            (TABLE_GATEWAY, "gw"),
            (TABLE_BANK, "bank"),
            (TABLE_ERP_GW_PRED, "pred_eg"),
            (TABLE_GW_BANK_PRED, "pred_gb"),
            (TABLE_ERP_GW_TRUE, "true_eg"),
            (TABLE_GW_BANK_TRUE, "true_gb"),
        ]
        for tbl, key in tables:
            try:
                data[key] = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
            except Exception:
                data[key] = pd.DataFrame()
    finally:
        conn.close()
    return data


def compute_kpis(data: Dict[str, pd.DataFrame]) -> Dict:
    """Calculate aggregate reconciliation metrics and performance statistics."""
    df_erp = data.get("erp", pd.DataFrame())
    df_gw = data.get("gw", pd.DataFrame())
    df_bank = data.get("bank", pd.DataFrame())
    df_peg = data.get("pred_eg", pd.DataFrame())
    df_pgb = data.get("pred_gb", pd.DataFrame())
    df_teg = data.get("true_eg", pd.DataFrame())
    df_tgb = data.get("true_gb", pd.DataFrame())

    # Deterministic vs AI Breakdown
    is_ai_eg = df_peg["matching_stage"].str.contains("AI|Fuzzy", case=False, na=False) if not df_peg.empty else pd.Series(dtype=bool)
    is_ai_gb = df_pgb["matching_stage"].str.contains("AI|Fuzzy", case=False, na=False) if not df_pgb.empty else pd.Series(dtype=bool)

    det_eg_count = int((~is_ai_eg).sum())
    ai_eg_count = int(is_ai_eg.sum())

    det_gb_count = int((~is_ai_gb).sum())
    ai_gb_count = int(is_ai_gb.sum())

    # Accuracy calculations against ground truth
    p1, r1, f1_1 = 100.0, 0.0, 0.0
    if not df_teg.empty and not df_peg.empty:
        true_eg_pairs = set(zip(df_teg["erp_id"], df_teg["gw_id"]))
        pred_eg_pairs = set(zip(df_peg["erp_order_id"], df_peg["gateway_payment_id"]))
        tp1 = len(pred_eg_pairs & true_eg_pairs)
        fp1 = len(pred_eg_pairs - true_eg_pairs)
        fn1 = len(true_eg_pairs - pred_eg_pairs)
        p1 = (tp1 / (tp1 + fp1) * 100) if (tp1 + fp1) > 0 else 0.0
        r1 = (tp1 / (tp1 + fn1) * 100) if (tp1 + fn1) > 0 else 0.0
        f1_1 = (2 * p1 * r1 / (p1 + r1)) if (p1 + r1) > 0 else 0.0

    p2, r2, f1_2 = 100.0, 0.0, 0.0
    if not df_tgb.empty and not df_pgb.empty:
        true_gb_pairs = set(zip(df_tgb["gw_id"], df_tgb["bank_id"]))
        pred_gb_pairs = set(zip(df_pgb["gateway_payment_id"], df_pgb["bank_entry_id"]))
        tp2 = len(pred_gb_pairs & true_gb_pairs)
        fp2 = len(pred_gb_pairs - true_gb_pairs)
        fn2 = len(true_gb_pairs - pred_gb_pairs)
        p2 = (tp2 / (tp2 + fp2) * 100) if (tp2 + fp2) > 0 else 0.0
        r2 = (tp2 / (tp2 + fn2) * 100) if (tp2 + fn2) > 0 else 0.0
        f1_2 = (2 * p2 * r2 / (p2 + r2)) if (p2 + r2) > 0 else 0.0

    return {
        "erp_total": len(df_erp),
        "gw_total": len(df_gw),
        "bank_total": len(df_bank),
        "total_txns": len(df_erp) + len(df_gw) + len(df_bank),
        "det_matches": det_eg_count + det_gb_count,
        "ai_matches": ai_eg_count + ai_gb_count,
        "total_pred_edges": len(df_peg) + len(df_pgb),
        "layer1_p": p1,
        "layer1_r": r1,
        "layer1_f1": f1_1,
        "layer2_p": p2,
        "layer2_r": r2,
        "layer2_f1": f1_2,
    }


from src.reporting.visualizer import (
    build_graph,
    compute_grid_layout,
    render_graph_html,
    get_erp_html,
    get_gw_html,
    get_bnk_html,
    COLOR_ERP,
    COLOR_GW,
    COLOR_BNK,
)

# Visual Constants for Deterministic vs AI
COLOR_DETERMINISTIC_EDGE = "#00FF00"  # Solid Grey
COLOR_AI_EDGE = "#00FF00"             # Dashed Bright Purple


def build_pyvis_network_from_visualizer(
    data: Dict[str, pd.DataFrame],
    max_components: int = 45,
    heading_title: str = "Reconciliation Graph Network",
) -> Path:
    """Build PyVis network from visualizer data."""
    df_erp = data.get("erp", pd.DataFrame())
    df_gw = data.get("gw", pd.DataFrame())
    df_bank = data.get("bank", pd.DataFrame())
    df_peg = data.get("pred_eg", pd.DataFrame())
    df_pgb = data.get("pred_gb", pd.DataFrame())

    G = nx.Graph()

    erp_dict = df_erp.set_index("erp_entry_id").to_dict("index") if not df_erp.empty else {}
    gw_dict = df_gw.set_index("payment_id").to_dict("index") if not df_gw.empty else {}
    bnk_dict = df_bank.set_index("bank_entry_id").to_dict("index") if not df_bank.empty else {}

    df_peg_sub = df_peg
    df_pgb_sub = df_pgb

    # 1. ERP <-> Gateway Edges
    if not df_peg_sub.empty:
        for _, row in df_peg_sub.iterrows():
            e_id = row["erp_order_id"]
            g_id = row["gateway_payment_id"]
            amt = float(row.get("allocated_amount", 0.0))
            stage = str(row.get("matching_stage", "Unknown"))
            m_type = str(row.get("match_type", "Exact"))
            score = float(row.get("confidence_score", 1.0))
            notes = str(row.get("notes", ""))

            is_ai = ("AI" in stage or "Fuzzy" in stage or "Fuzzy" in m_type)
            edge_color = COLOR_AI_EDGE if is_ai else COLOR_DETERMINISTIC_EDGE
            edge_label_tag = "🤖 AI Probabilistic" if is_ai else "⚙️ Deterministic"

            if not G.has_node(e_id):
                row_data = erp_dict.get(e_id, {})
                label = e_id.split("-")[-1] if "-" in e_id else e_id
                G.add_node(
                    e_id, group=1, color=COLOR_ERP,
                    title=get_erp_html(e_id, row_data, is_matched=True),
                    label=label, size=20,
                )

            if not G.has_node(g_id):
                row_data = gw_dict.get(g_id, {})
                label = g_id.split("-")[-1] if "-" in g_id else g_id
                G.add_node(
                    g_id, group=2, color=COLOR_GW,
                    title=get_gw_html(g_id, row_data, is_matched=True),
                    label=label, size=20,
                )

            tooltip = (
                f"<b>{edge_label_tag} Edge (ERP ↔ Gateway)</b><br>"
                f"<b>Allocated Amount:</b> ₹{amt:,.2f}<br>"
                f"<b>Matching Stage:</b> {stage}<br>"
                f"<b>Match Type:</b> {m_type}<br>"
                f"<b>Confidence Score:</b> {score:.4f}<br>"
                f"<b>Audit Notes:</b> {notes}"
            )
            G.add_edge(e_id, g_id, title=tooltip, color=edge_color, dashes=is_ai, width=5)

    # 2. Gateway <-> Bank Edges
    if not df_pgb_sub.empty:
        for _, row in df_pgb_sub.iterrows():
            g_id = row["gateway_payment_id"]
            b_id = row["bank_entry_id"]
            amt = float(row.get("allocated_amount", 0.0))
            stage = str(row.get("matching_stage", "Unknown"))
            m_type = str(row.get("match_type", "Exact"))
            score = float(row.get("confidence_score", 1.0))
            notes = str(row.get("notes", ""))

            is_ai = ("AI" in stage or "Fuzzy" in stage or "Fuzzy" in m_type)
            edge_color = COLOR_AI_EDGE if is_ai else COLOR_DETERMINISTIC_EDGE
            edge_label_tag = "🤖 AI Probabilistic" if is_ai else "⚙️ Deterministic"

            if not G.has_node(g_id):
                row_data = gw_dict.get(g_id, {})
                label = g_id.split("-")[-1] if "-" in g_id else g_id
                G.add_node(
                    g_id, group=2, color=COLOR_GW,
                    title=get_gw_html(g_id, row_data, is_matched=True),
                    label=label, size=20,
                )

            if not G.has_node(b_id):
                row_data = bnk_dict.get(b_id, {})
                label = b_id.split("-")[-1] if "-" in b_id else b_id
                G.add_node(
                    b_id, group=3, color=COLOR_BNK,
                    title=get_bnk_html(b_id, row_data, is_matched=True),
                    label=label, size=24,
                )

            tooltip = (
                f"<b>{edge_label_tag} Edge (Gateway ↔ Bank)</b><br>"
                f"<b>Allocated Amount:</b> ₹{amt:,.2f}<br>"
                f"<b>Matching Stage:</b> {stage}<br>"
                f"<b>Match Type:</b> {m_type}<br>"
                f"<b>Confidence Score:</b> {score:.4f}<br>"
                f"<b>Audit Notes:</b> {notes}"
            )
            G.add_edge(g_id, b_id, title=tooltip, color=edge_color, dashes=is_ai, width=2.5)

    # Compute layout
    compute_grid_layout(G)

    # Custom legend
    custom_legend = (
        "<span style='color:#9E9E9E;'>━━━━ Solid Grey: Deterministic Match</span> &nbsp;&nbsp;&nbsp;"
        "<span style='color:#9C27B0;'>- - - Dashed Purple: AI / Probabilistic Match</span>"
    )

    # Render HTML
    output_html_path = Path("/tmp/graph.html")
    try:
        render_graph_html(
            G,
            output_file=output_html_path,
            heading_title=heading_title,
            show_accuracy_legend=False,
            custom_edge_legend=custom_legend,
        )
    except TypeError:
        render_graph_html(
            G,
            output_file=output_html_path,
            heading_title=heading_title,
            show_accuracy_legend=False,
        )
    return output_html_path


# ==========================================
# MAIN STREAMLIT APP
# ==========================================
def main():
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60)
    st.sidebar.title("Recon Engine")
    st.sidebar.caption("Deterministic + XGBoost Residual Architecture")

    db_path = DB_PATH
    st.sidebar.write(f"📁 **Database**: `{db_path.name}`")

    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    data = load_data(str(db_path))

    if not data or data.get("erp", pd.DataFrame()).empty:
        st.error(f"No records found in database `{db_path}`. Please run `python main.py --all` first.")
        return

    kpis = compute_kpis(data)

    # Header Title
    st.markdown('<div class="main-title">Financial Reconciliation & Graph Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">End-to-End Visual Audit: Multi-Tier Deterministic Waterfall (Subset Sum) + XGBoost Residual AI Clustering</div>', unsafe_allow_html=True)

    # Top KPI Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Transactions", f"{kpis['total_txns']:,}")
    with c2:
        st.metric("Deterministic Matches", f"{kpis['det_matches']:,}")
    with c3:
        st.metric("AI / Probabilistic Matches", f"{kpis['ai_matches']:,}")
    with c4:
        st.metric("Layer 1 (ERP↔GW) Prec", f"{kpis['layer1_p']:.1f}%", f"Rec: {kpis['layer1_r']:.1f}%")
    with c5:
        st.metric("Layer 2 (GW↔Bank) Prec", f"{kpis['layer2_p']:.1f}%", f"Rec: {kpis['layer2_r']:.1f}%")

    st.markdown("---")

    # Main Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "🕸️ Tab 1: The Graph Network",
        "🧠 Tab 2: AI Inference & Insights",
        "📋 Tab 3: Full Audit Ledger",
        "⚠️ Tab 4: True Suspense Ledger",
    ])

    # =========================================================================
    # TAB 1: THE GRAPH (PyVis Network)
    # =========================================================================
    with tab1:
        st.subheader("Interactive Multi-Tier Reconciliation Network")

        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl1:
            st.markdown("""
            **Graph Legend**:
            - 🟦 **ERP Order Nodes** (Left Column) | 🟨 **Gateway Payments** (Middle Column) | 🟩 **Bank Deposits** (Right Column)
            - ➖ **Solid Grey Edge**: Deterministic Match
            - 🟪 **Dashed Purple Edge**: AI Probabilistic Match
            """)
        with col_ctrl2:
            max_c = st.slider("Max Bank Clusters", min_value=10, max_value=300, value=35, step=5)

        with st.spinner("Rendering network..."):
            tmp_path = build_pyvis_network_from_visualizer(data, max_components=max_c)
            with open(tmp_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=750, scrolling=True)

    # =========================================================================
    # TAB 2: AI INFERENCE & INSIGHTS
    # =========================================================================
    with tab2:
        st.subheader("XGBoost Residual Matching Engine Insights")

        df_pgb = data.get("pred_gb", pd.DataFrame())
        df_peg = data.get("pred_eg", pd.DataFrame())

        ai_pgb = df_pgb[df_pgb["matching_stage"].str.contains("AI|Fuzzy", case=False, na=False)].copy() if not df_pgb.empty else pd.DataFrame()
        ai_peg = df_peg[df_peg["matching_stage"].str.contains("AI|Fuzzy", case=False, na=False)].copy() if not df_peg.empty else pd.DataFrame()

        ai_combined = pd.concat([
            ai_peg.assign(Layer="Layer 1: ERP ↔ Gateway"),
            ai_pgb.assign(Layer="Layer 2: Gateway ↔ Bank"),
        ], ignore_index=True)

        col_ai1, col_ai2 = st.columns([1, 1])

        with col_ai1:
            st.markdown("#### XGBoost Probability Distribution")
            if not ai_combined.empty and "confidence_score" in ai_combined.columns:
                fig = px.histogram(
                    ai_combined,
                    x="confidence_score",
                    color="Layer",
                    nbins=25,
                    labels={"confidence_score": "Prediction Probability"},
                )
                fig.update_layout(bargap=0.1, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No probabilistic AI matches found.")

        with col_ai2:
            st.markdown("#### Noise Patterns Resolved by AI")
            st.markdown("""
            - **Truncated UTR Tokens**
            - **Missing & Corrupted Invoices**
            - **N:1 Batch Rollups**
            - **Fee / GST Rounding Drifts**
            """)
            st.metric("Total AI Matches", f"{len(ai_combined):,}")

        st.markdown("#### Detailed AI Matched Clusters")
        if not ai_combined.empty:
            st.dataframe(ai_combined, use_container_width=True, height=300)

    # =========================================================================
    # TAB 3: FULL AUDIT LEDGER
    # =========================================================================
    with tab3:
        st.subheader("Complete Reconciliation Audit Ledger")

        df_peg = data.get("pred_eg", pd.DataFrame())
        df_pgb = data.get("pred_gb", pd.DataFrame())

        unified_edges = []
        if not df_peg.empty:
            unified_edges.append(df_peg.assign(
                Layer="ERP ↔ Gateway",
                Left_Entity=df_peg["erp_order_id"],
                Right_Entity=df_peg["gateway_payment_id"],
            ))
        if not df_pgb.empty:
            unified_edges.append(df_pgb.assign(
                Layer="Gateway ↔ Bank",
                Left_Entity=df_pgb["gateway_payment_id"],
                Right_Entity=df_pgb["bank_entry_id"],
            ))

        if unified_edges:
            df_unified = pd.concat(unified_edges, ignore_index=True)

            col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
            with col_f1:
                layer_choice = st.selectbox("Filter Layer", ["All", "ERP ↔ Gateway", "Gateway ↔ Bank"])
            with col_f2:
                all_stages = ["All"] + sorted(df_unified["matching_stage"].dropna().unique().tolist())
                stage_choice = st.selectbox("Filter Stage", all_stages)
            with col_f3:
                search_query = st.text_input("🔍 Search by Entity ID", "").strip()

            filtered_df = df_unified.copy()
            if layer_choice != "All":
                filtered_df = filtered_df[filtered_df["Layer"] == layer_choice]
            if stage_choice != "All":
                filtered_df = filtered_df[filtered_df["matching_stage"] == stage_choice]
            if search_query:
                filtered_df = filtered_df[
                    filtered_df["Left_Entity"].str.contains(search_query, case=False, na=False) |
                    filtered_df["Right_Entity"].str.contains(search_query, case=False, na=False)
                ]

            st.write(f"Showing **{len(filtered_df):,}** of **{len(df_unified):,}** edges:")
            st.dataframe(filtered_df, use_container_width=True, height=450)

            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export as CSV",
                data=csv_data,
                file_name="reconciliation_audit_ledger.csv",
                mime="text/csv",
            )
        else:
            st.info("No prediction edges available.")

    # =========================================================================
    # TAB 4: TRUE SUSPENSE LEDGER
    # =========================================================================
    with tab4:
        st.subheader("True Suspense Ledger — Unreconciled & Partially Matched Records")
        st.caption("Records that don't have complete reconciliation chains (ERP → Gateway → Bank)")

        df_erp = data.get("erp", pd.DataFrame())
        df_gw = data.get("gw", pd.DataFrame())
        df_bank = data.get("bank", pd.DataFrame())
        df_peg = data.get("pred_eg", pd.DataFrame())
        df_pgb = data.get("pred_gb", pd.DataFrame())

        # Get matched IDs
        erp_with_gw = set(df_peg["erp_order_id"].dropna()) if not df_peg.empty else set()
        gw_with_erp = set(df_peg["gateway_payment_id"].dropna()) if not df_peg.empty else set()
        gw_with_bank = set(df_pgb["gateway_payment_id"].dropna()) if not df_pgb.empty else set()
        bank_with_gw = set(df_pgb["bank_entry_id"].dropna()) if not df_pgb.empty else set()
        
        all_erp_ids = set(df_erp["erp_entry_id"]) if not df_erp.empty else set()
        all_gw_ids = set(df_gw["payment_id"]) if not df_gw.empty else set()
        all_bank_ids = set(df_bank["bank_entry_id"]) if not df_bank.empty else set()
        
        # Categorize
        matched_erp_ids = erp_with_gw
        unmatched_erp_ids = all_erp_ids - erp_with_gw
        
        fully_matched_gw_ids = gw_with_erp & gw_with_bank
        partially_matched_gw_ids = gw_with_erp ^ gw_with_bank
        unmatched_gw_ids = all_gw_ids - gw_with_erp - gw_with_bank
        
        matched_bank_ids = bank_with_gw
        unmatched_bank_ids = all_bank_ids - bank_with_gw
        
        # Create DataFrames
        unmatched_erp = df_erp[df_erp["erp_entry_id"].isin(unmatched_erp_ids)].copy() if not df_erp.empty else pd.DataFrame()
        unmatched_gw = df_gw[df_gw["payment_id"].isin(unmatched_gw_ids)].copy() if not df_gw.empty else pd.DataFrame()
        partially_matched_gw = df_gw[df_gw["payment_id"].isin(partially_matched_gw_ids)].copy() if not df_gw.empty else pd.DataFrame()
        unmatched_bank = df_bank[df_bank["bank_entry_id"].isin(unmatched_bank_ids)].copy() if not df_bank.empty else pd.DataFrame()

        # Summary
        st.markdown("### 📊 Suspense Ledger Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🟦 ERP Orders")
            st.metric("Unmatched", len(unmatched_erp))
            st.metric("Matched", len(matched_erp_ids))
            if not unmatched_erp.empty:
                st.write(f"Value: **₹{unmatched_erp['gross_amount'].sum():,.2f}**")
        
        with col2:
            st.markdown("#### 🟨 Gateway Payments")
            st.metric("Unmatched", len(unmatched_gw))
            st.metric("Partially Matched", len(partially_matched_gw))
            st.metric("Fully Matched", len(fully_matched_gw_ids))
            if not unmatched_gw.empty:
                st.write(f"Unmatched: **₹{unmatched_gw['net_settled'].sum():,.2f}**")
            if not partially_matched_gw.empty:
                st.write(f"Partial: **₹{partially_matched_gw['net_settled'].sum():,.2f}**")
        
        with col3:
            st.markdown("#### 🟩 Bank Deposits")
            st.metric("Unmatched", len(unmatched_bank))
            st.metric("Matched", len(matched_bank_ids))
            if not unmatched_bank.empty:
                st.write(f"Value: **₹{unmatched_bank['credit_amount'].sum():,.2f}**")

        st.markdown("---")
        
        # Detailed views
        st.markdown("### 🔍 Detailed Views")
        
        detail_tab1, detail_tab2, detail_tab3 = st.tabs([
            "🟦 ERP Orders",
            "🟨 Gateway Payments", 
            "🟩 Bank Deposits"
        ])
        
        with detail_tab1:
            if not unmatched_erp.empty:
                st.write(f"**{len(unmatched_erp)} ERP orders with no gateway link:**")
                st.dataframe(unmatched_erp, use_container_width=True)
            else:
                st.success("✅ All ERP orders are matched!")
        
        with detail_tab2:
            gw_sub1, gw_sub2, gw_sub3 = st.tabs([
                f"Unmatched ({len(unmatched_gw)})",
                f"Partially Matched ({len(partially_matched_gw)})",
                f"Fully Matched ({len(fully_matched_gw_ids)})"
            ])
            
            with gw_sub1:
                if not unmatched_gw.empty:
                    display_gw = unmatched_gw.copy()
                    display_gw["Status"] = "No ERP or Bank link"
                    st.dataframe(display_gw, use_container_width=True)
                else:
                    st.success("✅ No unmatched gateways!")
            
            with gw_sub2:
                if not partially_matched_gw.empty:
                    display_pgw = partially_matched_gw.copy()
                    def get_status(gw_id):
                        if gw_id in gw_with_erp and gw_id not in gw_with_bank:
                            return "Has ERP → Missing Bank"
                        elif gw_id in gw_with_bank and gw_id not in gw_with_erp:
                            return "Has Bank → Missing ERP"
                        return "Unknown"
                    display_pgw["Status"] = display_pgw["payment_id"].apply(get_status)
                    st.dataframe(display_pgw, use_container_width=True)
                else:
                    st.success("✅ No partially matched gateways!")
            
            with gw_sub3:
                matched_gw = df_gw[df_gw["payment_id"].isin(fully_matched_gw_ids)].copy() if not df_gw.empty else pd.DataFrame()
                if not matched_gw.empty:
                    st.dataframe(matched_gw, use_container_width=True)
                else:
                    st.info("No fully matched gateways.")
        
        with detail_tab3:
            if not unmatched_bank.empty:
                st.write(f"**{len(unmatched_bank)} bank deposits with no gateway link:**")
                st.dataframe(unmatched_bank, use_container_width=True)
            else:
                st.success("✅ All bank deposits are matched!")


if __name__ == "__main__":
    main()
