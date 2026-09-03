#!/usr/bin/env python3
"""
Visualization Module: Pre-Computed Grid Network Visualizer.

Provides modular graph construction, grid-based layout computation,
and PyVis HTML rendering with interactive side-panel inspection.
"""

import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

import networkx as nx
import pandas as pd
from pyvis.network import Network

from src.core.config import (
    DB_PATH,
    DATA_DIR,
    VISUALS_DIR,
    RECONCILIATION_GRAPH_PATH,
    ALL_DATA_GRAPH_PATH,
    STAGE_T1_IDENTIFIER,
    STAGE_T2_SUBSET_SUM,
    STAGE_EXACT_ERP_GW,
    STAGE_FUZZY_ERP_GW,
    STAGE_FUZZY_GW_BANK,
    STAGE_AI_CLUSTER,
    TABLE_ERP,
    TABLE_GATEWAY,
    TABLE_BANK,
    TABLE_ERP_GW_TRUE,
    TABLE_GW_BANK_TRUE,
    TABLE_ERP_GW_PRED,
    TABLE_GW_BANK_PRED,
)
from src.core.database import get_connection

# Visual Palette
COLOR_ERP = "#4285F4"       # Blue (ERP Nodes)
COLOR_GW = "#FBBC05"        # Yellow (Gateway Nodes)
COLOR_BNK = "#34A853"       # Green (Bank Nodes)

COLOR_UNMATCHED_ERP = "#AECBFA"  # Muted Blue (Unmatched ERP)
COLOR_UNMATCHED_GW = "#FDE293"   # Muted Yellow (Unmatched GW)
COLOR_UNMATCHED_BNK = "#A8DAB5"  # Muted Green (Unmatched Bank)

COLOR_EDGE_EXACT = "#34A853"   # Solid Green
COLOR_EDGE_FUZZY = "#FBBC05"   # Amber
COLOR_EDGE_BULK  = "#4285F4"   # Royal Blue
COLOR_EDGE_DEFAULT = "#9AA0A6"
COLOR_EDGE_GROUND_TRUTH = "#1E88E5"  # Blue for all ground-truth edges

STAGE_COLORS = {
    STAGE_T1_IDENTIFIER: "#4285F4",
    STAGE_T2_SUBSET_SUM: "#8E44AD",
    STAGE_EXACT_ERP_GW: "#34A853",
    STAGE_FUZZY_ERP_GW: "#FBBC05",
    STAGE_FUZZY_GW_BANK: "#FF7043",
    STAGE_AI_CLUSTER: "#00BCD4",
}

# Prediction accuracy edge colors (overlay on predicted graph)
COLOR_EDGE_CORRECT = "#00E676"  # Bright green  – true positive
COLOR_EDGE_WRONG   = "#EA4335"  # Red           – false positive


def get_erp_html(node_id: str, row: dict, is_matched: bool = True) -> str:
    status_badge = "<span style='color:#34A853;'>[RECONCILED]</span>" if is_matched else "<span style='color:#EA4335;'>[UNMATCHED]</span>"
    return (
        f"<b>Source:</b> ERP Ledger {status_badge}<br>"
        f"<b>ERP Entry ID:</b> {node_id}<br>"
        f"<b>Invoice #:</b> {row.get('invoice_number', 'N/A')}<br>"
        f"<b>Account:</b> {row.get('customer_account_id', 'N/A')}<br>"
        f"<b>Gross Amount:</b> ₹{float(row.get('gross_amount', 0)):,.2f}<br>"
        f"<b>TDS Expected:</b> ₹{float(row.get('tds_expected', 0)):,.2f}<br>"
        f"<b>Booking Date:</b> {row.get('entry_date', 'N/A')}<br>"
        f"<b>Status:</b> {row.get('status', 'N/A')}"
    )


def get_gw_html(node_id: str, row: dict, is_matched: bool = True) -> str:
    status_badge = "<span style='color:#34A853;'>[RECONCILED]</span>" if is_matched else "<span style='color:#EA4335;'>[UNMATCHED]</span>"
    return (
        f"<b>Source:</b> Gateway Settlement {status_badge}<br>"
        f"<b>Payment ID:</b> {node_id}<br>"
        f"<b>Settlement ID:</b> {row.get('settlement_id', 'N/A')}<br>"
        f"<b>Gross Amount:</b> ₹{float(row.get('gross_amount', 0)):,.2f}<br>"
        f"<b>Fee (MDR):</b> ₹{float(row.get('fee_deducted', 0)):,.2f}<br>"
        f"<b>GST on Fee:</b> ₹{float(row.get('tax_on_fee', 0)):,.2f}<br>"
        f"<b>Net Settled:</b> ₹{float(row.get('net_settled', 0)):,.2f}<br>"
        f"<b>Settled At:</b> {row.get('settled_at', 'N/A')}<br>"
        f"<b>Bank UTR:</b> {row.get('bank_utr', 'N/A')}"
    )


def get_bnk_html(node_id: str, row: dict, is_matched: bool = True) -> str:
    status_badge = "<span style='color:#34A853;'>[RECONCILED]</span>" if is_matched else "<span style='color:#EA4335;'>[UNMATCHED]</span>"
    return (
        f"<b>Source:</b> Bank Statement (CAMT.053) {status_badge}<br>"
        f"<b>Bank Entry ID:</b> {node_id}<br>"
        f"<b>Value Date:</b> {row.get('value_date', 'N/A')}<br>"
        f"<b>Credit Amount:</b> ₹{float(row.get('credit_amount', 0)):,.2f}<br>"
        f"<b>Running Balance:</b> ₹{float(row.get('running_balance', 0)):,.2f}<br>"
        f"<b>Narrative (RmtInf):</b> {row.get('remittance_info', 'N/A')}"
    )


COLOR_EDGE_GROUND_TRUTH_ONLY = "#1E88E5"  # Blue for Ground Truth only (unreconciled)

def build_graph(
    df_erp: pd.DataFrame,
    df_gw: pd.DataFrame,
    df_bank: pd.DataFrame,
    df_eg: pd.DataFrame,
    df_gb: pd.DataFrame,
    include_unmatched: bool = False,
    is_ground_truth_graph: bool = False,
    true_eg_pairs: Optional[set] = None,
    true_gb_pairs: Optional[set] = None,
    pred_eg_pairs: Optional[set] = None,
    pred_gb_pairs: Optional[set] = None,
) -> nx.Graph:
    """
    Builds a NetworkX graph with node styling, metadata, and graph edges.

    Modes:
      1. Ground Truth Graph (is_ground_truth_graph=True):
         - Correctly predicted edges (in pred pairs): Green (#00E676)
         - Ground truth only edges (not in pred pairs): Blue (#1E88E5)
      2. Prediction Graph (is_ground_truth_graph=False):
         - True Positives (in true pairs): Green (#00E676)
         - False Positives (not in true pairs): Red (#EA4335)
    """
    G = nx.Graph()

    erp_dict = df_erp.set_index("erp_entry_id").to_dict("index") if not df_erp.empty else {}
    gw_dict  = df_gw.set_index("payment_id").to_dict("index")   if not df_gw.empty  else {}
    bnk_dict = df_bank.set_index("bank_entry_id").to_dict("index") if not df_bank.empty else {}

    matched_erp         = set(df_eg["erp_order_id"].dropna())       if not df_eg.empty else set()
    matched_gw_from_erp = set(df_eg["gateway_payment_id"].dropna()) if not df_eg.empty else set()
    matched_gw_from_bnk = set(df_gb["gateway_payment_id"].dropna()) if not df_gb.empty else set()
    matched_gw          = matched_gw_from_erp | matched_gw_from_bnk
    matched_bnk         = set(df_gb["bank_entry_id"].dropna())      if not df_gb.empty else set()

    # ── 1. ERP ↔ Gateway edges ─────────────────────────────────────────────
    if not df_eg.empty:
        for _, row in df_eg.iterrows():
            erp_id = row["erp_order_id"]
            gw_id  = row["gateway_payment_id"]
            amt    = float(row.get("allocated_amount", 0))
            m_type = str(row.get("match_type", "Exact 1:1")).upper()
            stage  = str(row.get("matching_stage") or row.get("match_type") or "Unknown")

            if is_ground_truth_graph:
                is_reconciled = (
                    pred_eg_pairs is not None and
                    ((erp_id, gw_id) in pred_eg_pairs or (gw_id, erp_id) in pred_eg_pairs)
                )
                edge_color = COLOR_EDGE_CORRECT if is_reconciled else COLOR_EDGE_GROUND_TRUTH_ONLY
                accuracy_label = "✓ Reconciled (Correctly Predicted)" if is_reconciled else "Ground Truth Only (Unreconciled)"
            else:
                is_tp = (
                    true_eg_pairs is not None and
                    ((erp_id, gw_id) in true_eg_pairs or (gw_id, erp_id) in true_eg_pairs)
                )
                edge_color = COLOR_EDGE_CORRECT if is_tp else COLOR_EDGE_WRONG
                accuracy_label = "✓ Correct (TP)" if is_tp else "✗ False Positive (FP)"

            if not G.has_node(erp_id):
                row_data = erp_dict.get(erp_id, {})
                label = erp_id.split("-")[-1] if "-" in erp_id else erp_id
                G.add_node(erp_id, group=1, color=COLOR_ERP,
                           title=get_erp_html(erp_id, row_data, is_matched=True), label=label, size=20)

            if not G.has_node(gw_id):
                row_data = gw_dict.get(gw_id, {})
                label = gw_id.split("-")[-1] if "-" in gw_id else gw_id
                G.add_node(gw_id, group=2, color=COLOR_GW,
                           title=get_gw_html(gw_id, row_data, is_matched=True), label=label, size=20)

            G.add_edge(erp_id, gw_id,
                       title=f"Allocated: ₹{amt:,.2f}<br>Stage: {stage}<br>Type: {m_type}<br>Status: {accuracy_label}",
                       value=amt, color=edge_color)

    # ── 2. Gateway ↔ Bank edges ────────────────────────────────────────────
    if not df_gb.empty:
        for _, row in df_gb.iterrows():
            gw_id  = row["gateway_payment_id"]
            bnk_id = row["bank_entry_id"]
            amt    = float(row.get("allocated_amount", 0))
            m_type = str(row.get("match_type", "Exact 1:1")).upper()
            stage  = str(row.get("matching_stage") or row.get("match_type") or "Unknown")

            if is_ground_truth_graph:
                is_reconciled = (
                    pred_gb_pairs is not None and
                    ((gw_id, bnk_id) in pred_gb_pairs or (bnk_id, gw_id) in pred_gb_pairs)
                )
                edge_color = COLOR_EDGE_CORRECT if is_reconciled else COLOR_EDGE_GROUND_TRUTH_ONLY
                accuracy_label = "✓ Reconciled (Correctly Predicted)" if is_reconciled else "Ground Truth Only (Unreconciled)"
            else:
                is_tp = (
                    true_gb_pairs is not None and
                    ((gw_id, bnk_id) in true_gb_pairs or (bnk_id, gw_id) in true_gb_pairs)
                )
                edge_color = COLOR_EDGE_CORRECT if is_tp else COLOR_EDGE_WRONG
                accuracy_label = "✓ Correct (TP)" if is_tp else "✗ False Positive (FP)"

            if not G.has_node(gw_id):
                row_data = gw_dict.get(gw_id, {})
                label = gw_id.split("-")[-1] if "-" in gw_id else gw_id
                G.add_node(gw_id, group=2, color=COLOR_GW,
                           title=get_gw_html(gw_id, row_data, is_matched=True), label=label, size=20)

            if not G.has_node(bnk_id):
                row_data = bnk_dict.get(bnk_id, {})
                label = bnk_id.split("-")[-1] if "-" in bnk_id else bnk_id
                G.add_node(bnk_id, group=3, color=COLOR_BNK,
                           title=get_bnk_html(bnk_id, row_data, is_matched=True), label=label, size=20)

            G.add_edge(gw_id, bnk_id,
                       title=f"Allocated: ₹{amt:,.2f}<br>Stage: {stage}<br>Type: {m_type}<br>Status: {accuracy_label}",
                       value=amt, color=edge_color)

    # ── 3. Unmatched nodes (dimmed) ────────────────────────────────────────
    if include_unmatched:
        for erp_id, row_data in erp_dict.items():
            if erp_id not in matched_erp:
                label = erp_id.split("-")[-1] if "-" in erp_id else erp_id
                G.add_node(erp_id, group=1, color=COLOR_UNMATCHED_ERP,
                           title=get_erp_html(erp_id, row_data, is_matched=False), label=label, size=15)

        for gw_id, row_data in gw_dict.items():
            if gw_id not in matched_gw:
                label = gw_id.split("-")[-1] if "-" in gw_id else gw_id
                G.add_node(gw_id, group=2, color=COLOR_UNMATCHED_GW,
                           title=get_gw_html(gw_id, row_data, is_matched=False), label=label, size=15)

        for bnk_id, row_data in bnk_dict.items():
            if bnk_id not in matched_bnk:
                label = bnk_id.split("-")[-1] if "-" in bnk_id else bnk_id
                G.add_node(bnk_id, group=3, color=COLOR_UNMATCHED_BNK,
                           title=get_bnk_html(bnk_id, row_data, is_matched=False), label=label, size=15)

    return G



def _group_by_partner(nodes: List[str], G: nx.Graph, partner_group: int) -> List[List[str]]:
    """
    Return lists of node clusters where each cluster shares the same set of
    neighbours in `partner_group`.  Clusters are sorted by size descending so
    that larger multi-node groups appear first (allowing tight visual grouping).
    Nodes with no partner in that group form singleton clusters at the end.
    """
    partner_map: Dict[str, frozenset] = {
        n: frozenset(nb for nb in G.neighbors(n) if G.nodes[nb].get("group") == partner_group)
        for n in nodes
    }
    buckets: Dict[frozenset, List[str]] = defaultdict(list)
    for n in nodes:
        buckets[partner_map[n]].append(n)

    # Sort: non-empty partner sets first, largest clusters first, then alphabetical
    sorted_keys = sorted(
        buckets.keys(),
        key=lambda k: (len(k) == 0, -len(buckets[k]), sorted(k) if k else [])
    )
    return [sorted(buckets[k]) for k in sorted_keys]


def compute_grid_layout(
    G: nx.Graph,
    cell_width: int = 440,
    cell_height_base: int = 100,
    node_spacing: int = 90,
) -> None:
    """
    Deterministic three-column grid layout.

    Column order (left → right):  ERP  |  Gateway  |  Bank

    Placement order within each component:
      1. Gateway nodes are placed first (no prior reference needed).
      2. ERP nodes are then placed, ordered so siblings sharing the same GW
         partner(s) sit adjacent, and each ERP node is vertically centred
         opposite its GW partner(s).
      3. Bank nodes are placed last with the same centering logic relative
         to their GW partner(s).

    Cell height scales dynamically with the tallest column in each component.
    """
    components = list(nx.connected_components(G))
    if not components:
        return

    # Largest connected components first, then isolated nodes
    components.sort(key=lambda c: (-len(c), not any(G.degree(n) > 0 for n in c)))

    cols = max(1, math.ceil(math.sqrt(len(components))))
    col_y_offsets: Dict[int, float] = {c: 0.0 for c in range(cols)}

    for i, comp in enumerate(components):
        col = i % cols
        cx = float(col * cell_width)
        cy = col_y_offsets[col]

        gw_nodes  = [n for n in comp if G.nodes[n].get("group") == 2]
        erp_nodes = [n for n in comp if G.nodes[n].get("group") == 1]
        bnk_nodes = [n for n in comp if G.nodes[n].get("group") == 3]

        max_col = max(len(erp_nodes), len(gw_nodes), len(bnk_nodes), 1)
        cell_height = float(cell_height_base + max_col * node_spacing)
        total_gw_span = (len(gw_nodes) - 1) * node_spacing if gw_nodes else 0.0
        gw_start_y = cy + (cell_height - total_gw_span) / 2

        # ── Step 1: Place GW nodes evenly in the centre column ──────────────
        gw_y: Dict[str, float] = {}
        for j, node in enumerate(sorted(gw_nodes)):
            y = gw_start_y + j * node_spacing
            G.nodes[node]["x"] = cx                  # centre column
            G.nodes[node]["y"] = y
            gw_y[node] = y

        # ── Step 2: Place ERP nodes (left column, x = cx - 140) ─────────────
        #   Order: cluster siblings that share the same GW partner together.
        #   Each ERP node is centred at the average y of its GW neighbours.
        #   Siblings that share a GW get the same average → they spread evenly
        #   around that average.
        erp_clusters = _group_by_partner(erp_nodes, G, partner_group=2)
        erp_y_cursor = gw_start_y  # fallback for isolated ERPs

        for cluster in erp_clusters:
            # Compute the target y: average GW y for nodes in this cluster
            all_gw_partners = set()
            for n in cluster:
                for nb in G.neighbors(n):
                    if G.nodes[nb].get("group") == 2 and nb in gw_y:
                        all_gw_partners.add(nb)

            if all_gw_partners:
                centre_y = sum(gw_y[p] for p in all_gw_partners) / len(all_gw_partners)
            else:
                centre_y = erp_y_cursor

            # Spread the cluster symmetrically around centre_y
            n_c = len(cluster)
            span = (n_c - 1) * node_spacing
            start = centre_y - span / 2
            for k, node in enumerate(cluster):
                G.nodes[node]["x"] = cx - 140.0
                G.nodes[node]["y"] = start + k * node_spacing

            erp_y_cursor = start + n_c * node_spacing

        # ── Step 3: Place Bank nodes (right column, x = cx + 140) ───────────
        #   Same logic as ERP but on the right side, centred on GW partners.
        bnk_clusters = _group_by_partner(bnk_nodes, G, partner_group=2)
        bnk_y_cursor = gw_start_y

        for cluster in bnk_clusters:
            all_gw_partners = set()
            for n in cluster:
                for nb in G.neighbors(n):
                    if G.nodes[nb].get("group") == 2 and nb in gw_y:
                        all_gw_partners.add(nb)

            if all_gw_partners:
                centre_y = sum(gw_y[p] for p in all_gw_partners) / len(all_gw_partners)
            else:
                centre_y = bnk_y_cursor

            n_c = len(cluster)
            span = (n_c - 1) * node_spacing
            start = centre_y - span / 2
            for k, node in enumerate(cluster):
                G.nodes[node]["x"] = cx + 140.0
                G.nodes[node]["y"] = start + k * node_spacing

            bnk_y_cursor = start + n_c * node_spacing

        col_y_offsets[col] += cell_height


def render_graph_html(
    G: nx.Graph,
    output_file: Path,
    heading_title: str = "Reconciliation Graph",
    show_accuracy_legend: bool = False,
    custom_edge_legend: Optional[str] = None,
) -> Path:
    """Renders the graph into an interactive self-contained HTML page with a side panel."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    net = Network(height="100vh", width="100%", bgcolor="#1a1a1a", font_color="white", cdn_resources="remote")
    net.from_nx(G)

    net.set_options("""
    var options = {
      "physics": {
        "enabled": false
      },
      "edges": {
        "smooth": false
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "dragNodes": true
      }
    }
    """)

    net.save_graph(str(output_file))

    # Move lib directory if generated locally
    root_lib = Path.cwd() / "lib"
    target_lib = output_file.parent / "lib"
    if root_lib.exists() and root_lib.is_dir():
        if target_lib.exists():
            shutil.rmtree(target_lib)
        shutil.move(str(root_lib), str(target_lib))

    # Inject UI & Interactive Click Listener
    with open(output_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Build edge legend
    if custom_edge_legend:
        edge_legend = custom_edge_legend
    elif show_accuracy_legend:
        edge_legend = (
            "<span style='color:#00E676;'>─── Correct Prediction (TP)</span> &nbsp;"
            "<span style='color:#EA4335;'>─── False Positive (FP)</span>"
        )
    else:
        edge_legend = (
            "<span style='color:#00E676;'>─── Correctly Predicted (green)</span> &nbsp;"
            "<span style='color:#1E88E5;'>─── Ground Truth Only / Unreconciled (blue)</span>"
        )

    ui_injection = f"""
    <!-- Title Banner -->
    <div style="position: absolute; top: 15px; left: 20px; color: #fff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; z-index: 1000; background: rgba(30,30,30,0.85); padding: 10px 18px; border-radius: 8px; border: 1px solid #444;">
        <h3 style="margin: 0; font-size: 16px; font-weight: 600;">{heading_title}</h3>
        <p style="margin: 4px 0 2px 0; font-size: 12px; color: #aaa;">
            Nodes: {G.number_of_nodes()} &nbsp;|&nbsp; Edges: {G.number_of_edges()} &nbsp;|&nbsp;
            <span style="color:#4285F4;">● ERP</span> &nbsp;
            <span style="color:#FBBC05;">● Gateway</span> &nbsp;
            <span style="color:#34A853;">● Bank</span>
        </p>
        <p style="margin: 2px 0 0 0; font-size: 12px; color: #aaa;">
            {edge_legend}
        </p>
    </div>

    <!-- Floating Side Panel -->
    <div id="side-panel" style="position: absolute; top: 20px; right: 20px; width: 340px; background: #2a2a2a; color: #fff; padding: 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: none; z-index: 1000; box-shadow: 0 4px 16px rgba(0,0,0,0.6); border: 1px solid #555;">
        <h3 style="margin-top: 0; border-bottom: 1px solid #555; padding-bottom: 10px; font-size: 16px;">Record Inspection</h3>
        <div id="panel-content" style="line-height: 1.7; font-size: 13px; color: #ddd;"></div>
        <button onclick="document.getElementById('side-panel').style.display='none'" style="margin-top: 15px; padding: 8px 14px; background: #444; color: #fff; border: 1px solid #666; border-radius: 4px; cursor: pointer; width: 100%; font-weight: 500;">Close</button>
    </div>

    <!-- Click Event Script -->
    <script>
    setTimeout(() => {{
        if (typeof network !== 'undefined') {{
            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    document.getElementById('side-panel').style.display = 'block';
                    document.getElementById('panel-content').innerHTML = node.title || "No data available.";
                }} else if (params.edges.length > 0) {{
                    var edgeId = params.edges[0];
                    var edge = edges.get(edgeId);
                    document.getElementById('side-panel').style.display = 'block';
                    document.getElementById('panel-content').innerHTML = edge.title || "No data available.";
                }} else {{
                    document.getElementById('side-panel').style.display = 'none';
                }}
            }});
        }}
    }}, 1000);
    </script>
    """

    html_content = html_content.replace("</body>", ui_injection + "\n</body>")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_file



def generate_graph_visualization(
    db_path: Path = DB_PATH,
    output_file: Path = ALL_DATA_GRAPH_PATH,
    use_ground_truth: bool = False,
    include_unmatched: bool = True,
    heading_title: str = "Multi-Source Reconciliation Graph",
) -> Path:
    """
    Loads database tables, builds network graph from either ground truth
    (erp_gw_true / gw_bank_true) or predicted matches (erp_gw_pred / gw_bank_pred),
    calculates grid layout, and saves HTML.

    When rendering ground truth (use_ground_truth=True), predictions are also loaded
    to highlight correctly predicted edges in green and ground-truth only in blue.
    """
    conn = get_connection(db_path)
    try:
        df_erp  = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP}",     conn)
        df_gw   = pd.read_sql_query(f"SELECT * FROM {TABLE_GATEWAY}", conn)
        df_bank = pd.read_sql_query(f"SELECT * FROM {TABLE_BANK}",    conn)

        # Always load ground truth
        df_eg_true_raw = pd.read_sql_query(
            f"SELECT erp_id AS erp_order_id, gw_id AS gateway_payment_id FROM {TABLE_ERP_GW_TRUE}", conn
        )
        df_gb_true_raw = pd.read_sql_query(
            f"SELECT gw_id AS gateway_payment_id, bank_id AS bank_entry_id FROM {TABLE_GW_BANK_TRUE}", conn
        )

        if use_ground_truth:
            df_eg_raw = pd.read_sql_query(
                f"SELECT erp_id AS erp_order_id, gw_id AS gateway_payment_id, erp_gw_amount AS allocated_amount FROM {TABLE_ERP_GW_TRUE}", conn
            )
            df_eg_raw["match_type"] = "Ground Truth"
            df_gb_raw = pd.read_sql_query(
                f"SELECT gw_id AS gateway_payment_id, bank_id AS bank_entry_id, gw_bank_amount AS allocated_amount FROM {TABLE_GW_BANK_TRUE}", conn
            )
            df_gb_raw["match_type"] = "Ground Truth"
            df_eg = df_eg_raw
            df_gb = df_gb_raw

            # Load predictions to identify correctly reconciled edges
            df_eg_pred = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP_GW_PRED}", conn)
            df_gb_pred = pd.read_sql_query(f"SELECT * FROM {TABLE_GW_BANK_PRED}", conn)
            pred_eg_pairs: set = set(zip(df_eg_pred["erp_order_id"], df_eg_pred["gateway_payment_id"]))
            pred_gb_pairs: set = set(zip(df_gb_pred["gateway_payment_id"], df_gb_pred["bank_entry_id"]))
            true_eg_pairs = None
            true_gb_pairs = None
        else:
            df_eg_pred = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP_GW_PRED}", conn)
            df_gb_pred = pd.read_sql_query(f"SELECT * FROM {TABLE_GW_BANK_PRED}", conn)
            df_eg = df_eg_pred
            df_gb = df_gb_pred

            true_eg_pairs: set = set(zip(df_eg_true_raw["erp_order_id"], df_eg_true_raw["gateway_payment_id"]))
            true_gb_pairs: set = set(zip(df_gb_true_raw["gateway_payment_id"], df_gb_true_raw["bank_entry_id"]))
            pred_eg_pairs = None
            pred_gb_pairs = None
    finally:
        conn.close()

    G = build_graph(
        df_erp, df_gw, df_bank, df_eg, df_gb,
        include_unmatched=include_unmatched,
        is_ground_truth_graph=use_ground_truth,
        true_eg_pairs=true_eg_pairs,
        true_gb_pairs=true_gb_pairs,
        pred_eg_pairs=pred_eg_pairs,
        pred_gb_pairs=pred_gb_pairs,
    )
    compute_grid_layout(G)
    saved_path = render_graph_html(G, output_file, heading_title=heading_title, show_accuracy_legend=not use_ground_truth)
    return saved_path


def open_html_in_browser(file_path: Union[str, Path]) -> bool:
    """
    Attempts to open the generated HTML visualization in the default web browser.
    If it fails or if in a headless environment, prints the file URI to open manually.
    """
    path_obj = Path(file_path).resolve()
    file_uri = path_obj.as_uri()
    opened = False

    try:
        import webbrowser
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if has_display or os.name == "nt" or sys.platform == "darwin":
            opened = webbrowser.open(file_uri)
    except Exception:
        opened = False

    if opened:
        print(f"[🌐] Opened visualization in web browser: {path_obj.name}")
        print(f"    Link: {file_uri}")
    else:
        print(f"[📄] Visualizer saved. Open in browser: {file_uri}")

    return opened


