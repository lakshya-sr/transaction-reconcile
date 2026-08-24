#!/usr/bin/env python3
"""
Graph Visualizer: Static Pre-Computed Reconciliation Network.

Nodes are rendered instantly via NetworkX pre-computation.
Edges are styled by confidence: 
- EXACT matches = Solid Green
- BULK/FUZZY matches = Dashed Orange
"""

import sys
from pathlib import Path

import pandas as pd
import networkx as nx
from pyvis.network import Network

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import DB_PATH, DATA_DIR, TABLE_ERP, TABLE_GATEWAY, TABLE_BANK
from src.database import get_connection

def main():
    print("=" * 80)
    print("  RECONCILIATION GRAPH VISUALIZER (STYLED EDGES)")
    print("=" * 80)
    
    conn = get_connection(DB_PATH)
    try:
        df_eg = pd.read_sql_query("SELECT * FROM erp_to_gateway_edges", conn)
        df_gb = pd.read_sql_query("SELECT * FROM gateway_to_bank_edges", conn)
        
        erp_dict = pd.read_sql_query(f"SELECT * FROM {TABLE_ERP}", conn).set_index("erp_entry_id").to_dict("index")
        gw_dict = pd.read_sql_query(f"SELECT * FROM {TABLE_GATEWAY}", conn).set_index("payment_id").to_dict("index")
        bnk_dict = pd.read_sql_query(f"SELECT * FROM {TABLE_BANK}", conn).set_index("bank_entry_id").to_dict("index")
    except Exception as e:
        print("[!] Could not load tables. Ensure Phase 2 and 3 have been run.")
        return
    finally:
        conn.close()

    if df_eg.empty and df_gb.empty:
        print("[!] No graph edges found to visualize.")
        return

    G = nx.Graph()

    COLOR_ERP = "#4285F4"  # Blue
    COLOR_GW  = "#FBBC05"  # Yellow
    COLOR_BNK = "#34A853"  # Green

    # Edge Styling Definitions
    COLOR_EDGE_EXACT = "#34A853"  # Solid Green
    COLOR_EDGE_FUZZY = "#FF9900"  # Dashed Orange

    print("[*] Building nodes and edges...")

    def get_erp_html(n_id):
        d = erp_dict.get(n_id, {})
        return f"<b>ID:</b> {n_id}<br><b>Invoice:</b> {d.get('invoice_number', 'N/A')}<br><b>Gross:</b> ₹{d.get('gross_amount', 0):.2f}<br><b>Date:</b> {d.get('entry_date', 'N/A')}"

    def get_gw_html(n_id):
        d = gw_dict.get(n_id, {})
        return f"<b>ID:</b> {n_id}<br><b>Gross:</b> ₹{d.get('gross_amount', 0):.2f}<br><b>Net:</b> ₹{d.get('net_settled', 0):.2f}<br><b>Date:</b> {d.get('settled_at', 'N/A')}<br><b>UTR:</b> {d.get('bank_utr', 'N/A')}"

    def get_bnk_html(n_id):
        d = bnk_dict.get(n_id, {})
        return f"<b>ID:</b> {n_id}<br><b>Credit:</b> ₹{d.get('credit_amount', 0):.2f}<br><b>Date:</b> {d.get('value_date', 'N/A')}<br><b>Ref:</b> {str(d.get('remittance_info', 'N/A'))[:30]}..."

    # Process ERP <-> Gateway Edges
    for _, row in df_eg.iterrows():
        erp_id, gw_id, amt, m_type = row["erp_order_id"], row["gateway_payment_id"], row["allocated_amount"], row["match_type"]
        
        if not G.has_node(erp_id):
            G.add_node(erp_id, group=1, color=COLOR_ERP, title=get_erp_html(erp_id), label=erp_id.split("-")[1], size=20)
        if not G.has_node(gw_id):
            G.add_node(gw_id, group=2, color=COLOR_GW, title=get_gw_html(gw_id), label=gw_id.split("-")[1], size=20)
            
        # Determine Edge Style
        is_exact = (m_type == "EXACT")
        e_color = COLOR_EDGE_EXACT if is_exact else COLOR_EDGE_FUZZY
        
        G.add_edge(erp_id, gw_id, title=f"Allocated: ₹{amt:,.2f}<br>Type: {m_type}", value=amt, color=e_color, dashes=not is_exact)

    # Process Gateway <-> Bank Edges
    for _, row in df_gb.iterrows():
        gw_id, bnk_id, amt, m_type = row["gateway_payment_id"], row["bank_entry_id"], row["allocated_amount"], row["match_type"]
        
        if not G.has_node(gw_id):
            G.add_node(gw_id, group=2, color=COLOR_GW, title=get_gw_html(gw_id), label=gw_id.split("-")[1], size=20)
        if not G.has_node(bnk_id):
            G.add_node(bnk_id, group=3, color=COLOR_BNK, title=get_bnk_html(bnk_id), label=bnk_id.split("-")[1], size=20)
            
        # Determine Edge Style
        is_exact = (m_type == "EXACT")
        e_color = COLOR_EDGE_EXACT if is_exact else COLOR_EDGE_FUZZY
        
        G.add_edge(gw_id, bnk_id, title=f"Allocated: ₹{amt:,.2f}<br>Type: {m_type}", value=amt, color=e_color, dashes=not is_exact)

    print(f"[*] Calculating physics layout in Python...")
    
    # Pre-compute coordinates
    pos = nx.spring_layout(G, k=0.15, iterations=50, scale=2000)
    for node_id, coords in pos.items():
        G.nodes[node_id]['x'] = float(coords[0])
        G.nodes[node_id]['y'] = float(coords[1])

    print(f"[*] Rendering static HTML for {G.number_of_nodes()} nodes and {G.number_of_edges()} edges...")
    
    net = Network(height="100vh", width="100%", bgcolor="#1a1a1a", font_color="white", cdn_resources="remote")
    net.from_nx(G)
    
    # Notice: Removed the `"color": {"inherit": "from"}` block so our custom edge colors render correctly
    net.set_options("""
    var options = {
      "physics": {
        "enabled": false
      },
      "edges": {
        "smooth": {
          "type": "continuous"
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "dragNodes": true
      }
    }
    """)
    
    output_file = str(DATA_DIR / "reconciliation_graph.html")
    net.save_graph(output_file)

    # Inject UI and Click Event Listener
    with open(output_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    ui_injection = """
    <!-- Floating Side Panel & Legend -->
    <div style="position: absolute; top: 20px; left: 20px; background: #2a2a2a; color: #fff; padding: 15px; border-radius: 8px; font-family: sans-serif; font-size: 13px; z-index: 1000; border: 1px solid #444;">
        <h4 style="margin: 0 0 10px 0;">Edge Legend</h4>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 25px; height: 3px; background: #34A853; margin-right: 10px;"></div> Exact Match
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 25px; height: 0px; border-top: 3px dashed #FF9900; margin-right: 10px;"></div> Bulk/Fuzzy Match
        </div>
    </div>
    
    <div id="side-panel" style="position: absolute; top: 20px; right: 20px; width: 320px; background: #2a2a2a; color: #fff; padding: 20px; border-radius: 8px; font-family: sans-serif; display: none; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.5); border: 1px solid #444;">
        <h3 style="margin-top: 0; border-bottom: 1px solid #555; padding-bottom: 10px;">Node Details</h3>
        <div id="panel-content" style="line-height: 1.6; font-size: 14px;"></div>
        <button onclick="document.getElementById('side-panel').style.display='none'" style="margin-top: 15px; padding: 8px 12px; background: #444; color: #fff; border: none; border-radius: 4px; cursor: pointer; width: 100%;">Close</button>
    </div>

    <!-- Click Listener Script -->
    <script>
    setTimeout(() => {
        if (typeof network !== 'undefined') {
            network.on("click", function (params) {
                if (params.nodes.length > 0) {
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    document.getElementById('side-panel').style.display = 'block';
                    document.getElementById('panel-content').innerHTML = node.title || "No data available.";
                } else if (params.edges.length > 0) {
                    var edgeId = params.edges[0];
                    var edge = edges.get(edgeId);
                    document.getElementById('side-panel').style.display = 'block';
                    document.getElementById('panel-content').innerHTML = edge.title || "No data available.";
                } else {
                    document.getElementById('side-panel').style.display = 'none';
                }
            });
        }
    }, 1000);
    </script>
    """
    
    html_content = html_content.replace("</body>", ui_injection + "\n</body>")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"[✔] Visualization saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()