# Hybrid Multi-Source Financial Reconciliation System

Reconciles financial records in a three way situation of merchant ERP, payment gateway, and bank. Uses deterministic engine and AI based clustering to reconcile ERP receipts, payment gateway settlements, and bank statements. Supports one-to-one, one-to-many, many-to-one, and many-to-many relationships between records for both ERP-Gateway and Gateway-Bank reconcilaiation.


# Running

The application is a cli tool. There is a dashboard but it is read-only, no modification to data can be performed from the dashboard.

## Github Codespaces

Use `Ctrl + Click` or `Right click > Open in New Tab` so you can continue to read the instruction here.

<a href="https://codespaces.new/lakshya-sr/transaction-reconcile" target="_blank" rel="noopener noreferrer">
  <img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub Codespaces" style="max-width: 200px;">
</a>

After opening wait for the setup to run and terminal to appear, then run the program using commands provided below. 

```bash
python main.py --all
```
then
```bash
python main.py --dashboard
```

After running the dashboard, switch to `Ports` tab and move pointer to the https link which will show three buttons, the middle button is for opening in browser, click that to open the dashboard.

## Windows

```powershell
# Clone repository
git clone https://github.com/lakshya-sr/transaction-reconcile.git
cd transaction-reconcile

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py --all

# Launch dashboard
python main.py --dashboard
```

## Linux

### 1. `pip` based

```bash
# Clone repository
git clone https://github.com/lakshya-sr/transaction-reconcile.git
cd transaction-reconcile

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py --all

# Launch dashboard
python main.py --dashboard
```

### 2. `uv` based

```bash
# Clone repository
git clone https://github.com/lakshya-sr/transaction-reconcile.git
cd transaction-reconcile

# Install dependencies with uv
uv pip install -r requirements.txt

# Run full pipeline
uv run main.py --all

# Launch dashboard
uv run main.py --dashboard
```

## MacOS (not tested)

```bash
# Clone repository
git clone https://github.com/lakshya-sr/transaction-reconcile.git
cd transaction-reconcile

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py --all

# Launch dashboard
python main.py --dashboard
```

## Available Commands

| Command | Description |
|---------|-------------|
| `python main.py --all` | Run entire pipeline |
| `python demo.py` | Step-by-step demo |
| `python main.py --evaluate` | Evaluate accuracy |
| `python main.py --benchmark` | Performance profiling |
| `python main.py --dashboard` | Launch Streamlit UI |
| `python main.py --generate` | Generate synthetic data |
| `python main.py --setup-db` | Initialize database |
| `python main.py --match` | Run deterministic matching |
| `python main.py --infer` | Run AI inference |

# Architecture Overview

The reconciliation pipeline processes transactions through sequential stages, each handling increasingly complex matching scenarios.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYNTHETIC DATA GENERATION                    │
│  Simulates 3 sources with realistic noise (fees, delays, etc.) │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATABASE INGESTION                         │
│  Loads raw records into SQLite with ground truth tables         │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DETERMINISTIC MATCHING                        │
│  • Exact identifier match (invoice, UTR, settlement ID)         │
│  • Subset sum for N:1 batch settlements                         │
│  • Connected-component sum balancing                            │
│  • Reserve split detection (85/15 MAIN/RSV)                     │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AI INFERENCE                              │
│  • Candidate cluster generation                                 │
│  • 11-feature extraction (amount, time, fuzzy scores)           │
│  • XGBoost scoring with calibrated threshold                    │
│  • Greedy conflict-free assignment                              │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        EVALUATION                               │
│  • Precision / Recall / F1 against ground truth                 │
│  • Layer 1: ERP↔Gateway | Layer 2: Gateway↔Bank                 │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   VISUALIZATION & REPORTING                     │
│  • Streamlit dashboard with graph network                       │
│  • Unreconciled records viewer                                  │
│  • Performance benchmark                                        │
└─────────────────────────────────────────────────────────────────┘
```
## Stage Details

### 1. Data Generation (`src/simulation/generate_data.py`)
Creates synthetic ERP, Gateway, and Bank records with ground truth edges. Injects realistic noise: MDR fees, GST, delayed settlements, corrupted UTRs, missing invoices, batch settlements, and reserve splits.

### 2. Deterministic Matching (`src/deterministic/matcher.py`)
Rule-based matching that catches ~70% of transactions:
- **Invoice matching**: Exact invoice number lookup with connected-component sum balancing
- **UTR matching**: Full and truncated UTR prefix matching
- **Settlement ID**: Direct identifier match
- **Subset sum**: Finds N:1 and 1:N combinations where amounts sum correctly
- **Amount + temporal**: Fallback for records with no identifiers

### 3. AI Inference (`src/ai/`)
XGBoost handles residual unmatched records:
- **Candidate generation**: Blocks of potential matches within time/amount windows
- **Feature extraction**: 11 features including amount diffs, time deltas, UTR/invoice fuzzy scores
- **Model scoring**: Predicts match probability with 98% precision threshold
- **Assignment**: Greedy conflict-free matching


### 4. Evaluation (`src/reporting/evaluate.py`)
Compares predicted edges against ground truth:
- **Layer 1**: ERP↔Gateway precision/recall/F1
- **Layer 2**: Gateway↔Bank precision/recall/F1
- **False positive tracking**: Identifies incorrect matches by stage

### 5. Visualization (`src/reporting/visualizer.py` and `src/ui/`)
- Interactive graph showing ERP → Gateway → Bank chains
- Solid edges = deterministic, dashed = AI
- Suspense ledger for unreconciled records
- Performance benchmark with per-stage timing

### 6. XGBoost Features

#### Gateway ↔ Bank Model

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `cluster_size` | Numeric | Number of gateway payments in candidate cluster |
| 2 | `amount_diff_abs` | Numeric | Absolute difference between bank credit and sum of gateway net settlements |
| 3 | `amount_diff_pct` | Numeric | Percentage amount difference relative to bank credit |
| 4 | `time_delta_min_hrs` | Numeric | Minimum time gap between bank deposit and gateway settlements |
| 5 | `time_delta_max_hrs` | Numeric | Maximum time gap between bank deposit and gateway settlements |
| 6 | `time_span_hrs` | Numeric | Time span between earliest and latest gateway in cluster |
| 7 | `best_utr_fuzz` | Numeric (0-1) | Best fuzzy similarity score between gateway UTR and bank remittance text |
| 8 | `utr_prefix_match` | Binary (0/1) | Whether UTR prefix (6-8 chars) appears in remittance info |
| 9 | `best_invoice_fuzz` | Numeric (0-1) | Best fuzzy similarity between invoice and remittance text |
| 10 | `invoice_prefix_match` | Binary (0/1) | Whether invoice number appears in remittance info |
| 11 | `is_single_day` | Binary (0/1) | Whether all transactions in cluster occurred on same day |

#### ERP ↔ Gateway Model

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `cluster_size` | Numeric | Number of ERP orders in candidate cluster |
| 2 | `gross_diff_abs` | Numeric | Absolute difference between gateway gross and sum of ERP gross amounts |
| 3 | `gross_diff_pct` | Numeric | Percentage amount difference relative to gateway gross |
| 4 | `time_delta_min_hrs` | Numeric | Minimum time gap between gateway settlement and ERP entries |
| 5 | `time_delta_max_hrs` | Numeric | Maximum time gap between gateway settlement and ERP entries |
| 6 | `time_span_hrs` | Numeric | Time span between earliest and latest ERP in cluster |
| 7 | `best_invoice_fuzz` | Numeric (0-1) | Best fuzzy similarity between ERP invoice and gateway invoices |
| 8 | `invoice_prefix_match` | Binary (0/1) | Whether invoice prefix (6 chars) matches between ERP and gateway |
| 9 | `invoice_token_overlap` | Numeric (0-1) | Jaccard token overlap between invoice strings |
| 10 | `exact_invoice_match` | Binary (0/1) | Whether exact invoice match exists |
| 11 | `is_single_day` | Binary (0/1) | Whether all transactions in cluster occurred on same day |


# Performance

## Prediction

| Reconciliation Graph Layer | Precision | Recall | F1 Score |
| -------------------------- | --------: | -----: | -------: |
| Layer 1: ERP ↔ Gateway     |    100.0% |  99.7% |    99.8% |
| Layer 2: Gateway ↔ Bank    |    100.0% | 100.0% |   100.0% |

Note: Yes the data is real, it is copied straight from a run of the pipeline. You can run and see the data yourself. Recall for Layer 1 may be lower as it is dependent on the particular case talked about in the video.

## Throughput

`100 txns/sec` is the throughput, it varies on different machines, as I alluded to in my video.

| Stage                     | Latency (ms) | Share (%) | Peak Memory |
| ------------------------- | -----------: | --------: | ----------: |
| 1. Data Generation        |    797.16 ms |      7.0% |     1.21 MB |
| 2. DB Ingestion           |   1122.71 ms |      9.9% |     1.01 MB |
| 3. Deterministic Matching |   1765.46 ms |     15.6% |     1.38 MB |
| 4. AI Inference           |   7652.83 ms |     67.5% |    16.46 MB |


# Gallery

![Reconciled records graph](img/rec-graph.png)

![Partially matched gateway transactions](img/matched.png)

# FAQ

## 1. Why does the engine have so many stages?

The stages have progressively more complex rules to match different situations. The deterministic layer is required to provide fast accurate matching for easy scenarios, and the XGBoost based AI layer provides more intelligent matching based on multiple inputs.

## 2. Why use a deterministic layer?

Because it is fast and reliable and explainable. Using the XGBoost model on all the data would be prohibitively expensive, so a fast stage is required to get rid of easy cases.

## 3. Why no interactive dashboard?

I spent too long fine tuning the deterministic matcher and XGBoost model so didn't have enough time to implement an interactive dashboard. There is a dashboard but it is only for viewing data, and even that is not very good. Sorry.
