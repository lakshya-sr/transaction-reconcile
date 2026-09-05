# Multi-Source Reconciliation Agent

Reconciles financial records in a three way situation of merchant ERP, payment gateway, and bank. Uses deterministic engine and AI based clustering to reconcile ERP receipts, payment gateway settlements, and bank statements. Supports one-to-one, one-to-many, many-to-one, and many-to-many relationships between records for both ERP-Gateway and Gateway-Bank reconcilaiation.


# Running

## Github Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/lakshya-sr/transaction-reconcile)

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
| `python main.py --generate` | Generate synthetic data |
| `python main.py --setup-db` | Initialize database |
| `python main.py --match` | Run deterministic matching |
| `python main.py --infer` | Run AI inference |
| `python main.py --evaluate` | Evaluate accuracy |
| `python main.py --benchmark` | Performance profiling |
| `python main.py --dashboard` | Launch Streamlit UI |
| `python main.py --all` | Run entire pipeline |
| `python demo.py` | Step-by-step demo |


# Gallery

![Reconciled records graph](img/rec-graph.png)

![Partially matched gateway transactions](img/matched.png)

