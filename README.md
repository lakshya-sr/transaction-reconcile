# 🏦 Multi-Source Reconciliation Agent

> **Razorpay AI Buildathon** | Enterprise Foundational Data & Logic Engine

A deterministic, multi-source financial reconciliation engine built in Python using `uv`, `sqlite3`, `pandas`, and `faker`. Conforms strictly to enterprise accounting standards: **ERPNext/BenchRec** for internal invoicing, **Razorpay JSON Payload** for gateway settlements, and **ISO 20022 CAMT.053** for bank statements.

---

## 🏗 Enterprise System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │           Phase 1: Data Generator            │
                    │             (generate_data.py)               │
                    └──────┬────────────────┬──────────────┬───────┘
                           │                │              │
             ┌─────────────▼──────┐  ┌──────▼──────┐  ┌────▼──────────────┐
             │   erp_ledger.json  │  │ gateway_    │  │ bank_             │
             │   (ERPNext Standard│  │ settlements │  │ statement.csv     │
             │   with TDS & Keys) │  │ (Razorpay)  │  │ (ISO 20022 CAMT)  │
             └─────────────┬──────┘  └──────┬──────┘  └────┬──────────────┘
                           │                │              │
                    ┌──────▼────────────────▼──────────────▼───────┐
                    │           Phase 2: Database Ingestion        │
                    │                (db_setup.py)                 │
                    │                                              │
                    │               reconciliation.db              │
                    │   ┌───────────────┬──────────────────────┐   │
                    │   │ erp_ledger    │ gateway_settlements  │   │
                    │   ├───────────────┼──────────────────────┤   │
                    │   │ bank_statement│ reconciliation_      │   │
                    │   │ (CAMT.053)    │ results              │   │
                    │   └───────────────┴──────────────────────┘   │
                    └───────────────────────┬──────────────────────┘
                                            │
                    ┌───────────────────────▼──────────────────────┐
                    │        Phase 3: Exact Matching Engine        │
                    │              (exact_matcher.py)              │
                    └───────┬──────────────────────────────┬───────┘
                            │                              │
             ┌──────────────▼──────────┐      ┌────────────▼──────────────┐
             │      Exact Matches      │      │     Unmatched Records     │
             │   (Match_Type: Exact 1:1│      │  (Classified Diagnostics  │
             │  Confidence_Score: 1.00)│      │  for LLM AI Matching)     │
             └─────────────────────────┘      └───────────────────────────┘
```

---

## 📊 Database Schemas

### 1. `erp_ledger` (ERPNext / BenchRec Standard)
Tracks gross billing, customer accounts, and expected statutory tax withholding (TDS).
```sql
CREATE TABLE erp_ledger (
    erp_entry_id VARCHAR(50) PRIMARY KEY,      -- Unique internal identifier
    customer_account_id VARCHAR(50),           -- Customer/Debtor account ID
    invoice_number VARCHAR(50),                -- Customer-facing invoice number
    gross_amount DECIMAL(15, 2),               -- Total amount billed to customer
    tds_expected DECIMAL(15, 2) DEFAULT 0,     -- Tax Deducted at Source (Section 194C/194J)
    currency VARCHAR(3) DEFAULT 'INR',         -- Billing currency
    entry_date TIMESTAMP,                      -- Booking timestamp
    status VARCHAR(20),                        -- 'Unpaid', 'Partially Paid', 'Paid'
    allocation_key VARCHAR(100)                -- Clustered matching key
);
```

### 2. `gateway_settlements` (Razorpay JSON Payload Standard)
Maps directly to Razorpay's settlement webhook/API payload with MDR fees and 18% GST calculation.
```sql
CREATE TABLE gateway_settlements (
    payment_id VARCHAR(50) PRIMARY KEY,        -- pay_...
    settlement_id VARCHAR(50),                 -- setl_...
    gateway_status VARCHAR(20),                -- 'captured', 'refunded', etc.
    gross_amount DECIMAL(15, 2),               -- Total paid by customer
    fee_deducted DECIMAL(10, 2),               -- Merchant Discount Rate (MDR)
    tax_on_fee DECIMAL(10, 2),                 -- 18% GST on MDR fee
    net_settled DECIMAL(15, 2),                -- Actual transferred amount (Gross - Fee - GST - Refunds)
    amount_reversed DECIMAL(15, 2) DEFAULT 0,  -- Outgoing refunds
    settled_at TIMESTAMP,                      -- Settlement timestamp
    bank_utr VARCHAR(50)                       -- Unique Transaction Reference linking to Bank
);
```

### 3. `bank_statement` (ISO 20022 CAMT.053 Standard)
Modeled on ISO 20022 XML `<AcctSvcrRef>`, `<ValDt>`, and `<RmtInf>` standard tags.
```sql
CREATE TABLE bank_statement (
    bank_entry_id VARCHAR(50) PRIMARY KEY,     -- <AcctSvcrRef> Account Servicer Reference
    value_date DATE,                           -- <ValDt> Clearing/value date
    transaction_type VARCHAR(10),              -- 'CRDT' (Credit) or 'DBIT' (Debit)
    credit_amount DECIMAL(15, 2),              -- Credit entry amount
    debit_amount DECIMAL(15, 2),               -- Debit entry amount
    running_balance DECIMAL(15, 2),            -- Balance after entry
    remittance_info TEXT,                      -- Messy narrative text string
    reversal_indicator BOOLEAN DEFAULT FALSE   -- <RvslInd> Returned/bounced indicator
);
```

### 4. `reconciliation_results` (Audit Trail Ledger)
Tracks deterministic matches and AI confidence ratings.
```sql
CREATE TABLE reconciliation_results (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    erp_order_id VARCHAR(50),                  -- Links to erp_ledger.erp_entry_id
    gateway_payment_id VARCHAR(50),            -- Links to gateway_settlements.payment_id
    bank_utr VARCHAR(50),                      -- Links to bank_statement UTR
    match_type VARCHAR(20),                    -- 'Exact 1:1', 'Fuzzy Net Match', 'Many:1 Bulk', 'TDS Exception'
    confidence_score DECIMAL(3, 2),            -- 1.00 for deterministic, < 1.00 for AI matching
    notes TEXT                                 -- Explanatory narrative explaining resolution & fees
);
```

---

## 🚀 Execution Guide (Using `uv`)

### 1. Environment Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
```bash
uv run run_pipeline.py
```

### 3. Run Individual Modules
```bash
# Phase 1: Synthetic Data Generation
uv run generate_data.py

# Phase 2: Database Ingestion
uv run db_setup.py

# Phase 3: Exact Matching Engine
uv run exact_matcher.py
```

### 4. Run Test Suite
```bash
uv run pytest -v
```

---

## 🧪 Matching Logic & Diagnostics

1. **Deterministic 3-Way Match (`Exact 1:1`, Confidence: `1.00`)**:
   - Parses `INV-XXXXX` and `UTR` from messy CAMT.053 remittance narrations.
   - Reconciles `Bank.Credit_Amount == Gateway.Net_Settled` (where `Net_Settled = Gross - Fee - GST`).
   - Verifies `Gateway.Gross_Amount == ERP.Gross_Amount` and `TDS_Expected == 0.00`.
   - Persists match into `reconciliation_results`.

2. **Unmatched Anomaly Classification (LLM Input)**:
   - **TDS Withholding Exception**: Flagged when ERP expects TDS (Section 194C/194J).
   - **Obscured CAMT.053 Narrations**: Bank descriptions lacking structured tags, queued for LLM semantic resolution.
   - **Fee Variance / Dispute Discrepancies**: Bank credit differs from gateway net settled.
   - **Orphaned Entries**: Transactions present in Gateway/Bank but missing in ERP.
