# Multi-Source Reconciliation Agent

Reconciles financial records in a three way situation of merchant ERP, payment gateway, and bank. Uses deterministic engine and AI based clustering to reconcile ERP receipts, payment gateway settlements, and bank statements. Supports one-to-one, one-to-many, many-to-one, and many-to-many relationships between records for both ERP-Gateway and Gateway-Bank reconcilaiation.

## Running

### One click run on Github Codespaces

### Windows

### Linux

#### 1. `pip` based

#### 2. `uv` based

### MacOS (not tested)


## Database Schemas

### 1. `erp_ledger` (ERPNext / BenchRec Standard)
Tracks gross billing, customer accounts, and expected statutory tax withholding (TDS).
```sql
CREATE TABLE IF NOT EXISTS {TABLE_ERP} (
    erp_entry_id TEXT PRIMARY KEY,
    customer_account_id TEXT,
    invoice_number TEXT,
    gross_amount REAL,
    tds_expected REAL,
    currency TEXT,
    entry_date TEXT,
    status TEXT,
    allocation_key TEXT
);
```

### 2. `gateway_records` (Razorpay transaction API)

```sql
CREATE TABLE IF NOT EXISTS {TABLE_GATEWAY} (
    payment_id TEXT PRIMARY KEY,
    settlement_id TEXT,
    gateway_status TEXT,
    gross_amount REAL,
    fee_deducted REAL,
    tax_on_fee REAL,
    net_settled REAL,
    amount_reversed REAL,
    settled_at TEXT,
    bank_utr TEXT,
    invoices TEXT
)
```

### 3. `bank_statement` (ISO 20022 CAMT.053 Standard)
Modeled on ISO 20022 XML `<AcctSvcrRef>`, `<ValDt>`, and `<RmtInf>` standard tags.
```sql
CREATE TABLE IF NOT EXISTS {TABLE_BANK} (
    bank_entry_id TEXT PRIMARY KEY,
    value_date TEXT,
    transaction_type TEXT,
    credit_amount REAL,
    debit_amount REAL,
    running_balance REAL,
    remittance_info TEXT,
    reversal_indicator BOOLEAN
);
```

### 4. `reconciled_edges` (Audit Trail Ledger)
Tracks all records that have been matched so far.
```sql
CREATE TABLE IF NOT EXISTS erp_to_gateway_edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    erp_order_id TEXT NOT NULL,
    gateway_payment_id TEXT NOT NULL,
    allocated_amount REAL NOT NULL,
    match_type TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    notes TEXT
)

CREATE TABLE IF NOT EXISTS gateway_to_bank_edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    gateway_payment_id TEXT NOT NULL,
    bank_entry_id TEXT NOT NULL,
    allocated_amount REAL NOT NULL,
    match_type TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    notes TEXT
)
```


