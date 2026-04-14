# 🔐 Paytm Financial Fraud Detection System

### End-to-end fraud analytics — from raw transactions to ML model deployment

!\[Fraud Detection](https://img.shields.io/badge/Domain-Fintech%20%7C%20Fraud%20Detection-DC2626?style=for-the-badge)
!\[Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
!\[SQL](https://img.shields.io/badge/SQL-PostgreSQL-336791?style=for-the-badge\&logo=postgresql\&logoColor=white)
!\[Excel](https://img.shields.io/badge/Excel-Advanced%20Analysis-217346?style=for-the-badge\&logo=microsoft-excel\&logoColor=white)
!\[PowerBI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge\&logo=powerbi\&logoColor=black)
!\[ML](https://img.shields.io/badge/ML-Random%20Forest-059669?style=for-the-badge)

\---

## 🎯 Project Statement

> \*"How do you detect fraud in 1,00,000 UPI transactions — before the money leaves the account?"\*

This project builds a **complete financial fraud detection pipeline** for Paytm — from raw transaction data analysis to a deployed Random Forest model achieving **99.59% accuracy** and **96.62% recall**, reducing manual review workload by **95.4%**.

\---

## 📊 Project Overview

|Attribute|Detail|
|-|-|
|**Dataset**|1,00,000 UPI transactions|
|**Period**|January 2023 – December 2024|
|**Fraud Rate**|3.7% (3,700 fraud cases)|
|**Fraud Patterns**|6 embedded patterns|
|**Tools**|Python · PostgreSQL · Excel · Power BI|
|**ML Model**|Random Forest (200 trees)|
|**Deliverables**|SQL queries · Excel workbook · Python notebook · Power BI dashboard|

\---

## 🚨 6 Fraud Patterns Analysed

|#|Fraud Type|Cases|% of Fraud|Key Signal|
|-|-|-|-|-|
|1|Velocity Fraud|1,094|29.6%|8+ transactions per hour|
|2|Location Mismatch|788|21.3%|Transaction city ≠ registered city|
|3|New Device Takeover|745|20.1%|Never-seen device + failed logins|
|4|Amount Anomaly|485|13.1%|5x+ the user's 30-day average|
|5|Night Transaction|343|9.3%|12AM–4AM window (4.86% fraud rate)|
|6|New Account Fraud|245|6.6%|Account <30 days + high value|

\---

## 🤖 ML Model — Random Forest Results

### Performance on holdout test set (20,000 transactions)

|Metric|Score|Business Meaning|
|-|-|-|
|**Accuracy**|**99.59%**|19,917 of 20,000 correctly classified|
|**Fraud Precision**|**92.50%**|92 of every 100 alerts are genuine fraud|
|**Fraud Recall**|**96.62%**|715 of 740 actual frauds caught|
|**F1 Score**|**94.51%**|Balanced precision-recall score|
|**AUC-ROC**|**99.96%**|Near-perfect fraud/genuine separation|
|**CV AUC-ROC**|**99.99% ± 0.00%**|Consistent across 5-fold validation|

### Confusion Matrix

||Predicted: Legitimate|Predicted: Fraud|
|-|-|-|
|**Actual: Legitimate**|19,202 ✓|58 ✗|
|**Actual: Fraud**|25 ✗|715 ✓|

### Business Impact

```
Without ML:   Review 100% of 1,00,000 transactions manually
With ML:      Review only 4,627 transactions (4.6%)
Savings:      95,373 transactions auto-approved
Fraud caught: 715 of 740 (96.6%)
False alarms: 58 (0.30% of legitimate transactions)
```

\---

## 🏆 Top Fraud Signals (Feature Importance)

```
1. Location\_Mismatch        16.2% — City hopping detected
2. Amount\_vs\_Avg\_Ratio      15.9% — Sudden high-value spike
3. Txn\_Count\_Last\_1Hr       13.5% — Velocity fraud pattern
4. Amount\_Risk              11.3% — 3x+ average amount flag
5. Velocity\_Risk            10.0% — 5+ transactions per hour
6. Multi\_Flag                9.0% — Multiple risk flags combined
7. Txn\_Count\_Last\_24Hr       7.7% — Daily velocity pattern
8. Is\_New\_Device             3.2% — New device fingerprint
9. Failed\_Login\_Attempts     2.7% — Brute-force signal
10. Hour                     1.8% — Night-time pattern
```

\---

## 📁 Repository Structure

```
paytm-fraud-detection/
│
├── 📁 data/
│   ├── paytm\_fraud\_transactions.csv      # 1,00,000 raw transactions (36 columns)
│   ├── paytm\_fraud\_scored.csv            # Dataset with ML predictions added
│   └── data\_dictionary.md               # Column definitions
│
├── 📁 python/
│   └── Paytm\_Fraud\_RF\_Notebook.py        # Complete ML pipeline (12 cells)
│
├── 📁 sql/
│   └── Paytm\_Fraud\_SQL\_Queries.sql       # 20 queries across 8 business problems
│
├── 📁 excel/
│   └── Paytm\_Fraud\_Detection\_Analysis.xlsx  # 6-sheet analysis workbook
│
├── 📁 powerbi/
│   └── screenshots/                      # Dashboard screenshots (all 6 pages)
│
├── 📁 outputs/
│   ├── fraud\_eda\_charts.png              # 6 EDA visualizations
│   ├── fraud\_model\_results.png           # Confusion matrix + ROC + Feature importance
│   └── fraud\_precision\_recall.png        # Precision-recall curve
│
└── README.md
```

\---

## 🛠️ How to Run

### Python ML Pipeline

```bash
# Step 1: Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

# Step 2: Place CSV in same folder as script
# paytm\_fraud\_transactions.csv must be in same directory

# Step 3: Run the notebook
python Paytm\_Fraud\_RF\_Notebook.py
```

**Generated outputs:**

* `fraud\_eda\_charts.png` — 6 EDA visualizations
* `fraud\_model\_results.png` — Confusion matrix + ROC curve + Feature importance
* `fraud\_precision\_recall.png` — Precision-recall tradeoff
* `paytm\_fraud\_scored.csv` — Full dataset with fraud probability scores

### SQL Queries (PostgreSQL)

```sql
-- Step 1: Create table
CREATE TABLE paytm\_fraud (
    Transaction\_ID TEXT, Date DATE, Time TIME, Hour INT,
    Day\_of\_Week TEXT, Month TEXT, Month\_Num INT, Quarter TEXT,
    User\_ID TEXT, Account\_Age\_Days INT, Device\_ID TEXT,
    Device\_Type TEXT, OS TEXT, App\_Version TEXT, IP\_Address TEXT,
    Is\_New\_Device INT, Transaction\_Type TEXT, Payment\_Category TEXT,
    Amount BIGINT, Txn\_Count\_Last\_1Hr INT, Txn\_Count\_Last\_24Hr INT,
    Avg\_Txn\_Amount\_30D BIGINT, Amount\_vs\_Avg\_Ratio NUMERIC(8,2),
    Failed\_Login\_Attempts INT, Txn\_City TEXT, Txn\_State TEXT,
    Registered\_City TEXT, Registered\_State TEXT, Login\_City TEXT,
    Location\_Mismatch INT, Is\_Night\_Transaction INT,
    Is\_New\_Account INT, Is\_High\_Value\_Flag INT,
    Fraud\_Score INT, Is\_Fraud INT, Fraud\_Type TEXT
);

-- Step 2: Load data
COPY paytm\_fraud FROM '/path/to/paytm\_fraud\_transactions.csv'
DELIMITER ',' CSV HEADER;

-- Step 3: Run any query from Paytm\_Fraud\_SQL\_Queries.sql
```

### Power BI Dashboard

```
1. Open Power BI Desktop
2. Get Data → Text/CSV → load paytm\_fraud\_scored.csv
3. Transform Data → fix column types → Close \& Apply
4. Rename table to fraud\_data
5. Create DAX measures (listed in README below)
6. Build 6-page dashboard
```

\---

## 📐 Key DAX Measures

```dax
-- Total fraud cases
Total Fraud Cases =
CALCULATE(COUNTROWS(fraud\_data), fraud\_data\[Is\_Fraud] = 1)

-- Fraud rate percentage
Fraud Rate % =
DIVIDE(\[Total Fraud Cases], COUNTROWS(fraud\_data), 0) \* 100

-- Model precision (live from data)
Model Precision =
VAR TP = CALCULATE(COUNTROWS(fraud\_data),
    fraud\_data\[Is\_Fraud]=1, fraud\_data\[RF\_Fraud\_Prediction]=1)
VAR FP = CALCULATE(COUNTROWS(fraud\_data),
    fraud\_data\[Is\_Fraud]=0, fraud\_data\[RF\_Fraud\_Prediction]=1)
RETURN DIVIDE(TP, TP+FP, 0) \* 100

-- Manual review savings
Manual Review Savings % =
DIVIDE(
    CALCULATE(COUNTROWS(fraud\_data), fraud\_data\[RF\_Risk\_Tier]="Low"),
    COUNTROWS(fraud\_data), 0
) \* 100
```

\---

## 📈 Power BI Dashboard — 6 Pages

|Page|Content|Key Visual|
|-|-|-|
|1 · Executive Summary|5 KPI cards + fraud type donut + risk tier bar|Overview of entire project|
|2 · Fraud Patterns|Fraud rate by transaction type + scatter chart|Pattern deep-dive|
|3 · Time Analysis|24-hour heatmap + monthly trend combo chart|Night fraud concentration|
|4 · Geography|India map + top 10 cities bar chart|City-wise fraud hotspots|
|5 · ML Model Results|Precision/Recall/AUC-ROC cards + confusion matrix|Model performance|
|6 · Risk Tier Drill|Risk tier distribution + fraud rate by tier|99.33% Critical tier rate|

&#x20;

\---

## 📋 SQL Queries — 8 Business Problems

|Problem|Query|Key Finding|
|-|-|-|
|P1|Velocity Fraud Detection|8+ txn/hr = 4x fraud rate|
|P2|Location Mismatch Analysis|City hopping = 100% fraud signal|
|P3|New Device Fingerprint|New device + failed login = highest risk|
|P4|Amount Anomaly Detection|5x avg amount = 78% fraud rate|
|P5|Night Transaction Heatmap|12AM-4AM = 4.86% vs 3.38% day rate|
|P6|New Account Rapid Spending|<30 day account + high value = 67% fraud|
|P7|Failed Login Pattern|3+ failed attempts before transaction|
|P8|Payment Category Deviation|International transfers highest risk category|

\---

## 🔑 Key EDA Findings

```
Night vs Day fraud rate:     4.86% vs 3.38%  (+44% higher at night)
New Device fraud rate:       100% (synthetic dataset signal)
Location Mismatch rate:      100% (synthetic dataset signal)
Fraud avg transaction:       ₹41,768 vs ₹11,902 legitimate (3.5x higher)
Riskiest hour:               3AM (5.09% fraud rate)
Riskiest transaction type:   Investment (3.92% fraud rate)
```

\---

## 📦 Tech Stack

```
Data Generation    →  Python 3.13 (pandas, numpy)
Exploratory Analysis → Python (matplotlib, seaborn)
Database           →  PostgreSQL 15
Spreadsheet        →  Microsoft Excel (openpyxl, pivot tables)
ML Model           →  scikit-learn (RandomForestClassifier)
Visualisation      →  Power BI Desktop (DAX measures, 6-page dashboard)
Version Control    →  Git / GitHub
```

\---

## 👤 About

Built by **DEVESH KHURANA** — Data Analytics student passionate about fintech and fraud detection.

\---

## 📄 License

MIT License — dataset is synthetically generated for educational purposes.

\---

*If this project helped you, please ⭐ the repository.*

