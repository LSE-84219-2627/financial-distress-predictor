# Financial Distress Prediction Pipeline

End-to-end pipeline that predicts whether a US public company will enter financial distress in the following fiscal year — from raw SEC regulatory filings through to an interactive dashboard.

---

## Screenshots

*(Dashboard screenshots to be added)*

---

## Summary

This project builds a complete, auditable pipeline for corporate financial distress prediction. It downloads XBRL accounting data directly from the SEC EDGAR API, parses and stores it in a local SQLite database, engineers six financial-ratio features from the raw figures, trains logistic regression and XGBoost classifiers under a grouped cross-validation scheme that respects firm boundaries, and presents results through a Streamlit dashboard. Every step from raw data to evaluated prediction is reproducible from a single command sequence. The goal is not a production model but a rigorous demonstration of the full pipeline — including honest documentation of where it works and where it does not.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt

# 2. Set your SEC User-Agent (required by SEC fair-use policy)
#    Format: "Your Name your@email.com"
export SEC_USER_AGENT="Jane Smith jane@example.com"   # macOS/Linux
# $env:SEC_USER_AGENT="Jane Smith jane@example.com"   # Windows PowerShell

# 3. Run the pipeline in order
python ingest/edgar_download.py      # download EDGAR data → data/raw/companyfacts/
python db/build_db.py                # parse JSON → SQLite (db/distress.db)
python features/build_annual.py      # collapse to one row per firm-year → annual table
python features/build_labels.py      # compute distress labels → annual_labeled table
python features/build_models.py      # train models, save OOF predictions → oof_predictions table

# 4. Launch the dashboard
streamlit run app.py
# Open http://localhost:8501
```

**What gets downloaded:** ~40 MB of EDGAR XBRL JSON files (one per firm). The SEC rate limit is 10 requests/second; the script uses 0.15 s delays and backs off on 429s automatically.

**What gets created locally (not committed to git):**
- `data/raw/companyfacts/` — cached JSON files
- `db/distress.db` — SQLite database (~5 MB)

Both are regenerable by re-running the pipeline above.

---

## Data

| Property | Detail |
|----------|--------|
| Source | SEC EDGAR XBRL companyfacts API (public, no licence) |
| Firms | 10 US companies across sectors — tech (AAPL, MSFT, GOOGL), industrials (GE, GM, F), retail (BBBY, GME, M), entertainment (AMC) |
| Time span | ~2008–2025, approximately 17 annual observations per firm |
| Core fields | Total assets, total liabilities, stockholders' equity, revenue, net income — all five must be non-null for a row to enter the modelling sample |
| Extended fields | Operating income, interest expense, OCF, current assets/liabilities (used where available) |
| Modelling table | 138 firm-year rows with a valid next-year label; 30 positives (21.7% base rate) |

The tickers to download are listed in `tickers.txt`. Add or remove lines to change the universe.

---

## Method

### Distress label

A firm is labelled distressed in fiscal year *t* using a primary rule with a documented fallback:

- **Primary (`label_source = coverage`):** operating income < interest expense, i.e. interest coverage ratio (ICR) < 1. Both values must be non-null. This is the standard academic definition: the firm cannot cover its debt service from operating earnings.
- **Fallback (`label_source = fallback`):** when either component is missing from EDGAR, the firm is labelled distressed if net income was negative in *both* year *t* and year *t−1*. Two consecutive loss years are required to avoid flagging one-off charges.
- **Neither rule fires:** label is left as NaN and the row is excluded from the modelling sample.

### Prediction framing

The target variable is `distressed_next` — the distress label at *t+1*. Features are values at *t*, available no earlier than the SEC filing date (`available_from`). This enforces point-in-time correctness: no feature from a filing can be used before that filing was publicly accepted by EDGAR.

### Features — six financial ratios

All derived from core accounting fields at time *t*:

| Ratio | Formula | Expected direction |
|-------|---------|-------------------|
| Leverage | liabilities / assets | higher → more distress |
| Equity ratio | equity / assets | lower → more distress |
| Retained earnings to assets | retained earnings / assets | lower → more distress |
| ROA | net income / assets | lower → more distress |
| Margin | net income / revenue | lower → more distress |
| Current ratio | current assets / current liabilities | lower → more distress |

Equity ratio is dropped from logistic regression (it equals approximately 1 − leverage by the balance-sheet identity, producing near-collinear features and an unstable coefficient). XGBoost retains all six features — tree splits handle collinearity without issue.

### Validation: leave-one-firm-out (LOFO) cross-validation

10 folds, one per firm. Train on 9 firms, predict the held-out 10th. Pool predictions across all folds to produce a single out-of-fold (OOF) probability estimate per row.

This is harder than random k-fold. Random k-fold can put 14 of AMC's 15 years in training and predict the 15th — the model learns AMC's firm-level characteristics and is trivially accurate. LOFO asks whether a model trained on 9 firms can predict distress for a 10th it has never seen. That is the operationally relevant question.

---

## Results

**LR is the better operating model.** At a fixed alert budget of the 15 predicted-riskiest firm-years, logistic regression flags 8 true positives out of 15 (precision 0.53) versus 6 for XGBoost (0.40) and 3 for a random ranker (0.22).

| Model | ROC-AUC | PR-AUC | Top-15 TP | Top-15 precision |
|-------|---------|--------|-----------|-----------------|
| Logistic regression | 0.78 | **0.51** | 8 | 0.53 |
| XGBoost | 0.76 | 0.38 | 6 | 0.40 |
| Random baseline | 0.50 | 0.22 | 3 | 0.22 |

PR-AUC (precision-recall area under curve) is the right metric for this imbalanced dataset — a random ranker scores equal to the base rate (0.22). Logistic regression reaches 0.51.

**Margin (net income / revenue) is the dominant signal** in both models: largest logistic regression coefficient (−1.53 standardised) and 45% of XGBoost's total information gain. Firms heading into distress tend to show sustained margin compression years before the ICR formally crosses 1.

**Distress is highly persistent.** Once a firm is distressed this year, there is a 62% chance it is distressed next year. The persistence baseline (predict next year = this year) achieves F1 = 0.63 and is a strong competitor.

**The model does not reliably predict onset.** Of the 10 firm-years where a currently healthy firm enters distress the following year, logistic regression catches 1–2 at an operationally useful alert budget. XGBoost appears to catch all 10, but only by setting its alarm threshold so low it flags ~40% of all firm-years — not useful in practice. The model's real value is **risk ranking**: separating riskier names from safer ones, not pinpointing the exact year distress begins.

---

## Illustrative firms

**Apple — clean negative control.** LR assigns probability below 0.09 in every one of Apple's 16 years. The model never flags it. Strong margins and minimal leverage produce consistently low scores, which is the right answer.

**AMC — entrenched multi-year decline.** AMC has been interest-coverage-distressed since FY2017. The model correctly assigns high probability through the declining years. The FY2016 onset (the year AMC first became distressed) is not caught — at that point AMC's ratios were not yet extreme enough to rank in the top flags. This is an honest limitation.

**Macy's — sharp COVID shock caught a year early.** Macy's had a single distress event in FY2020 (COVID store closures). LR assigned a probability of 0.89 in FY2019, while Macy's was still technically healthy, making it the clearest example of genuine early-warning signal in this dataset.

*(Dashboard screenshots showing these three firms to be added here)*

---

## Rigour: what was built carefully

**Point-in-time correctness.** Every feature row is stamped with the SEC filing date (`available_from`). A backtested model cannot use a number before it was publicly filed. This is the most common data-leakage failure in accounting-based prediction work.

**Grouped cross-validation.** No firm appears in both training and test sets within any fold, by construction. See the validation section above.

**Three real pipeline bugs found and fixed during development:**

1. *Comparative-period overwrite.* SEC 10-K filings contain the current and prior year's balance sheet in the same submission, both tagged with the filing year. The original grouping logic silently picked the prior-year figure for roughly half the rows. Fixed by filtering to the maximum period-end date within each filing before selecting the annual value.

2. *FY = filing year, not period year.* EDGAR tags every row in a 10-K with the filing year, not the period the row describes. A FY2012 10-K therefore contained three consecutive full-year revenue figures all labelled `fy=2012`. Fixed by using the period end date as the grouping key rather than the `fy` field.

3. *Tag-aliasing gaps.* Several firms use non-standard XBRL tag names for common concepts. GME used `SalesRevenueGoodsNet` instead of `Revenues`; Ford used the continuing-operations OCF tag. These showed as unexplained NaNs in the panel. Fixed by building an explicit alias map — a COMBINE-ALL union for revenue (the tag name changed with a 2018 accounting standard) and a PREFER-BASE fallback for equity and OCF.

---

## Limitations

- **Ten firms is a small sample.** Results are suggestive, not generalisable. A real study would require hundreds of firms across sectors and economic cycles.
- **Positive concentration.** Half of all 30 positive firm-years come from two firms (AMC and BBBY). Any train/test split is sensitive to where these firms land, and LOFO performance on those two firms dominates the aggregate metrics.
- **Annual frequency misses intra-year deterioration.** A firm can deteriorate sharply within a fiscal year and not appear distressed until the annual filing. Quarterly data would surface this earlier.
- **Label-measurement weakness when interest tags are missing.** When `InterestExpense` is absent from EDGAR, the fallback rule fires on net income alone. BBBY had positive net income in FY2021 and FY2022 (from one-off asset-sale gains), so the fallback labelled it *not distressed* in both years — even though it filed for bankruptcy 12–18 months later. The label is only as good as EDGAR's tag coverage, and missing interest tags in the final years of a declining firm are not random.

---

## Contributors

- **[Name 1]** — end-to-end pipeline design and implementation: data ingestion, database schema, feature engineering, model training and validation, dashboard
- **[Name 2]** — sector analysis: comparing distress patterns across industrials, retail, and tech; contextualising BBBY and GME trajectories against real-world events
- **[Name 3]** — company case studies: detailed narrative of each firm's financial trajectory mapped to events (COVID, meme-stock period, corporate restructurings)
