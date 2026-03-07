# OBV + Moving Average Integration Design

## 1) Architecture understanding

This repository currently performs:

1. **Universe selection** (`data/fetcher.py` / Streamlit sidebar)
2. **Historical data fetch** from Upstox
3. **OBV computation + divergence detection**
4. **Filter to OBV-qualified stocks** (`results_container = [r for r in results_container if r[2]]`)
5. **Render scan outputs in Streamlit** (`appV2.py`)

The Moving Average project (provided map) performs:

1. Data ingestion (Upstox / Yahoo)
2. Feature generation (`compute_sma`, `compute_ema`, signal generation)
3. Optimization / adaptive MA type selection (`select_ma_type`)
4. Report generation and UI display

## 2) Key MA functions to integrate

From the MA project design, the integration-critical functions are:

- **MA calculation**: `compute_sma`, `compute_ema`
- **Signal generation**: `generate_signals`
- **Adaptive selection**: `compute_volatility`, `compute_trend_strength`, `compute_noise_ratio`, `select_ma_type`

These were added as reusable local module functions under `moving_average/` so OBV logic remains untouched.

## 3) Integration approach

- Keep OBV screening unchanged.
- Hook MA analysis **after** OBV filtering.
- Reuse already-fetched OHLCV dataframe from OBV stage (no re-fetch).
- Produce MA metadata columns for each OBV-qualified stock:
  - MA Type (SMA/EMA)
  - MA Pair (fast/slow)
  - Latest MA Signal
  - Latest MA Crossover Date

This satisfies the required pipeline:

Stock Universe → OBV Screener → Filtered Stocks → Moving Average Analysis → Final Signals

## 4) Suggested folder structure

```text
MERGER-MA-OBV/
├── moving_average/
│   ├── __init__.py
│   ├── features.py          # compute_sma, compute_ema, generate_signals
│   ├── analyzer.py          # adaptive selection + OBV-filtered analysis runner
│   └── INTEGRATION.md       # design and data flow notes
├── appV2.py                 # OBV Streamlit app + integration hook
├── data/
├── charts/
└── ...
```

## 5) Data flow between systems

1. Streamlit app scans universe and computes OBV divergences.
2. Stocks without OBV signals are discarded.
3. Remaining `results_container` rows are passed to `analyze_obv_filtered_stocks(...)`.
4. MA module computes adaptive MA type and crossover signal using the same dataframe.
5. MA outputs are merged into `summary_rows` for final presentation/export.

The integration is one-way and non-invasive: OBV remains the gate, MA is downstream analysis only.
