# Structural Credit Risk Modeling with Merton's Model

**Application to Intel Corporation (INTC) — 36 months of monthly data (June 2023 – May 2026)**

This project implements Merton's structural model of credit risk on Intel Corporation, recovering the firm's unobservable asset value and volatility from observable market data, then deriving its term structure of default probabilities, survival curve, hazard rates, and risky bond pricing.

The work is part of an Advanced Quantitative Finance course and is built as a portfolio-grade implementation, with reproducible code, validated calculations, and documented methodology.

---

## 1. Project context

In Merton's framework, equity is interpreted as a European call option on the firm's assets, with the strike equal to the firm's debt. This identification, combined with Black–Scholes formulas, allows one to recover the unobservable asset value `V_A` and asset volatility `σ_A` from the observable equity value `E` and equity volatility `σ_E`, by solving a nonlinear 2×2 system:

```
E   = V_A · N(d1) − D · exp(−rτ) · N(d2)
σ_E · E = N(d1) · V_A · σ_A
```

From the calibrated `V_A` and `σ_A`, one derives the **distance-to-default** `DD(t,T)`, the **survival probability** `S(t,T) = N(DD)`, and the corresponding **default probability** `N(−DD)`. The survival curve then feeds into the pricing of risky bonds and the term structure of credit spreads, completing the credit risk picture.

Intel was chosen for its rich storyline over the 2023–2026 period: a deep restructuring, a 60% stock crash in August 2024 (15,000 layoffs and dividend suspension), the appointment of a new CEO, US government equity injection, and a strategic Apple partnership. This volatility makes Intel a particularly informative case for a credit risk model.

---

## 2. Project structure

```
credit-risk-merton-intel/
├── data/
│   ├── raw/                  # Raw downloads (balance sheet snapshots, etc.)
│   └── processed/            # Cleaned, merged datasets
├── src/
│   ├── data_loader.py        # Phase 1: full data pipeline (E, σ_E, r, D)
│   └── explore_debt.py       # Phase 1: balance sheet exploration script
├── outputs/                  # Plots and final figures
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Methodology and phases

The project is structured in six phases, executed sequentially:

| Phase | Step                                          | Main output                          | Status |
|-------|-----------------------------------------------|--------------------------------------|--------|
| 0     | Project setup, environment, version control   | Reproducible Python project          | ✅      |
| 1     | Monthly database construction                 | 662 daily rows: E, σ_E, D, r         | ✅      |
| 2     | Merton calibration on each monthly date       | Time series of V_A, σ_A              | 🔵 In progress |
| 3     | Distance-to-default, survival curve, hazards  | S(t,T), λ(t,T)                       | ⚪      |
| 4     | Risky coupon bond pricing (40% recovery)      | Theoretical price, credit spread     | ⚪      |
| 5     | Term structure of risky zero-coupon prices    | Implied credit spread curve          | ⚪      |
| 6     | Monte-Carlo validation                        | Simulated equity price (10,000 paths)| ⚪      |

---

## 4. Phase 1 — Data pipeline

The daily dataset combines four observable quantities required by the Merton calibration:

| Variable | Symbol  | Source                              | Range observed                  |
|----------|---------|-------------------------------------|---------------------------------|
| Stock price (adjusted) | P_t  | yfinance (`auto_adjust=True`)        | $30 → $124                       |
| Equity volatility      | σ_E  | Rolling 63-day std of log returns, annualized by √252 | 30% → 85%                |
| Shares outstanding     | —    | yfinance `get_shares_full`           | 4.19 → 5.03 billion shares       |
| Market capitalization  | E_t  | Price × shares outstanding           | $79B → $651B                     |
| Risk-free rate         | r_t  | FRED `DGS1` (1-year US Treasury)     | 3.40% → 5.49%                    |
| Default-triggering debt| D_t  | Manual ST + ½ × LT (Macrotrends + yfinance) | $23.5B → $28.9B          |

Two practical points worth noting:
- **Frequency alignment.** All daily variables are merged on a common business-day index and forward-filled where appropriate (shares outstanding and debt are quarterly publications).
- **Debt extraction.** yfinance does not provide enough quarterly balance sheet history for Intel beyond ~5 quarters; the earlier quarters were retrieved manually from Macrotrends and Stockanalysis.com. This hybrid approach reflects the reality of working with public financial data.

The script `data_loader.py` performs the full pipeline end-to-end and saves the result to `data/processed/intel_daily_dataset.csv`.

---

## 5. Reproducing the project

### Prerequisites
- Python 3.10+
- A free [FRED API key](https://fredaccount.stlouisfed.org/apikeys) (used for the 1-year Treasury yield)

### Setup
```bash
git clone https://github.com/Yassos2000/credit-risk-merton-intel.git
cd credit-risk-merton-intel

python -m venv .venv
.venv\Scripts\activate            # on Windows
# source .venv/bin/activate       # on macOS/Linux

pip install -r requirements.txt
```

### Configure the FRED key
```powershell
$env:FRED_API_KEY="your_key_here"   # PowerShell (Windows)
# export FRED_API_KEY=your_key_here # bash (macOS/Linux)
```

### Run the data pipeline
```bash
python src/data_loader.py
```
The script downloads market data, builds the full daily dataset, prints descriptive statistics, and saves the dataset to `data/processed/`.

---

## 6. Preliminary observations

Even before calibration, the dataset already tells a coherent economic story:

- **Equity volatility doubled** over three years (36% in 2023 → 74% in 2026), confirming a structural change in Intel's risk profile.
- **The five largest daily log-returns** correspond to identified events: the August 2024 crash (−30% on layoffs and dividend cut), the September 2025 rally (US government support), the January 2026 drop, and the April 2026 surge on the Apple deal.
- **Market capitalization** ranges from $79B (post-crash trough) to $651B (Apple deal high), an 8× spread.
- **Debt remained relatively stable** ($23.5B – $28.9B), peaking in mid-2024 (operational stress) and decreasing by 2026 (deleveraging).

These observations suggest the calibrated default probability should decrease significantly over the period, despite the higher asset volatility — the asset/debt ratio increasing fast enough to dominate.

---

## 7. Technical stack

- **Language**: Python 3.12
- **Data**: pandas, numpy
- **Market data**: yfinance, fredapi
- **Numerical solvers**: scipy.optimize (for Newton–Raphson calibration in Phase 2)
- **Visualization**: matplotlib

---

## 8. Author

**Yassine** — Graduate of the Institut National de Statistique et d'Économie Appliquée (INSEA), focusing on quantitative finance. Moroccan chess champion.

GitHub: [@Yassos2000](https://github.com/Yassos2000)

---

## 9. References

- Merton, R. C. (1974). *On the Pricing of Corporate Debt: The Risk Structure of Interest Rates*. Journal of Finance, 29(2), 449–470.
- KMV / Moody's Analytics documentation on the structural credit risk approach.
- Course materials, *Advanced Quantitative Finance* (2026).
