"""Prepare monthly dataset for Merton model calibration.

This script loads the daily Intel dataset, resamples it to month-end values,
cleans missing observations, and writes the resulting monthly file.
"""

from pathlib import Path

# PROJECT_ROOT always points to the repository root regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def load_daily_data(csv_path: Path) -> pd.DataFrame:
    """Load the daily dataset from a CSV with a datetime index."""
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)


def prepare_monthly_data(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to month-end values and drop rows with missing data."""
    monthly_df = daily_df.resample("ME").last()
    monthly_df = monthly_df.dropna(how="any")
    return monthly_df


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "processed" / "intel_daily_dataset.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "intel_monthly_dataset.csv"

    daily_df = load_daily_data(input_path)
    monthly_df = prepare_monthly_data(daily_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(output_path)

    print(f"Monthly rows obtained: {len(monthly_df)}")
    print("\nFirst 5 rows of monthly dataset:")
    print(monthly_df.head(5).to_string())
    print("\nLast 5 rows of monthly dataset:")
    print(monthly_df.tail(5).to_string())

    stats = monthly_df[["market_cap", "sigma_E", "r", "D"]].copy()
    stats["market_cap"] = stats["market_cap"] / 1e9
    stats["D"] = stats["D"] / 1e9
    stats["r"] = stats["r"] * 100

    print("\nDescriptive statistics:")
    print(stats.describe().rename(index={
        "count": "Count",
        "mean": "Mean",
        "std": "Std",
        "min": "Min",
        "25%": "25%",
        "50%": "50%",
        "75%": "75%",
        "max": "Max",
    }).to_string())

    # -----------------------------
    # Merton calibration (single-date POC)
    # -----------------------------
    # Define the system of equations for the Merton model to solve for
    # the firm's asset value V_A and asset volatility sigma_A given
    # observable equity value E, equity volatility sigma_E, debt D,
    # risk-free rate r and time-to-maturity tau.
    def merton_system(unknowns, E, sigma_E, D, r, tau):
        V_A, sigma_A = unknowns
        # Avoid invalid values in log or division
        if V_A <= 0 or sigma_A <= 0:
            return [1e6, 1e6]

        d1 = (np.log(V_A / D) + (r + 0.5 * sigma_A ** 2) * tau) / (sigma_A * np.sqrt(tau))
        d2 = d1 - sigma_A * np.sqrt(tau)

        eq1 = V_A * norm.cdf(d1) - D * np.exp(-r * tau) * norm.cdf(d2) - E
        eq2 = norm.cdf(d1) * V_A * sigma_A - sigma_E * E
        return [eq1, eq2]

    def calibrate_single(E, sigma_E, D, r, tau=1.0):
        """Calibrate V_A and sigma_A for a single observation using fsolve.

        Returns dict with keys 'V_A' and 'sigma_A'.
        """
        # Reasonable initial guess: assets equal equity + debt, and
        # asset vol scaled from equity vol by leverage
        V_A_0 = float(E + D)
        sigma_A_0 = float(sigma_E * E / (E + D)) if (E + D) != 0 else max(1e-4, sigma_E)

        solution, info, ier, mesg = fsolve(
            merton_system,
            x0=(V_A_0, sigma_A_0),
            args=(E, sigma_E, D, r, tau),
            full_output=True,
        )

        if ier != 1:
            raise RuntimeError(f"Calibration did not converge: {mesg}")

        V_A_sol, sigma_A_sol = solution
        return {"V_A": float(V_A_sol), "sigma_A": float(sigma_A_sol)}

    # Use the last row of the monthly dataset as a proof-of-concept
    last_row = monthly_df.tail(1).squeeze()
    E = float(last_row["market_cap"])  # equity market cap (USD)
    sigma_E = float(last_row["sigma_E"])  # equity vol (annualized)
    D = float(last_row["D"])  # debt (USD)
    r = float(last_row["r"])  # risk-free rate (decimal)
    tau = 1.0

    print("\n--- Merton calibration (single-date proof of concept) ---")
    print(f"Inputs: E={E}, sigma_E={sigma_E}, D={D}, r={r}, tau={tau}")

    try:
        sol = calibrate_single(E, sigma_E, D, r, tau=tau)
        V_A = sol["V_A"]
        sigma_A = sol["sigma_A"]

        print(f"Calibrated V_A: {V_A / 1e9:.6f} (USD billions)")
        print(f"Calibrated sigma_A: {sigma_A * 100:.4f} %")

        # Verification: compute residuals
        resid1, resid2 = merton_system((V_A, sigma_A), E, sigma_E, D, r, tau)
        print("Verification residuals:")
        print(f"eq1 residual: {resid1:.6e}")
        print(f"eq2 residual: {resid2:.6e}")
    except Exception as e:
        print(f"Calibration failed: {e}")

    # -----------------------------
    # Apply calibration to all monthly dates
    # -----------------------------
    print("\n--- Batch calibration across all monthly dates ---")

    V_results = []
    sigmaA_results = []
    failed = []

    for idx, row in monthly_df.iterrows():
        E_row = float(row["market_cap"]) if not pd.isna(row["market_cap"]) else np.nan
        sigmaE_row = float(row["sigma_E"]) if not pd.isna(row["sigma_E"]) else np.nan
        D_row = float(row["D"]) if not pd.isna(row["D"]) else np.nan
        r_row = float(row["r"]) if not pd.isna(row["r"]) else np.nan
        tau_row = 1.0

        try:
            sol = calibrate_single(E_row, sigmaE_row, D_row, r_row, tau=tau_row)
            V_A_row = sol["V_A"]
            sigma_A_row = sol["sigma_A"]

            # Verify residuals are small
            res1, res2 = merton_system((V_A_row, sigma_A_row), E_row, sigmaE_row, D_row, r_row, tau_row)
            if not (abs(res1) < 1e-3 and abs(res2) < 1e-3):
                failed.append((idx, res1, res2))

        except Exception as e:
            V_A_row = np.nan
            sigma_A_row = np.nan
            failed.append((idx, str(e)))

        V_results.append(V_A_row)
        sigmaA_results.append(sigma_A_row)

    # Attach results to dataframe
    monthly_df["V_A"] = V_results
    monthly_df["sigma_A"] = sigmaA_results

    # Warn if any failures
    if failed:
        print(f"\nWarning: Calibration issues for {len(failed)} dates:")
        for item in failed:
            print(f" - {item}")
    else:
        print("\nAll dates calibrated successfully with residuals below threshold.")

    # Save the enriched monthly dataframe
    calibrated_path = PROJECT_ROOT / "data" / "processed" / "intel_monthly_calibrated.csv"
    calibrated_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(calibrated_path)
    print(f"Saved calibrated monthly dataset to {calibrated_path}")

    # Print summary table (V_A in billions, sigma_A in percent)
    summary = monthly_df[["V_A", "sigma_A"]].copy()
    summary["V_A_bil"] = summary["V_A"] / 1e9
    summary["sigma_A_pct"] = summary["sigma_A"] * 100

    print("\nCalibrated values (V_A in USD billions, sigma_A in %):")
    print(summary[["V_A_bil", "sigma_A_pct"]].to_string())

    print("\nDescriptive statistics for calibrated columns:")
    print(summary[["V_A_bil", "sigma_A_pct"]].describe().to_string())

    # -----------------------------
    # Compute credit risk metrics for the monthly time series (tau = 1 year)
    # -----------------------------
    tau = 1.0
    monthly_df["DD"] = (
        np.log(monthly_df["V_A"] / monthly_df["D"]) +
        (monthly_df["r"] - 0.5 * monthly_df["sigma_A"] ** 2) * tau
    ) / (monthly_df["sigma_A"] * np.sqrt(tau))
    monthly_df["S"] = norm.cdf(monthly_df["DD"])
    monthly_df["PD"] = 1.0 - monthly_df["S"]

    credit_metrics_path = PROJECT_ROOT / "data" / "processed" / "intel_monthly_credit_metrics.csv"
    credit_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(credit_metrics_path)
    print(f"Saved monthly credit metrics to {credit_metrics_path}")

    credit_stats = monthly_df[["DD", "S", "PD"]].copy()
    credit_stats["S_pct"] = credit_stats["S"] * 100
    credit_stats["PD_pct"] = credit_stats["PD"] * 100

    print("\nDescriptive statistics for credit metrics:")
    print(credit_stats[["DD", "S_pct", "PD_pct"]].describe().rename(columns={
        "S_pct": "Survival (%)",
        "PD_pct": "Default Probability (%)",
    }).to_string())

    # -----------------------------
    # Plot the credit risk time series
    # -----------------------------
    date_fmt = mdates.DateFormatter("%Y-%m")
    fig_cr, axs_cr = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), constrained_layout=True)

    axs_cr[0].plot(monthly_df.index, monthly_df["PD"] * 100, marker="o", linestyle="-", color="#d62728")
    axs_cr[0].set_title("Default Probability $PD(t)$ Over Time")
    axs_cr[0].set_xlabel("Date")
    axs_cr[0].set_ylabel("Default Probability (%)")
    axs_cr[0].grid(alpha=0.25)
    max_idx = monthly_df["PD"].idxmax()
    axs_cr[0].annotate(
        "Maximum PD",
        xy=(max_idx, monthly_df.loc[max_idx, "PD"] * 100),
        xytext=(0, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "alpha": 0.6},
        fontsize=9,
    )

    axs_cr[1].plot(monthly_df.index, monthly_df["DD"], marker="o", linestyle="-", color="#2ca02c")
    axs_cr[1].axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axs_cr[1].set_title("Distance to Default $DD(t)$ Over Time")
    axs_cr[1].set_xlabel("Date")
    axs_cr[1].set_ylabel("Distance to Default")
    axs_cr[1].grid(alpha=0.25)

    axs_cr[0].xaxis.set_major_formatter(date_fmt)
    axs_cr[1].xaxis.set_major_formatter(date_fmt)
    for ax in axs_cr:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    credit_plot_path = PROJECT_ROOT / "outputs" / "credit_risk_timeseries.png"
    credit_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig_cr.savefig(credit_plot_path, dpi=120)
    print(f"Saved credit risk time-series plot to {credit_plot_path}")

    # -----------------------------
    # Survival curve at the final date (latest calibration)
    # -----------------------------
    latest_row = monthly_df.iloc[-1]
    V_A_final = latest_row["V_A"]
    sigma_A_final = latest_row["sigma_A"]
    D_final = latest_row["D"]
    r_final = latest_row["r"]
    maturities = np.arange(0.5, 5.1, 0.5)

    survival_dd = (
        np.log(V_A_final / D_final) + (r_final - 0.5 * sigma_A_final ** 2) * maturities
    ) / (sigma_A_final * np.sqrt(maturities))
    survival_S = norm.cdf(survival_dd)
    survival_PD = 1.0 - survival_S

    survival_table = pd.DataFrame({
        "Maturity (years)": maturities,
        "Survival (%)": survival_S * 100,
        "Default Probability (%)": survival_PD * 100,
    })
    print("\nSurvival curve at latest date (2026-05-31):")
    print(survival_table.to_string(index=False, float_format="{:.4f}".format))

    fig_sc, ax_sc = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax_sc.plot(maturities, survival_S * 100, marker="o", linestyle="-", color="#9467bd")
    ax_sc.set_title("Survival Curve at Latest Date (2026-05-31)")
    ax_sc.set_xlabel("Time to Maturity (years)")
    ax_sc.set_ylabel("Survival Probability (%)")
    ax_sc.grid(alpha=0.25)
    ax_sc.set_ylim(0, 100)

    survival_plot_path = PROJECT_ROOT / "outputs" / "survival_curve_latest.png"
    survival_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig_sc.savefig(survival_plot_path, dpi=120)
    print(f"Saved latest survival curve to {survival_plot_path}")

    # -----------------------------
    # Plot the calibrated asset series over time
    # -----------------------------
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), constrained_layout=True)

    # Plot asset value over time in billions of USD
    axs[0].plot(monthly_df.index, monthly_df["V_A"] / 1e9, marker="o", linestyle="-", color="#1f77b4")
    axs[0].set_title("Calibrated Asset Value $V_{A,t}$")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Asset Value (USD billions)")
    axs[0].grid(alpha=0.25)

    # Optionally annotate key events on the V_A plot
    def _nearest_row(timestamp: pd.Timestamp):
        if timestamp < monthly_df.index.min() or timestamp > monthly_df.index.max():
            return None
        position = monthly_df.index.get_indexer([timestamp], method="nearest")[0]
        return monthly_df.iloc[position]

    event_row = _nearest_row(pd.Timestamp("2024-08-31"))
    if event_row is not None:
        axs[0].annotate(
            "Layoffs / dividend cut",
            xy=(pd.Timestamp("2024-08-31"), event_row["V_A"] / 1e9),
            xytext=(-80, 30),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "alpha": 0.6},
        )

    event_row = _nearest_row(pd.Timestamp("2026-04-30"))
    if event_row is not None:
        axs[0].annotate(
            "Apple deal",
            xy=(pd.Timestamp("2026-04-30"), event_row["V_A"] / 1e9),
            xytext=(-80, -35),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "alpha": 0.6},
        )

    # Plot asset volatility over time in percent
    axs[1].plot(monthly_df.index, monthly_df["sigma_A"] * 100, marker="o", linestyle="-", color="#ff7f0e")
    axs[1].set_title("Calibrated Asset Volatility $\sigma_{A,t}$")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Asset Volatility (%)")
    axs[1].grid(alpha=0.25)

    # Format the shared date axis for Year-Month display
    date_fmt = mdates.DateFormatter("%Y-%m")
    axs[0].xaxis.set_major_formatter(date_fmt)
    axs[1].xaxis.set_major_formatter(date_fmt)
    for ax in axs:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plot_path = PROJECT_ROOT / "outputs" / "V_A_sigma_A_timeseries.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    print(f"Saved time-series plot to {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()
