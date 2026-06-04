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


if __name__ == "__main__":
    main()
