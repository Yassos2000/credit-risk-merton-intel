import os
import numpy as np
import pandas as pd
from fredapi import Fred
import yfinance as yf


def main() -> None:
    """Download and display Intel adjusted daily prices."""
    ticker = "INTC"
    start_date = "2023-06-01"
    end_date = "2026-05-31"

    # Download data from Yahoo Finance with automatic adjustment for dividends and splits
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        group_by="column",
    )

    # Flatten MultiIndex columns if returned by yfinance for a single ticker
    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    # Select the adjusted close prices from the Close column
    adjusted_data = data[["Close"]].copy()

    # Compute daily log returns from adjusted close prices
    adjusted_data["log_return"] = np.log(adjusted_data["Close"] / adjusted_data["Close"].shift(1))

    # Compute rolling annualized equity volatility using a 63-trading-day window
    trading_days = 252
    window_size = 63
    adjusted_data["sigma_E"] = (
        adjusted_data["log_return"].rolling(window=window_size)
        .std(ddof=1)
        * np.sqrt(trading_days)
    )

    # Retrieve historical shares outstanding and align to the daily price index
    shares_outstanding = yf.Ticker(ticker).get_shares_full(
        start=start_date,
        end=end_date,
    )
    shares_outstanding = shares_outstanding.loc[
        ~shares_outstanding.index.duplicated(keep="last")
    ]
    if getattr(shares_outstanding.index, "tz", None) is not None:
        shares_outstanding.index = shares_outstanding.index.tz_convert(None)

    shares_outstanding = shares_outstanding.reindex(
        adjusted_data.index,
        method="ffill",
    )

    adjusted_data["shares_outstanding"] = shares_outstanding
    adjusted_data["market_cap"] = adjusted_data["Close"] * adjusted_data["shares_outstanding"]

    # Retrieve the 1-year US Treasury yield from FRED and convert to decimal
    fred_api_key = os.environ.get("FRED_API_KEY")
    if fred_api_key:
        fred = Fred(api_key=fred_api_key)
        r_series = fred.get_series(
            "DGS1",
            observation_start=start_date,
            observation_end=end_date,
        )
    else:
        # Fallback to the public FRED CSV data URL if no API key is configured
        csv_url = "https://fred.stlouisfed.org/series/DGS1/downloaddata/DGS1.csv"
        r_df = pd.read_csv(
            csv_url,
            parse_dates=["DATE"],
            index_col="DATE",
            na_values=".",
        )
        r_series = r_df.iloc[:, 0]

    r_series = r_series / 100.0
    r_series = r_series.reindex(adjusted_data.index, method="ffill")
    adjusted_data["r"] = r_series

    # Define the default-triggering debt D_t using quarterly debt values in USD
    debt_quarterly = pd.Series({
        "2023-09-30": 25584e6,
        "2023-12-31": 25777e6,
        "2024-03-31": 28516e6,
        "2024-06-30": 28862e6,
        "2024-09-30": 27001e6,
        "2024-12-31": 26870e6,
        "2025-03-31": 27696e6,
        "2025-06-30": 28744e6,
        "2025-09-30": 24525e6,
        "2025-12-31": 24542e6,
        "2026-03-31": 23518e6,
    })
    debt_quarterly.index = pd.to_datetime(debt_quarterly.index)
    adjusted_data["D"] = debt_quarterly.reindex(adjusted_data.index, method="ffill")

    # Verification for D_t before dropping rows with NaN due to rolling volatility window
    print("\n=== Default-triggering debt D_t ===")
    print(adjusted_data[["D"]].head())
    print("\nLast 5 rows of D_t:")
    print(adjusted_data[["D"]].tail())
    print("\nD_t descriptive stats (USD billions):")
    print((adjusted_data["D"] / 1e9).describe())
    missing_D = adjusted_data["D"].isna().sum()
    print("\nMissing values in D after forward-fill:")
    print(missing_D)

    # Remove rows with NaN values produced by the rolling window
    adjusted_data = adjusted_data.dropna()

    print(f"Retrieved {len(adjusted_data)} rows for {ticker} from {start_date} to {end_date} after volatility calculation.")
    print("\nFirst 5 rows:")
    print(adjusted_data.head())
    print("\nLast 5 rows:")
    print(adjusted_data.tail())

    print("\n=== Shares outstanding and market capitalization ===")
    print(adjusted_data[["shares_outstanding", "market_cap"]].head())
    print("\nLast 5 rows of shares outstanding and market cap:")
    print(adjusted_data[["shares_outstanding", "market_cap"]].tail())

    print("\n=== Market cap descriptive stats (USD billions) ===")
    print((adjusted_data["market_cap"] / 1e9).describe())

    print("\n=== Risk-free rate r (1-year US Treasury) ===")
    print(adjusted_data[["r"]].head())
    print("\nLast 5 rows of r:")
    print(adjusted_data[["r"]].tail())

    print("\n=== r descriptive stats (percent) ===")
    print((adjusted_data["r"] * 100).describe())

    missing_r = adjusted_data["r"].isna().sum()
    print("\n=== Missing values in r after forward-fill ===")
    print(missing_r)

    missing_shares = adjusted_data["shares_outstanding"].isna().sum()
    print("\n=== Shares outstanding missing values after forward-fill ===")
    print(missing_shares)

    print("\n=== Statistiques globales de sigma_E ===")
    print(adjusted_data["sigma_E"].describe())

    print("\n=== sigma_E moyenne par annee ===")
    print(adjusted_data["sigma_E"].groupby(adjusted_data.index.year).mean())

    print("\n=== Top 5 plus gros log-rendements ===")
    print(adjusted_data["log_return"].abs().nlargest(5))
    # Get the dates of the top 5 absolute log-returns and show their signed values
    top5_dates = adjusted_data["log_return"].abs().nlargest(5).index
    signed_top5 = adjusted_data["log_return"].loc[top5_dates]
    print("\n=== Top 5 plus gros log-rendements (avec signe) ===")
    print(signed_top5)

    # Save the final daily dataset for later use
    processed_path = os.path.join("..", "data", "processed", "intel_daily_dataset.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    adjusted_data.to_csv(processed_path)
    print(f"Saved processed daily dataset to {processed_path}")

if __name__ == "__main__":
    main()
