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
    adjusted_data = data[["Close"]]

    print(f"Retrieved {len(adjusted_data)} rows for {ticker} from {start_date} to {end_date}.")
    print("\nFirst 5 rows:")
    print(adjusted_data.head())
    print("\nLast 5 rows:")
    print(adjusted_data.tail())


if __name__ == "__main__":
    main()
