import yfinance as yf


def main() -> None:
    """Download and display Intel adjusted daily prices."""
    ticker = "INTC"
    start_date = "2023-06-01"
    end_date = "2026-05-31"

    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Keep only adjusted close prices, if available
    adjusted_data = data["Adj Close"].to_frame()

    print(f"Retrieved {len(adjusted_data)} rows for {ticker} from {start_date} to {end_date}.")
    print("\nFirst 5 rows:")
    print(adjusted_data.head())
    print("\nLast 5 rows:")
    print(adjusted_data.tail())


if __name__ == "__main__":
    main()
