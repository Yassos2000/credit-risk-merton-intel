import os
import yfinance as yf


def main() -> None:
    """Explore Intel's quarterly and annual balance sheets."""
    ticker = yf.Ticker("INTC")

    # Fetch quarterly and annual balance sheet data from yfinance
    quarterly_bs = ticker.quarterly_balance_sheet
    annual_bs = ticker.balance_sheet

    # Print the available row names for the quarterly balance sheet
    print("Quarterly balance sheet row names:")
    print(quarterly_bs.index.tolist())
    print()

    # Print the full quarterly balance sheet DataFrame
    print("Quarterly balance sheet:")
    print(quarterly_bs)
    print()

    # Save the quarterly balance sheet to CSV for inspection
    output_path = os.path.join("..", "data", "raw", "intel_balance_sheet.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    quarterly_bs.to_csv(output_path)
    print(f"Quarterly balance sheet saved to {output_path}")
    print()

    # Extract specific debt rows and compute a combined debt metric per quarter
    try:
        debt_rows = quarterly_bs.loc[["Current Debt", "Long Term Debt"]]
        print("Current Debt and Long Term Debt by quarter:")
        print(debt_rows)
        print()

        d_series = debt_rows.loc["Current Debt"] + 0.5 * debt_rows.loc["Long Term Debt"]
        print("D = Current Debt + 0.5 × Long Term Debt:")
        print(d_series)
        print()
    except KeyError as error:
        print(f"Debt rows not found in quarterly balance sheet: {error}")

    # If annual balance sheet data is available, print its row names too
    if annual_bs is not None and not annual_bs.empty:
        print("Annual balance sheet row names:")
        print(annual_bs.index.tolist())
        print()
        print("Annual balance sheet:")
        print(annual_bs)
    else:
        print("Annual balance sheet data is not available.")


if __name__ == "__main__":
    main()
