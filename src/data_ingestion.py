import yfinance as yf
import pandas as pd
import argparse
import os

def fetch_stock_data(ticker, start_date, end_date, interval='1h', save_dir='data'):
    """
    Fetches stock data from yfinance and saves it to a CSV file.

    Args:
    - ticker (str): Stock ticker symbol (e.g., 'AAPL').
    - start_date (str): Start date for data retrieval (e.g., '2023-01-01').
    - end_date (str): End date for data retrieval (e.g., '2023-12-31').
    - interval (str): Data resolution interval (e.g., '1h' for hourly data).
    - save_dir (str): Directory where the data will be saved.

    Returns:
    - df (pd.DataFrame): The fetched stock data.
    """
    # Fetch the data
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the data to CSV
    file_path = os.path.join(save_dir, f"{ticker}_{start_date}_to_{end_date}.csv")
    df.to_csv(file_path)
    
    print(f"Data for {ticker} from {start_date} to {end_date} saved at: {file_path}")
    
    return df


def main():
    """
    Main function to run the data ingestion script from the command line.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fetch and save stock data from yfinance.')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., NVDA)')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (e.g., 2023-01-01)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (e.g., 2023-12-31)')
    parser.add_argument('--interval', type=str, default='1h', help='Data interval (default: 1h)')
    parser.add_argument('--save_dir', type=str, default='data', help='Directory to save data (default: data)')

    args = parser.parse_args()

    # Call the data fetching function with the provided arguments
    fetch_stock_data(args.ticker, args.start_date, args.end_date, args.interval, args.save_dir)


if __name__ == "__main__":
    main()
