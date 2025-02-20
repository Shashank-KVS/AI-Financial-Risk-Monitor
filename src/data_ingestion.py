import asyncio
import yfinance as yf
import pandas as pd
import os
import time
import logging.handlers
import numpy as np

from datetime import datetime
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError
from logging.handlers import RotatingFileHandler
from requests.exceptions import HTTPError 

# Set up our logging - helps us track any issues when pulling stock data
logger = logging.getLogger("DataIngestion")
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler("data_ingestion.log", maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Simple config - change these via env vars if needed
CSV_FILE = os.getenv("CSV_FILE", "multi_stock_data.csv")
DEFAULT_PERIOD = os.getenv("DEFAULT_PERIOD", "max")

# We're using retry logic here because Yahoo's API can be a bit flaky
# Backs off exponentially to avoid hammering their servers
@retry(
    wait=wait_exponential(multiplier=60, min=60, max=300),
    stop=stop_after_attempt(3),
)
def fetch_stock_data(ticker: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Pulls stock data from Yahoo Finance
    
    Added a small delay between requests to be nice to their API.
    Will retry a few times if we hit rate limits.
    """
    logger.info(f"Grabbing data for {ticker}")
    try:
        # Quick pause to avoid hitting rate limits
        time.sleep(2)
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            raise ValueError(f"Huh, no data came back for {ticker}")
    except HTTPError as e:
        if "429" in str(e):  # Rate limit hit
            logger.error(f"Yahoo's telling us to slow down for {ticker}: {e}")
            raise  # Let the retry handle this
        else:
            logger.error(f"Got a HTTP error for {ticker}: {e}")
            raise
    except Exception as e:
        logger.error(f"Something went wrong with {ticker}: {e}")
        raise

    # Clean up the data before sending it back
    data.reset_index(inplace=True)
    data["Ticker"] = ticker
    data["Date"] = pd.to_datetime(data["Date"])
    logger.info(f"Got {len(data)} data points for {ticker}")
    return data

def calculate_returns_and_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Works out all our key financial metrics
    
    Handles daily returns, different volatility windows,
    and some basic price analysis. Takes care of any missing
    data too.
    """
    try:
        # Start fresh with a copy
        df = df.copy()
        df = df.reset_index(drop=True)
        df = df.sort_values(by=["Ticker", "Date"])
        
        # Basic daily returns - pretty standard stuff
        df["Daily Return"] = df.groupby("Ticker")["Close"].transform(
            lambda x: x.pct_change()
        )
        
        # Log returns are handy for some calculations
        df["Log Return"] = df.groupby("Ticker")["Close"].transform(
            lambda x: np.log(x/x.shift(1))
        )
        
        # Look at volatility over different time periods
        windows = [7, 14, 30]  # 1 week, 2 weeks, month
        
        for window in windows:
            df[f"Volatility_{window}d"] = df.groupby("Ticker")["Daily Return"].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # Scale it up to annual
            df[f"Volatility_{window}d_Ann"] = df[f"Volatility_{window}d"] * np.sqrt(252)
        
        # Week-long volatility is our go-to measure
        df["Volatility"] = df["Volatility_7d"]
        
        # Track how prices are moving
        df["Price_Range"] = df["High"] - df["Low"]
        df["Price_Range_Pct"] = df["Price_Range"] / df["Close"]
        
        # Keep an eye on volume too
        df["Volume_Change"] = df.groupby("Ticker")["Volume"].transform(
            lambda x: x.pct_change()
        )
        
        # Fill in any gaps in the data
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ["Date", "Stock Splits", "Dividends"]:
                df[col] = df.groupby("Ticker")[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(0)
                )
        
        logger.info("Got all our calculations done")
        logger.info(f"Data shape now: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Hit a snag in the calculations: {str(e)}")
        raise

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Gets our data ready for analysis
    
    Just makes sure everything's calculated and clean.
    Keeps track of what we're doing in the logs.
    """
    try:
        logger.info("Starting our cleanup")
        logger.info("Missing data check:\n" + df.isnull().sum().to_string())
        
        df = df.reset_index(drop=True)
        df = calculate_returns_and_volatility(df)
        
        logger.info("Quick look at what we calculated:")
        logger.info("\n" + df[["Daily Return", "Volatility"]].describe().to_string())
        
        logger.info("Final missing data check:\n" + df.isnull().sum().to_string())
        
        return df
        
    except Exception as e:
        logger.error(f"Something went wrong cleaning the data: {str(e)}")
        raise

async def update_stock_csv_async(ticker: str, csv_file: str = CSV_FILE, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Updates our stock data file
    
    Grabs fresh data and adds it to what we already have.
    Handles any problems without crashing.
    """
    try:
        data = await asyncio.to_thread(fetch_stock_data, ticker, period)
    except RetryError as e:
        logger.error(f"[{ticker}] Gave up after a few tries: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"[{ticker}] Something went wrong: {e}")
        return pd.DataFrame()

    try:
        # Mix in the new data with what we already have
        if os.path.exists(csv_file):
            existing_data = pd.read_csv(csv_file, parse_dates=["Date"])
            combined_data = pd.concat([existing_data, data], ignore_index=True)
            combined_data.drop_duplicates(subset=["Date", "Ticker"], keep="last", inplace=True)
            combined_data.sort_values(by=["Ticker", "Date"], inplace=True)
        else:
            combined_data = data

        combined_data = clean_dataset(combined_data)
        combined_data.to_csv(csv_file, index=False)
        logger.info(f"[{ticker}] All updated and saved")
    except Exception as e:
        logger.error(f"[{ticker}] Had trouble with the CSV {csv_file}: {e}")
        return pd.DataFrame()
    
    return combined_data

async def update_all_tickers(csv_file: str = CSV_FILE, tickers: list = None, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Updates everything at once
    
    Runs all our updates in parallel to save time.
    Falls back to empty frame if everything fails.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]  # Our default watchlist
    tasks = [update_stock_csv_async(ticker, csv_file, period) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    non_empty_results = [df for df in results if not df.empty]
    if not non_empty_results:
        logger.error("Nothing came back - might want to check what's up")
        return pd.DataFrame()
    combined = pd.concat(non_empty_results, ignore_index=True)
    logger.info("All done!")
    return combined

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]
    combined_data = asyncio.run(update_all_tickers(csv_file=CSV_FILE, tickers=tickers))
    logger.info("Here's what we got:")
    print(combined_data.tail())
