# This is a sample Python script.
from datetime import datetime
import time
import pf as pf

from src.pairs import *
import yfinance as yf

from src.pairs.Portfolio import Portfolio
from src.util.DataFetcher import *
# Suppress all warnings of the category FutureWarning


tickers = ["DPZ", "AAPL", "GOOG", "AMD", "GME", "SPY", "NFLX", "BA", "WMT", "GS", "XOM", "META", "BRK-B", "MSFT",
            "QQQ"]

# tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "ETC-USD", "TRX-USD", "BCH-USD", "SOL-USD", "DOT-USD", "LTC-USD", "XRP-USD"]
file_path = 'data.csv'
start_date = datetime(2018, 1, 1)
end_date = datetime(2024, 1, 27)
tickData = get_all_adjusted_close_data(start_date, end_date, tickers)
tickData.to_csv(file_path)   
