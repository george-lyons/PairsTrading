# This is a sample Python script.
from datetime import datetime
import time
import pf as pf

from src.pairs import *
import yfinance as yf

from src.pairs.pairs_strategy import *
from src.util.DataFetcher import *
# Suppress all warnings of the category FutureWarning
# warnings.filterwarnings('ignore', category=FutureWarning)

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def print_hi(name):
    # Use a breakpoint in the code line below to
    # debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tickers = ["DPZ", "AAPL", "GOOG", "AMD", "GME", "SPY", "NFLX", "BA", "WMT", "GS", "XOM", "META", "BRK-B", "MSFT", "QQQ"]

    start_date = datetime(2018, 1, 1)
    end_date = datetime(2024, 1, 27)
    tickData = get_all_adjusted_close_data(start_date, end_date, tickers)
    tickData = tickData.dropna()
    print(tickData.shape)
    # Most Recent Data
    tickData.head()

    portfolio = Portfolio(tickData)
    for p in portfolio.selected_pairs:
        p.plot_pair()

    # Sleep for 5 seconds
 





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
