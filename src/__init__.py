# from datetime import datetime

# import pandas as pd

from src.pairs import *
from src.util.DataFetcher import get_all_adjusted_close_data

tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "ETC-USD", "TRX-USD", "BCH-USD", "SOL-USD", "DOT-USD", "LTC-USD", "XRP-USD"]


file_path = 'data.csv'
# # start_date = datetime(2018, 1, 1)
# # end_date = datetime(2024, 1, 27)
# # tickData = get_all_adjusted_close_data(start_date, end_date, tickers)
# # tickData.to_csv('data.csv')
# df = pd.read_csv(file_path)

# pair = MyPair(df['BTC-USD'],df['ETH-USD'])


# df.head()