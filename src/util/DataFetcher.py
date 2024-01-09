import pandas as pd
import yfinance as yf

def get_all_adjusted_close_data(start_date, end_date, tickers):
    """This function returns a pd dataframe with all of the adjusted closing information"""
    data = pd.DataFrame()
    names = list()
    for i in tickers:
        data = pd.concat([data, pd.DataFrame(yf.download(i, start=start_date, end=end_date).iloc[: ,4])], axis=1)
        names.append(i)
    data.columns = names
    return data

def fetch_prices_df(start_date, end_date, ticker_symbol):
    """This function returns df with yfinance stock data for ticker"""
    # Fetch the data
    df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
    return df