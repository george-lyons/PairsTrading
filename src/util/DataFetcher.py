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


def get_pivoted_test_train_data(stockX, stockY,
                end_look_back_date = None,
                lookback_period=365,
                train_test_split_ratio=1):

    #assertions
    indexPivotDate = stockX.index.get_loc(end_look_back_date)
    trainLength = train_test_split_ratio * lookback_period

    #Test and train set
    normalized_x = normalize_start_1(stockX[indexPivotDate - lookback_period:indexPivotDate + trainLength])
    normalized_y = normalize_start_1(stockY[indexPivotDate - lookback_period:indexPivotDate + trainLength])

    x_train = normalized_x[indexPivotDate - lookback_period: indexPivotDate]
    y_train = normalized_y[indexPivotDate - lookback_period: indexPivotDate]
    x_test = normalized_x[indexPivotDate: indexPivotDate + trainLength]
    y_test = normalized_y[indexPivotDate: indexPivotDate + trainLength]
    
    return x_train, y_train, x_test, y_test

def normalize_start_1(prices):
    return prices/prices.iloc[0]
