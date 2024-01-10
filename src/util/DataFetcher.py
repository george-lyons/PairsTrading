import pandas as pd
import yfinance as yf

def get_all_adjusted_close_data(start_date, end_date, tickers):
    """
    Fetches adjusted close prices for a list of ticker symbols over a specified date range.

    Parameters:
    - start_date (str or datetime): The start date for the data retrieval.
    - end_date (str or datetime): The end date for the data retrieval.
    - tickers (list of str): A list of ticker symbols.

    Returns:
    - pandas.DataFrame: A DataFrame where each column represents the adjusted close prices 
      for one of the ticker symbols over the specified date range.
    """
    data = pd.DataFrame()
    names = list()
    for i in tickers:
        data = pd.concat([data, pd.DataFrame(yf.download(i, start=start_date, end=end_date).iloc[: ,4])], axis=1)
        names.append(i)
    data.columns = names
    return data

# class TestTrainData:
#     def __init__(self, x_train, y_train, x_test, y_test):
#         self.x_train = x_train
#         self.y_train = y_train
#         self.x_test = x_test
#         self.y_test = y_test     

def get_pivoted_test_train_data(stockY,stockX,end_look_back_date = None,lookback_period=365, train_test_split_ratio=1, normalize=False):
    """
    Splits the stock data into training and testing sets based on a pivot date, lookback period, 
    and train-test split ratio.

    Parameters:
    - stockX (pandas.Series): Time series data for stock X.
    - stockY (pandas.Series): Time series data for stock Y.
    - end_look_back_date (str or datetime, optional): The pivot date to split the data. Defaults to None.
    - lookback_period (int): Number of days to look back from the pivot date for the training set. 
    Defaults to 365.
    - train_test_split_ratio (float): The ratio to determine the length of the test set. 
    Defaults to 1 (equal to the lookback period).

    Returns:
    - tuple: Contains four elements (x_train, y_train, x_test, y_test), each a pandas.Series
    representing the training and testing sets for stock X and Y

    - 'normalize_start_1' function is used to normalize the stock data, which should be defined elsewhere.
    """
    indexPivotDate = stockX.index.get_loc(end_look_back_date)
    trainLength = train_test_split_ratio * lookback_period
    #Test and train set
    x_period = stockX[indexPivotDate - lookback_period:indexPivotDate + trainLength]
    y_period = stockY[indexPivotDate - lookback_period:indexPivotDate + trainLength]

    if normalize:
        x_period = normalize_start_1(x_period)
        y_period = normalize_start_1(y_period)

    x_train = x_period[indexPivotDate - lookback_period: indexPivotDate]
    y_train = y_period[indexPivotDate - lookback_period: indexPivotDate]
    x_test = x_period[indexPivotDate: indexPivotDate + trainLength]
    y_test = y_period[indexPivotDate: indexPivotDate + trainLength]

    return x_train, y_train, x_test, y_test

def normalize_start_1(prices):
    return prices/prices.iloc[0]
