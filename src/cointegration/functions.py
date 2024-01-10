import pandas as pd
import statsmodels.tsa.stattools as ts 
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

from statsmodels.tsa.api import VAR
# Set option to display more rows, for example, 100 rows
pd.set_option('display.max_rows', 100)

def find_high_correlation_pairs(self, corr_matrix, threshold=0.9):
    """Returns pairs with correlation > than threshold"""
    # Extract the column names (assets)
    assets = corr_matrix.columns
    # List to store the pairs with high correlation
    high_corr_pairs = []
    
    # Iterate through the columns
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):  # Only check above the diagonal
            if corr_matrix.iloc[i, j] > threshold:
                # If correlation is above the threshold, add the
                #  pair to the list
                high_corr_pairs.append((assets[i], assets[j], corr_matrix.iloc[i, j]))       
    return high_corr_pairs

# def linear_regression(Y,X, constant=True):
#     """
#     Perform linear OLS regression on prices A and B
#     Parameters:
#     A (array-like): Prices of Asset A (dependent variable)
#     B (array-like): Prices of Asset B (independent variable)    """
#     if constant :
#         X = sm.add_constant(X)  # Adding a constant for the intercept
#     model = OLS(Y, X).fit()
#     df_results = pd.DataFrame({
#         'Estimate': model.params,
#         'SD of Estimate': model.bse,
#         't-Statistic': model.tvalues,
#         'p-value': model.pvalues
#     })
#     res =  pd.DataFrame(model.resid)
#     res.columns = ['Residuals'] 
#     return model,df_results,res

# add a VAR function with info needed
def var():
    print()

def pairwise_engle_granger_coint(series1, series2):
    """Engle granger test, using """
    coin = ts.coint(series1, series2)
    t_statistic = coin[0]
    p_val = coin[1]
    critical_values_test_statistic_at_1_5_10 = coin[2]
    return t_statistic, p_val, critical_values_test_statistic_at_1_5_10

def rolling_multiple_coint_test(asset1, asset2, window_size):
    """
    Perform rolling cointegration estimation on residuals in sliding window to find cointegrated periods
    Return those where ADF statistic lesss than critical 5 % value
    Also perform on each of the assets
    """
    if len(asset1) != len(asset2):
        raise ValueError("The lengths of both asset time series must be the same.")
        
    dfStationary = pd.DataFrame(columns=['start', 'end', '1% DF', '5% DF', '10% DF', 'adf_statistic', 'adf_p_value', 'adf_is_stationary', 'coint_p_value', 'coint_is_stationary'])
        
    for start in range(len(asset1) - window_size + 1):
        end = start + window_size
        window_data_asset1 = asset1[start:end]
        window_data_asset2 = asset2[start:end]
        
        t_statistic, coint_p_val, critical_values_test_statistic_at_1_5_10 = pairwise_engle_granger_coint(window_data_asset1,window_data_asset2)
        model,df,residuals = linear_regression(window_data_asset1,window_data_asset2)

        # Perform the ad fuller test on residuals
        adf_result = ad_fuller(residuals)
        adf_p_value = adf_result[1]
        critical_values = adf_result[4]
        adf_statistic = adf_result[0]
        adf_is_stationary = adf_p_value < 0.05

        start_date = asset1.index[start]
        end_date = asset1.index[end - 1]  # Subtract 1 because end is exclusive

        row_to_add = pd.DataFrame([[start_date, end_date, critical_values["1%"], critical_values["5%"], critical_values["10%"], adf_statistic, adf_p_value, adf_is_stationary, coint_p_val, coint_p_val < 0.05]], columns=dfStationary.columns)
        dfStationary = dfStationary._append(row_to_add, ignore_index=True)

    return dfStationary

def rolling_ad_fuller_test(asset1, asset2, window_size):
    """
    Perform rolling cointegration estimation on residuals in sliding window to find cointegrated periods
    Return those where ADF statistic lesss than critical 5 % value
    Also perform on each of the assets
    """
    if len(asset1) != len(asset2):
        raise ValueError("The lengths of both asset time series must be the same.")
        
    dfStationary = pd.DataFrame(columns=['start', 'end', '1% DF', '5% DF', '10% DF', 'adf_statistic', 'p_value'])
        
    for start in range(len(asset1) - window_size + 1):
        end = start + window_size
        window_data_asset1 = asset1[start:end]
        window_data_asset2 = asset2[start:end]

        model,df,residuals = linear_regression(window_data_asset1,window_data_asset2)

        # Perform the ad fuller test on residuals
        adf_result = ad_fuller(residuals)
        adf_p_value = adf_result[1]
        critical_values = adf_result[4]
        adf_statistic = adf_result[0]
        is_stationary = adf_p_value < 0.05

        if is_stationary:
            start_date = asset1.index[start]
            end_date = asset1.index[end - 1]  # Subtract 1 because end is exclusive

            row_to_add = pd.DataFrame([[start_date, end_date, critical_values["1%"], critical_values["5%"], critical_values["10%"], adf_statistic, adf_p_value]], columns=dfStationary.columns)
            dfStationary = dfStationary._append(row_to_add, ignore_index=True)

    return dfStationary

def ad_fuller_to_df(e):
    """
    Perform rad fuller estimation on residuals in sliding window to find cointegrated periods
    Return those where ADF statistic lesss than critical 5 % value
    """
    dfStationary = pd.DataFrame(columns=['1% DF', '5% DF', '10% DF', 'adf_statistic', 'p_value', 'is_stationary'])
    adf_result = ad_fuller(e)
    adf_p_value = adf_result[1]
    critical_values = adf_result[4]
    adf_statistic = adf_result[0]
    is_stationary = adf_p_value < 0.05

    row_to_add = pd.DataFrame([[critical_values["1%"], critical_values["5%"], critical_values["10%"], adf_statistic, adf_p_value, is_stationary]], columns=dfStationary.columns)
    dfStationary = dfStationary._append(row_to_add, ignore_index=True)
    return adf_result, dfStationary

def ad_fuller(e):
    """
    Perform ad fuller test on residuals
    return the ad fuller result
    """
    res = e.copy()
    # do not include trend in ADF
    adf_result = adfuller(res, maxlag=1, regression='c')
    return adf_result

