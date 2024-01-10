
import math
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from hurst import compute_Hc
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint
from src.pairs.coint_functions import *
from src.cointegration.linear_regression import regresion_ols

## Initiate a Pair with a date and lookback window
## do stationarity checks on residuals
## Estimates are made from the lookback period for our bounds 
## put all data you want, have a lookback window as training period, and a look forward window as the test period
class TradingPair:
    P_VALUE_THRESHOLD = 0.05
    LOOK_BACK_PERIOD = 365
    TEST_TRAIN_RATIO = 1

    def __init__(self,
                 stockX, stockY,
                 end_look_back_date = None,
                 lookback_period=LOOK_BACK_PERIOD,
                 train_test_split=TEST_TRAIN_RATIO):
        self.lookback_period = lookback_period
        self.normalized_x = self._normalize_start_1(stockX)
        self.normalized_y = self._normalize_start_1(stockY)

        #We could just doa train test split
        self._generate_training_lookback_set(end_look_back_date)
        self._generate_test_set(train_test_split)

        self.name = f"{stockX.name}, {stockY.name}"

        #add t logs the train and test set details
        print(f"(Pair Created {self.name} End Lookback Date {end_look_back_date} Lookback Period {lookback_period})")

        self._train_lookback_ols_regression()
        self._do_lookback_stationarity_checks()
        

    #just for test train window
    def _normalize_start_1(self, prices):
        return prices/prices.iloc[0]

    def _generate_test_set(self, test_train_ratio):
        # Calculate start index for the test set
        start_test_index = self.end_look_back_index
        test_length = test_train_ratio * self.lookback_period
        # Calculate end index for the test set, ensuring it doesn't go beyond the length of the data
        end_test_index = min(start_test_index + test_length, len(self.normalized_x))
        # Slice the normalized data to get the test sets
        self.test_trading_x = self.normalized_x[start_test_index:end_test_index]
        self.test_trading_y = self.normalized_y[start_test_index:end_test_index]
        # Optionally, print information about the test set
        print(f"Test set range: Start={start_test_index}, End={end_test_index}")

    def _set_end_look_back_index(self, end_look_back_date):
        try:
            index_positionX = self.normalized_x.index.get_loc(end_look_back_date)
            index_positionY = self.normalized_y.index.get_loc(end_look_back_date)
            print(f"Index positions for {end_look_back_date}: X={index_positionX}, Y={index_positionY}")
            assert index_positionX == index_positionY, "Index positions for X and Y do not match."
            self.end_look_back_index = index_positionX
            print('Set end lookback index ', self.end_look_back_index)
        except KeyError:
            self.error = True
            print(f"No data found for date: {end_look_back_date}")

    def _generate_training_lookback_set(self, end_look_back_date):
        if end_look_back_date:
            self._set_end_look_back_index(end_look_back_date)
        else:
            self.end_look_back_index = len(self.normalized_x)
            
        self.start_look_back_index = max(self.end_look_back_index - self.lookback_period, 0)
        print('Set start look back', self.start_look_back_index)
        self.look_back_trading_x = self.normalized_x[self.start_look_back_index:self.end_look_back_index]
        self.look_back_trading_y = self.normalized_y[self.start_look_back_index:self.end_look_back_index]
    
    def _train_lookback_ols_regression(self):
        print('linear regression OLS lookback window')
        regression = regresion_ols(self.look_back_trading_y, self.look_back_trading_x)
        regression.fit()
        self.look_back_residuals = regression.residuals
        self.beta = regression.beta
        self.c = regression.c
        self._test_residuals_predict(regression)

    def _test_residuals_predict(self, regression):
        #Predicting the residuals for test set from training regression
        self.test_residual_predict = regression.predict_residuals(self.test_trading_y, self.test_trading_x)

    def _do_lookback_stationarity_checks(self):  
        print('beta', self.beta, 'c', self.c)
        adf_result, self.adf_lookback_df = ad_fuller_to_df(self.look_back_residuals)
        self.adf_p_value = adf_result[1]
        critical_values = adf_result[4]
        adf_statistic = adf_result[0]
        print('Ad Fuller P ', self.adf_p_value, 'statistic', adf_statistic, 'critical vals', critical_values)

    def is_valid_pair(self):
        return (self.adf_p_value < self.P_VALUE_THRESHOLD)

    # ### Just put this into the CONSTRUCTOR - or not, but we may want to have an iterate through, recalculate after each period
    # # we use the residual and calculation from our models
    # # use the beta calc ahead for that
    # # I am still so confused how to implement the long short, but we will figure this out
    # def get_test_set_ahead(self, percentage=50):
    #     #Percentag of the training set to look ahead and test on unknown data
    #     test_length = (percentage/100) * self.lookback_period
    #     # Retrieve the data from start_index to X (exclusive of X)
    #     # we could re
    #     test_period_x = self.normalized_x[self.end_look_back_index:test_length]
    #     test_period_y = self.normalized_y[self.end_look_back_index:test_length]
    #     print()


    def __repr__(self):
        s = f"Pair [{self.normalized_x.name}, {self.normalized_y.name}]"
        s += f"\n\tp-value: {self.adf_p_value}"
        # s += f"\n\tMean crosses: {self.mean_crosses}"
        # s += f"\n\tProfitable trades (%): {self.profitable_trades_perc}"
        # s += f"\n\tAverage holding period (days): {self.average_holding_period}"
        s += f"\n\tPair eligible: {self.is_valid_pair()}"
        return s


