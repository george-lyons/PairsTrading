
import math
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from hurst import compute_Hc
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint
from src.pairs.coint_functions import *
from src.util.DataFetcher import *
from src.cointegration.linear_regression import regresion_ols
from enum import Enum


## Initiate a Pair with a date and lookback window
## do stationarity checks on residuals
## Estimates are made from the lookback period for our bounds 
## put all data you want, have a lookback window as training period, and a look forward window as the test period
class TradingPair:

    P_VALUE_THRESHOLD = 0.05
    LOOK_BACK_PERIOD = 365
    TEST_TRAIN_RATIO = 1

    class Status(str, Enum):
        TRAIN = 'train'
        TEST = 'test'

    def __init__(self,
                    x_train, y_train,
                    x_test, y_test):
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
    
            self.name = f"{self.x_train.name}, {self.y_train.name}"
            #add t logs the train and test set details
            print(f"Pair Created ({self.name})")
            print(f"Start train {self.x_train.index.min()}) End train {self.x_train.index.max()})")
            print(f"Start test {self.x_test.index.min()}) End test {self.x_test.index.max()})")

            self._train_lookback_ols_regression()
            #we could stationary check the TEST set also (for info)
            self._do_lookback_stationarity_checks()
            self._ou_fit()
            self._generate_trading_strategy()

    # def __init__(self,
    #              stockX, stockY,
    #              end_look_back_date = None,
    #              lookback_period=LOOK_BACK_PERIOD,
    #              train_test_split=TEST_TRAIN_RATIO):
    #     self.lookback_period = lookback_period

    #     #assertions
    #     indexPivotDate = stockX.index.get_loc(end_look_back_date)
    #     trainLength = train_test_split * lookback_period

    #     #All data so we can move window
    #     self.X = stockX
    #     self.Y = stockY

    #     #Test and train set
    #     self.normalized_x = self._normalize_start_1(stockX[indexPivotDate - lookback_period:indexPivotDate + trainLength])
    #     self.normalized_y = self._normalize_start_1(stockY[indexPivotDate - lookback_period:indexPivotDate + trainLength])

    #     self.x_train = self.normalized_x[indexPivotDate - lookback_period: indexPivotDate]
    #     self.y_train = self.normalized_y[indexPivotDate - lookback_period: indexPivotDate]
    #     self.x_test = self.normalized_x[indexPivotDate: indexPivotDate + trainLength]
    #     self.y_test = self.normalized_y[indexPivotDate: indexPivotDate + trainLength]
     
    #     self.name = f"{stockX.name}, {stockY.name}"
    #     #add t logs the train and test set details
    #     print(f"(Pair Created {self.name} End Lookback Date {end_look_back_date} Lookback Period {lookback_period})")
    #     self._train_lookback_ols_regression()
    #     #we could stationary check the TEST set also (for info)
    #     self._do_lookback_stationarity_checks()
    #     self._ou_fit()
    #     self._generate_trading_strategy()
 
    def _generate_trading_strategy(self):
        print('TODO TRADE STRAT')

    def _ou_fit(self):
        print('TODO OU FIT')

    #just for test train window
    def _normalize_start_1(self, prices):
        return prices/prices.iloc[0]

    def _train_lookback_ols_regression(self):
        print('linear regression OLS lookback window')
        regression = regresion_ols(self.y_train, self.x_train)
        regression.fit()
        self.look_back_residuals = regression.residuals
        self.beta = regression.beta
        self.c = regression.c
        self._test_residuals_predict(regression)

    def _test_residuals_predict(self, regression):
        #Predicting the residuals for test set from training regression
        self.test_residual_predict = regression.predict_residuals(self.y_test, self.x_test)

    def _do_lookback_stationarity_checks(self):  
        print('beta', self.beta, 'c', self.c)
        adf_result, self.adf_lookback_df = ad_fuller_to_df(self.look_back_residuals)
        self.adf_p_value = adf_result[1]
        critical_values = adf_result[4]
        adf_statistic = adf_result[0]
        print('Ad Fuller P ', self.adf_p_value, 'statistic', adf_statistic, 'critical vals', critical_values)

    def is_valid_pair(self):
        return (self.adf_p_value < self.P_VALUE_THRESHOLD)

    def __repr__(self):
        # s = f"Pair [{self.normalized_x.name}, {self.normalized_y.name}]"
        s = f"\n\tp-value: {self.adf_p_value}"
        # s += f"\n\tMean crosses: {self.mean_crosses}"
        # s += f"\n\tProfitable trades (%): {self.profitable_trades_perc}"
        # s += f"\n\tAverage holding period (days): {self.average_holding_period}"
        s += f"\n\tPair eligible: {self.is_valid_pair()}"
        return s


