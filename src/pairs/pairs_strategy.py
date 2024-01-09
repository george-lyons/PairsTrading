
import math
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from hurst import compute_Hc
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint
from src.cointegration.functions import ad_fuller,linear_regression,pairwise_engle_granger_coint,ad_fuller_to_df
from src.cointegration.linear_regression import regresion_ols

## Initiate a Pair with a date and lookback window
## do stationarity checks on residuals
## Estimates are made from the lookback period for our bounds 
## put all data you want, have a lookback window as training period, and a look forward window as the test period
class TradingPair:
    P_VALUE_THRESHOLD = 0.05
    LOOK_BACK_PERIOD = 365
    TRAIN_TEST_SPLIT_PERC = 50

    def __init__(self,
                 stockX, stockY,
                 end_look_back_date = None,
                 lookback_period=LOOK_BACK_PERIOD,
                 train_test_split=TRAIN_TEST_SPLIT_PERC):
        #either end or we set to date in window
        self.name = f"{stockX.name}, {stockY.name}"
        self.lookback_period = lookback_period

        self.normalized_x = self._normalize_start_1(stockX)
        self.normalized_y = self._normalize_start_1(stockY)

        print('(Pair Created', self.name, 'End Lookback Date', end_look_back_date, 'Lookback Period', lookback_period, ')')
        if end_look_back_date != None:
            try:
                index_positionX = self.normalized_x.index.get_loc(end_look_back_date)
                index_positionY = self.normalized_y.index.get_loc(end_look_back_date)
                print(f"Index position for X {end_look_back_date}: {index_positionX}")
                print(f"Index position for Y {end_look_back_date}: {index_positionY}")
                assert index_positionX == index_positionY
                self.end_look_back_index = index_positionX
            except KeyError:
                print(f"No data found for date: {end_look_back_date}")
        else:
            self.end_look_back_index = len(stockX)

        # Calculate the start index for slicing - and
        self.start_look_back_index = self.end_look_back_index - self.lookback_period
        # Ensure start_index is not negative
        self.start_look_back_index = max(self.start_look_back_index, 0)

        # Retrieve the data from start_index to X (exclusive of X)
        self.look_back_trading_x = self.normalized_x[self.start_look_back_index:self.end_look_back_index]
        self.look_back_trading_y = self.normalized_y[self.start_look_back_index:self.end_look_back_index]
        self.error = False
        self._train_lookback_ols_regression()
        self._do_lookback_stationarity_checks()
        
    def _normalize_start_1(self, prices):
        return prices/prices.iloc[0]
    
    #train on our lookback window
    def _train_lookback_ols_regression(self):
        print('linear regression OLS lookback window')
        print('linear regression OLS lookback window')
        regression = regresion_ols(self.look_back_trading_y, self.look_back_trading_x)
        regression.fit()
        self.look_back_residuals = regression.residuals
        self.beta = regression.beta
        self.c = regression.c


        # regression_model,self.regression_ln_df,residuals = linear_regression(self.look_back_trading_y, self.look_back_trading_x)
        # self.c = regression_model.params.iloc[0]
        # self.beta = regression_model.params.iloc[1]
        # self.lb_residuals = residuals
        

    def _do_lookback_stationarity_checks(self): 
      
        print('beta', self.beta, 'c', self.c)
        adf_result, self.adf_lookback_df = ad_fuller_to_df(self.lb_residuals)
        self.adf_p_value = adf_result[1]
        critical_values = adf_result[4]
        adf_statistic = adf_result[0]
        print('Ad Fuller P ', self.adf_p_value, 'statistic', adf_statistic, 'critical vals', critical_values)

    def is_valid_pair(self):
        return (self.adf_p_value < self.P_VALUE_THRESHOLD)

    ### Just put this into the CONSTRUCTOR - or not, but we may want to have an iterate through, recalculate after each period
    # we use the residual and calculation from our models
    # use the beta calc ahead for that
    # I am still so confused how to implement the long short, but we will figure this out
    def get_test_set_ahead(percentage=50):
        #Percentag of the training set to look ahead and test on unknown data
        test_length = (percentage/100) * self.lookback_period
        # Retrieve the data from start_index to X (exclusive of X)
        # we could re
        test_period_x = self.normalized_x[self.end_look_back_index:test_length]
        test_period_y = self.normalized_y[self.end_look_back_index:test_length]
        print()


    def __repr__(self):
        s = f"Pair [{self.normalized_x}, {self.normalized_y.name}]"
        s += f"\n\tp-value: {self.p_value}"
        # s += f"\n\tMean crosses: {self.mean_crosses}"
        # s += f"\n\tProfitable trades (%): {self.profitable_trades_perc}"
        # s += f"\n\tAverage holding period (days): {self.average_holding_period}"
        s += f"\n\tPair eligible: {self.is_valid_pair()}"
        return s


