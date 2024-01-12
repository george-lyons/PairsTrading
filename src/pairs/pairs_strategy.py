
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
from src.cointegration.ou_fit import OUFit
from enum import Enum
import numpy as np

## Initiate a Pair with a date and lookback window
## do stationarity checks on residuals
## Estimates are made from the lookback period for our bounds 
## put all data you want, have a lookback window as training period, and a look forward window as the test period
class TradingPair:

    P_VALUE_THRESHOLD = 0.05
    LOOK_BACK_PERIOD = 365
    TEST_TRAIN_RATIO = 1

    class Data(str, Enum):
        TRAIN = 'train'
        TEST = 'test'

    def __init__(self,
                    x_train, y_train,
                    x_test, y_test):
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
            self.tau = 365
    
            self.name = f"{self.y_train.name}, {self.x_train.name}"
            #add t logs the train and test set details
            print(f"Pair Created ({self.name})")
            print(f"Start train {self.x_train.index.min()}) End train {self.x_train.index.max()})")
            print(f"Start test {self.x_test.index.min()}) End test {self.x_test.index.max()})")

            self._simple_returns()
            self._train_lookback_ols_regression()
            #we could stationary check the TEST set also (for info)
            self._do_lookback_stationarity_checks()
            self._ou_fit()
            self._generate_trading_strategy()

    def _generate_trading_strategy(self):
        print('TODO TRADE STRAT')

    def _ou_fit(self, z=1):
        """
        Fits the residual (training) to OU process
        Calculates the bounds (upper and lower and mean) for our process
        """
        self.ou = OUFit(self.train_residuals, 1/self.tau)
        self.ou.fit()
        #get range bound frame - for training set 
        self.ou_range_bound_train_df, self.oU_res_train = self.ou.getRangeBoundFrame(z)
        # apply range bound frame - for test set 
        ou_range_bound_test_df = pd.DataFrame(self.test_residual_predict, columns=['Residuals'])
        self.ou_range_bound_test_df = ou_range_bound_test_df.assign(**self.oU_res_train)
        self.z_norm_test_residuals = (self.train_residuals - self.oU_res_train['mu_e']) / self.ou.sigmaeq
        

    def _simple_returns(self):
        self.returns_train_x = np.log(self.x_train).diff().dropna()
        self.returns_train_y = np.log(self.y_train).diff().dropna()
        self.returns_test_x = np.log(self.x_test).diff().dropna()
        self.returns_test_y = np.log(self.y_test).diff().dropna()

    #just for test train window
    def _normalize_start_1(self, prices):
        return prices/prices.iloc[0]

    def _train_lookback_ols_regression(self):
        """
        This method performs Ordinary Least Squares (OLS) regression on the training data.
        It fits the regression model using y_train as the dependent variable and x_train
        as the independent variable, stores the residuals, beta coefficient, and intercept (c),
        and then predicts residuals for the test set using training model
        """
        print('linear regression OLS lookback window')
        regression = regresion_ols(self.y_train, self.x_train)
        regression.fit()
        self.linear_regression_training_df = regression.df_results
        self.train_residuals = regression.residuals
        self.beta = regression.beta
        self.c = regression.c
        self._test_residuals_predict(regression)

    def _test_residuals_predict(self, regression):
        """
        Predicts the residuals for the test set using the regression model fitted on the training data.
        The residuals are stored in the instance variable test_residual_predict.
        """
        self.test_residual_predict = regression.predict_residuals(self.y_test, self.x_test)

    def _do_lookback_stationarity_checks(self):  
        """
        Performs stationarity checks on the lookback residuals using the Augmented Dickey-Fuller test.
        """
        adf_result, self.adf_lookback_df = ad_fuller_to_df(self.train_residuals)
        self.adf_p_value = adf_result[1]
        critical_values = adf_result[4]
        adf_statistic = adf_result[0]
        print('Ad Fuller P ', self.adf_p_value, 'statistic', adf_statistic, 'critical vals', critical_values)

    def is_valid_pair(self):
        return (self.adf_p_value < self.P_VALUE_THRESHOLD)

    def __repr__(self):
        s = f"Pair [{self.x_train.name}, {self.y_train.name}]"
        s = f"\n\tp-value: {self.adf_p_value}"
        # s += f"\n\tMean crosses: {self.mean_crosses}"
        # s += f"\n\tProfitable trades (%): {self.profitable_trades_perc}"
        # s += f"\n\tAverage holding period (days): {self.average_holding_period}"
        s += f"\n\tPair eligible: {self.is_valid_pair()}"
        return s


