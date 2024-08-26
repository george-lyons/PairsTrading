from unittest import TestCase

import math
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from hurst import compute_Hc
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint
from src.cointegration.functions import ad_fuller,linear_regression,pairwise_engle_granger_coint,ad_fuller_to_df

# This is a sample Python script.
from datetime import datetime
import time
import pf as pf

from src.pairs import *
import yfinance as yf

from src.pairs.pairs_strategy import *
from src.util.DataFetcher import *

class TestPair(TestCase):


    def test_ols_same(self):
        tickers = ["BTC-USD", "ETH-USD"]

        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 1, 27)
        tickData = get_all_adjusted_close_data(start_date, end_date, tickers)
        tickData = tickData.dropna()
        print(tickData.shape)
        # Most Recent Data
        tickData.head()

        print('linear regression OLS lookback window')
        regression_model,self.regression_ln_df,residuals = linear_regression(tickData['BTC-USD'], tickData['ETH-USD'])
        self.c = regression_model.params.iloc[0]
        self.beta = regression_model.params.iloc[1]
        self.lb_residuals = residuals

        print(self.beta)
        
        # ### observe - for the look forward period we would use regression calc ????
        # self.beta2 = OLS(self.look_back_trading_y, self.look_back_trading_x).fit().params[0]
        # self.spread = self.look_back_trading_y -  self.beta2  * self.look_back_trading_x
        # self.normalized_spread_trading = (self.spread - self.spread.mean()) / self.spread.std() 
