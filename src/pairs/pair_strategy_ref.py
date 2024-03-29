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
from statsmodels.tools.tools import add_constant


class PairStrategy(object):
    P_VALUE_THRESHOLD = 0.05

    #   start: str
    #         start date for data retrieval
    #     end:  str
    #         end date for data retrieval
    #     start_in: str
    #         start date for training spread calc
    #     end_in: str
    #         end date for training spread calc

        # ctf: float 
        # overall equity execution fees per trade (buy or sell) => trading fee + bid-offer + slippage (delay, broker fees, etc)
        # Damodaran: Overall Costs US Large Cap = 0.3%-0.4% vs 3.8%-5.8% US LAGll Cap => ctf = 2% default assumption is reasonable
        # http://people.stern.nyu.edu/adamodar/pdfiles/invphiloh/tradingcosts.pdf
    # crypto 365 tau
    def __init__(self, y_series, x_series, start_full, end_full, start_train, end_train, trading_fee=0.002, normalize_window=True, tau_denom=365):
        self.y_series = y_series
        self.x_series = x_series
        self.y_full = y_series.loc[start_full:end_full].dropna()
        self.x_full = x_series.loc[start_full:end_full].dropna()
        self.trading_fee = trading_fee
 
        if normalize_window :
            self.y_full = self._normalize_start_1(self.y_full)
            self.x_full = self._normalize_start_1(self.x_full)
        
        self.y_train = self.y_full.loc[start_train:end_train].dropna()
        self.x_train = self.x_full.loc[start_train:end_train].dropna()

        self.tau_denom = tau_denom
        self.name = f"{self.y_series.name}, {self.x_series.name}"
        print(f"Pair Created ({self.name})")
        self.window_string = f"[Train start ({start_train}) End train ({end_train}), " + f"Full start ({start_full}) End full ({end_full})]"
        print(self.window_string)
        self._ols()
        self._stationarity_check(self.res_calc) #on the calculated residuals (train data)
        self._ou_fit()
        self.__get_data()

        self.error = False

    def __get_data(self):
        # This is our train (calc residual), and used across whole setto calc full residual
        raw = pd.DataFrame({'price_y': self.y_full, 'price_x': self.x_full,'full_resid': self.OU_PARAMS_DICT['full_resid'], 'b': self.OU_PARAMS_DICT['b']})
        raw['return_y'] = np.log(raw['price_y'] / raw['price_y'].shift(1)).fillna(0)
        raw['return_x'] = np.log(raw['price_x'] / raw['price_x'].shift(1)).fillna(0)
        raw['return'] = raw['return_y'] - raw['b'] * raw['return_x']
        self.data = raw.round(4)

    def is_valid_pair(self, p_value_threshold=P_VALUE_THRESHOLD):
        if self.error:
            return False
        elif self.adf_p_value <= p_value_threshold:
            return True
        return False
  
    def _ols(self):
        x_t = add_constant(self.x_train)  # add intercept = columns of 1s to x_t
        ols = OLS(self.y_train, x_t).fit()  # validate result with statsmodels
        self.c = ols.params[0]
        self.b = ols.params[1]
        x_t = x_t.iloc[:, 1:]  # exclude constant as it will be accounted in c
        self.res_calc = self.y_train - self.c - self.b * x_t[x_t.columns[0]]
        self.res_model = ols.resid
        self.df_results = pd.DataFrame({
            'Estimate': ols.params,
            'SD of Estimate': ols.bse,
            't-Statistic': ols.tvalues,
            'p-value': ols.pvalues
        }) 
        self.OLS_STATS_DICT = {'spread_calc': self.res_calc, 'c': self.c, 'b': self.b}

    # def plot_pair(self):
    #     fig, (ax_stockX, ax_spread) = plt.subplots(2, 1)

    #     ax_stockX.title.set_text("Stocks prices")
    #     ax_spread.title.set_text("Normalized spread")

    #     ax_stockX.plot(self.stockX_trading, color="b", label=self.stockX_trading.name)
    #     ax_stockY = ax_stockX.twinx()
    #     ax_stockY.plot(self.stockY_trading, color="y", label=self.stockY_trading.name)

    #     ax_spread.plot(self.normalized_spread_trading[-self.trading_period:])
    #     ax_spread.axhline(self.trading_bound, linestyle='--', color="g", label="Trading bound")
    #     ax_spread.axhline(-self.trading_bound, linestyle='--', color="g")
    #     ax_spread.axhline(self.stop_loss, linestyle='--', color="r", label="Stop loss")
    #     ax_spread.axhline(-self.stop_loss, linestyle='--', color="r")

    #     ax_spread.plot_date([self.normalized_spread_trading.index[i] for i in self.open_positions],
    #                         [self.normalized_spread_trading[i] for i in self.open_positions], label='Open position',
    #                         marker='^', markeredgecolor='b', markerfacecolor='b', markersize=16)
    #     ax_spread.plot_date([self.normalized_spread_trading.index[i] for i in self.closed_positions],
    #                         [self.normalized_spread_trading[i] for i in self.closed_positions], label='Closed position',
    #                         marker='P', markeredgecolor='g', markerfacecolor='g', markersize=16)
    #     ax_spread.plot_date([self.normalized_spread_trading.index[i] for i in self.stop_losses],
    #                         [self.normalized_spread_trading[i] for i in self.stop_losses], label='Stop loss',
    #                         marker='X', markeredgecolor='r', markerfacecolor='r', markersize=16)

    #     ax_stockX.legend()
    #     ax_stockY.legend()
    #     ax_spread.legend()

    #     fig.set_size_inches(18.5, 10.5, forward=True)
    #     plt.savefig(self.name + '.jpg', dpi=400)
        

    # def plot_data(self):
    #     fig, (ax_stockX, ax_spread) = plt.subplots(2, 1)
    #     ''' Plots original data and trading indicators .
    #     '''
    #     if self.results is None:
    #         print('No results to plot yet. Run a strategy.')
    #     title = '%s | slip = %d, sd = %d' % ('Spread and Position', self.slip, self.z)

    #     # fig, ax1 = plt.subplots()
    #     ax2 = ax1.twinx()
    #     plt.title(title)
    #     ax1.plot(self.results.index, self.results['full_resid'])
    #     ax2.plot(self.results.index, self.results['position'], 'ro')
    #     ax1.set_xlabel('Time')
    #     ax1.set_ylabel('Price', color='g')
    #     ax2.set_ylabel('Position', color='r')
    #     plt.show()

    def plot_pair(self, results):
        fig, (ax_stockX, ax_spread, ax_position) = plt.subplots(3, 1)
        spread = results['full_resid']
        position = results['position']

        upper_bound = results['upper_bound'].iloc[0]
        lower_bound = results['lower_bound'].iloc[0]
        mu = results['mu'].iloc[0]


        ax_stockX.title.set_text("Stocks prices")
        ax_spread.title.set_text("Residual")
   
        ax_spread.plot(spread)
        ax_stockX.plot(self.x_full, color="b", label=self.x_full.name)
        ax_stockY = ax_stockX.twinx()
        ax_stockY.plot(self.y_full, color="y", label=self.y_full.name)



        ax_spread.axhline(upper_bound, linestyle='--', color="g", label="upper bound")
        ax_spread.axhline(lower_bound, linestyle='--', color="r", label="lower bound")
        ax_spread.axhline(mu, linestyle='--', color="orange", label="mu e")


        ax_position.title.set_text("Position")
        ax_position.plot(position, color="g", label=position.name)

        # ax_spread.axhline(self.stop_loss, linestyle='--', color="r", label="Stop loss")
        # ax_spread.axhline(-self.stop_loss, linestyle='--', color="r")

        # ax_spread.plot_date([spread.index[i] for i in self.open_positions],
        #                     [spread[i] for i in self.open_positions], label='Open position',
        #                     marker='^', markeredgecolor='b', markerfacecolor='b', markersize=16)
        # ax_spread.plot_date([self.normalized_spread_trading.index[i] for i in self.closed_positions],
        #                     [self.normalized_spread_trading[i] for i in self.closed_positions], label='Closed position',
        #                     marker='P', markeredgecolor='g', markerfacecolor='g', markersize=16)
        # ax_spread.plot_date([self.normalized_spread_trading.index[i] for i in self.stop_losses],
        #                     [self.normalized_spread_trading[i] for i in self.stop_losses], label='Stop loss',
        #                     marker='X', markeredgecolor='r', markerfacecolor='r', markersize=16)

        ax_stockX.legend()
        ax_stockY.legend()
        ax_spread.legend()
      
        ax_position.legend()


        fig.set_size_inches(18.5, 10.5, forward=True)
        plt.savefig(self.name + '.jpg', dpi=400)


    # def plot_pair(self, results):
    #     fig, (ax_stockX, ax_spread) = plt.subplots(2, 1)
    #     spread = results['full_resid']
    #     upper_bound = results['upper_bound'].iloc[0]
    #     lower_bound = results['lower_bound'].iloc[0]
    #     mu = results['mu'].iloc[0]

    #     ax_stockX.title.set_text("Stocks prices")
    #     ax_spread.title.set_text("Residual")
    #     ax_stockX.plot(self.x_full, color="b", label=self.x_full.name)
    #     ax_stockY = ax_stockX.twinx()
    #     ax_stockY.plot(self.y_full, color="y", label=self.y_full.name)

    #     ax_spread.plot(spread)
    #     ax_spread.axhline(upper_bound, linestyle='--', color="g", label="upper bound")
    #     ax_spread.axhline(lower_bound, linestyle='--', color="r", label="lower bound")
    #     ax_spread.axhline(mu, linestyle='--', color="orange", label="mu e")


    #     # ax_spread.axhline(self.stop_loss, linestyle='--', color="r", label="Stop loss")
    #     # ax_spread.axhline(-self.stop_loss, linestyle='--', color="r")

    #     # ax_spread.plot_date([spread.index[i] for i in self.open_positions],
    #     #                     [spread[i] for i in self.open_positions], label='Open position',
    #     #                     marker='^', markeredgecolor='b', markerfacecolor='b', markersize=16)
    #     # ax_spread.plot_date([self.normalized_spread_trading.index[i] for i in self.closed_positions],
    #     #                     [self.normalized_spread_trading[i] for i in self.closed_positions], label='Closed position',
    #     #                     marker='P', markeredgecolor='g', markerfacecolor='g', markersize=16)
    #     # ax_spread.plot_date([self.normalized_spread_trading.index[i] for i in self.stop_losses],
    #     #                     [self.normalized_spread_trading[i] for i in self.stop_losses], label='Stop loss',
    #     #                     marker='X', markeredgecolor='r', markerfacecolor='r', markersize=16)

    #     ax_stockX.legend()
    #     ax_stockY.legend()
    #     ax_spread.legend()

    #     fig.set_size_inches(18.5, 10.5, forward=True)
    #     plt.savefig(self.name + '.jpg', dpi=400)

    # def plot_strat(self, z=1, slip=):
    #     ''' 
    #     Backtests pair trading strategy.
    #     Params
    #     ------- 
    #     sd  = 1 default. Multiplier to be applied to sigma_eq to trade the spread.
    #     slip = 1 day default. Lag between signal and trade execution.
    #     Output:
    #     'Abs Net P&L | Abs Net P&L vs bmk | An_Vol | Sharpe'
    #     '''
    #     self.z = z
    #     self.slip = slip
    #     data = self.data
    #     spread = self.OU_PARAMS_DICT['full_resid']
    #     mu = self.OU_PARAMS_DICT['mu_e']
    #     sigma = self.OU_PARAMS_DICT['sigma_eq']
    #     data['sma'] = mu
    #     data['dist'] = data['full_resid'] - data['sma']
    #     data['upper'] = mu - sigma * z
    #     data['lower'] = mu + sigma * z
    #     data = data.copy().dropna()
    #     # positions - where does bete come into play
    #     data['position'] = np.where(data['dist'].shift(1) * data['dist'] < 0, 0, np.nan)
    #     data['position'] = np.where(data['full_resid'] > data['upper'], -1, data['position'])  # sell signals
    #     data['position'] = np.where(data['full_resid'] < data['lower'], 1, data['position'])  # buy signals
    #     data['position'].ffill(inplace=True)  # fill forward na values.
    #     data['position'].shift(self.slip)  # enter slippage assumption
    #     data['position'].fillna(0, inplace=True)  # fill na gaps.
    #     # returns:
    #     data['strategy'] = (data['position'] * data['return']) + 0  # add zero to avoid negative zeros
    #     data['fees'] = np.where(data['position'] == data['position'].shift(1), 0, self.trading_fee)
    #     data['fees'] = data['fees'].fillna(0)
    #     data['fees'][0] = int(0)  # first obs as it was a bug from the former rule.
    #     data['net_strategy'] = (data['strategy'] - data['fees']) + 0  # add zero to avoid negative zeros.
    #     data['creturns'] = data['return'].cumsum().apply(np.exp)  # long spread buy-and-hold
    #     # data['cbmkreturns'] = data['bmk_return'].cumsum().apply(np.exp)  # long bmk buy-and-hold
    #     data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)  # our strategy
    #     data['cnetstrategy'] = data['net_strategy'].cumsum().apply(np.exp)  # our strategy after fees.
    #     self.results = data
    #     # absolute performance of the strategy
    #     aperf = data['cstrategy'].iloc[-1] - 1
    #     aperf_net = data['cnetstrategy'].iloc[-1] - 1
    #     # out-/underperformance vs bmk
    #     # operf = aperf - (data['cbmkreturns'].iloc[-1] - 1)
    #     # operf_net = aperf_net - (data['cbmkreturns'].iloc[-1] - 1)
    #     # vol and sharpe:
    #     # a_vol = an_vol(data['net_strategy'], self.freq)
    #     # sharpe_ratio = sharpe(data['net_strategy'], self.freq, self.rf)
    #     # return {'strat cum P&L': round(aperf_net, 4), 'cum perf vs bmk': round(operf_net, 4), 'strat vol': round(a_vol, 4), 'strat sharpe': round(sharpe_ratio, 4)}
    #     return data
        


    def __level_crosses(self, series, level):
        change = []
        for i, el in enumerate(series):
            if i != 0 and el > level and series[i - 1] < level:
                change.append(1)
            elif i != 0 and el < level and series[i - 1] > level:
                change.append(-1)
            else:
                change.append(0)
        return change
    



    def run_strategy(self, z=1, slip=1):
        ''' 
        Backtests pair trading strategy.
        Params
        ------- 
        sd  = 1 default. Multiplier to be applied to sigma_eq to trade the spread.
        slip = 1 day default. Lag between signal and trade execution.
        Output:
        'Abs Net P&L | Abs Net P&L vs bmk | An_Vol | Sharpe'
        '''
        self.z = z
        self.slip = slip
        data = self.data
        spread = self.OU_PARAMS_DICT['full_resid']
        mu = self.OU_PARAMS_DICT['mu_e']
        sigma = self.OU_PARAMS_DICT['sigma_eq']
        data['mu'] = mu
        data['dist'] = data['full_resid'] - data['mu']


        upper_bound = mu + sigma * z
        lower_bound = mu - sigma * z

        # add stop loss

        print(upper_bound)
        print(lower_bound)
        data['lower_bound'] = lower_bound
        data['upper_bound'] = upper_bound
        data = data.copy().dropna()
        # # # positions
        data['position'] = np.where(data['dist'].shift(1) * data['dist'] < 0, 0, np.nan)
        data['position'] = np.where(data['full_resid'] > data['upper_bound'], -1, data['position'])  # sell signals
        data['position'] = np.where(data['full_resid'] < data['lower_bound'], 1, data['position'])  # buy signals

        data['position'].ffill(inplace=True)  # fill forward na values.
        data['position'].shift(self.slip)  # enter slippage assumption
        data['position'].fillna(0, inplace=True)  # fill na gaps.


        # twoooooo
        data['upper_cross'] = self.__level_crosses(spread, level=upper_bound)
        data['lower_cross'] = self.__level_crosses(spread, level=lower_bound)
        # self.upper_stop = self.__level_crosses(spread, level=self.stop_loss)
        # self.lower_stop = self.__level_crosses(spread, level=-self.stop_loss)
        data['mean_cross'] = self.__level_crosses(spread, level=mu)
        # self.mean_crosses = self.mean.count(1) + self.mean.count(-1)


        # # returns:
        # data['strategy'] = (data['position'] * data['return']) + 0  # add zero to avoid negative zeros
        # data['fees'] = np.where(data['position'] == data['position'].shift(1), 0, self.trading_fee)
        # data['fees'] = data['fees'].fillna(0)
        # data['fees'][0] = int(0)  # first obs as it was a bug from the former rule.
        # data['net_strategy'] = (data['strategy'] - data['fees']) + 0  # add zero to avoid negative zeros.
        # data['creturns'] = data['return'].cumsum().apply(np.exp)  # long spread buy-and-hold
        # # data['cbmkreturns'] = data['bmk_return'].cumsum().apply(np.exp)  # long bmk buy-and-hold
        # data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)  # our strategy
        # data['cnetstrategy'] = data['net_strategy'].cumsum().apply(np.exp)  # our strategy after fees.
        self.results = data
        # # absolute performance of the strategy
        # aperf = data['cstrategy'].iloc[-1] - 1
        # aperf_net = data['cnetstrategy'].iloc[-1] - 1
        # out-/underperformance vs bmk
        # operf = aperf - (data['cbmkreturns'].iloc[-1] - 1)
        # operf_net = aperf_net - (data['cbmkreturns'].iloc[-1] - 1)
        # vol and sharpe:
        # a_vol = an_vol(data['net_strategy'], self.freq)
        # sharpe_ratio = sharpe(data['net_strategy'], self.freq, self.rf)
        # return {'strat cum P&L': round(aperf_net, 4), 'cum perf vs bmk': round(operf_net, 4), 'strat vol': round(a_vol, 4), 'strat sharpe': round(sharpe_ratio, 4)}
        # return {'strat cum P&L': round(aperf_net, 4)}

    # #lectures
    def _ou_fit(self):
        tau = 1 / self.tau_denom 
        # OLS regression: OU SDE Solution Regression: e_t = C + B*et_1 + eps_t_tau
        # Add a constant to the residuals DataFrame for the intercept term
        residuals_df = sm.add_constant(self.res_calc.shift(1).fillna(0))      
        # Fit the AR(1) model using OLS
        cointresid_AR1 = OLS(self.res_calc, residuals_df).fit()
        # print(cointresid_AR1.summary())
        C, B = cointresid_AR1.params
        mu = C / (1 - B)
        theta = -np.log(B) / tau
        # Calculate the half-life of mean reversion
        half_life = np.log(2) / theta
        days = half_life / tau
        sse = np.sum(cointresid_AR1.resid**2)
        denom = (1 - np.exp(-2*theta*tau))
        sigmaeq = np.sqrt(sse * tau / denom)
        sigmaOU = sigmaeq * np.sqrt(2*theta) 

        full_resid = self.y_full - self.c - self.b * self.x_full  # using new dates - append to end
        full_z_resid = (full_resid - mu) / sigmaeq
        # output: spread and parameters
        self.OU_PARAMS_DICT = {'price': full_resid, 'full_resid': full_resid, 'mu_e': mu, 'tau': tau, 'theta': theta,'sigma_OU': sigmaOU, 'sigma_eq': sigmaeq, 'b': self.b, 'half life': half_life, 'days': days}
     

    def _stationarity_check(self, residuals):  
        adf_result, adf_df = ad_fuller_to_df(residuals)
        self.adf_p_value = adf_result[1]
        critical_values = adf_result[4]
        adf_statistic = adf_result[0]
        self.AFF_DICT = {'adf_result': adf_result, 'adf_p_value': self.adf_p_value,'1%': critical_values['1%'], '5%': critical_values['5%'], '10%': critical_values['10%'], 'df': adf_df, 'adf_statistic': adf_statistic}

    def _normalize_start_1(self, prices):
        return prices/prices.iloc[0]
    
    #TBC - NEED ALSO THE Z RESID
    def plotRangeBoundFrame(self, Z=1):
        cointpair_sigma = self.OU_PARAMS_DICT['sigma_eq']
        OU_res = {
            'mu_e': self.OU_PARAMS_DICT['mu_e'],
            'upper': self.OU_PARAMS_DICT['mu_e']+ Z * cointpair_sigma,
            'lower': self.OU_PARAMS_DICT['mu_e']- Z * cointpair_sigma
        }
        cointresid_OUFit = pd.DataFrame(self.OU_PARAMS_DICT['full_resid'], columns=['Residuals'])
        cointresid_OUFit = cointresid_OUFit.assign(**OU_res)
        cointresid_OUFit.plot()
        return cointresid_OUFit, OU_res
    
    def __repr__(self):
        s = f"Pair [{self.x_train.name}, {self.y_train.name}]"
        s += "\n\tWindow: " + self.window_string
        s += f"\n\tp-value: {self.adf_p_value}"
        s += f"\n\tPair eligible: {self.is_valid_pair()}"
        return s



  