import math
import re
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
# from hurst import compute_Hc
from statsmodels.regression.linear_model import OLS
# from statsmodels.tsa.stattools import coint
from src.pairs.coint_functions import *
from src.util.DataFetcher import *
# from src.cointegration.linear_regression import regresion_ols
# from src.cointegration.ou_fit import OUFit
from enum import Enum
import numpy as np
from statsmodels.tools.tools import add_constant
import matplotlib.patches as mpatches


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
    def __init__(self, benchmark_series, y_series, x_series, start_full, end_full, start_train, end_train, trading_fee=0.01, freq='daily', normalize_window=True, tau_denom=365, rf=0.02):
        self.y_series = y_series
        self.x_series = x_series
        self.benchmark_series = benchmark_series
        self.y_full = y_series.loc[start_full:end_full].dropna()
        self.x_full = x_series.loc[start_full:end_full].dropna()
        self.benchmark_full = benchmark_series.loc[start_full:end_full].dropna()
  
        self.trading_fee = trading_fee
        self.end_train = end_train
        self.end_full = end_full

        self.rf = rf
        self.freq = freq
 
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
        raw = pd.DataFrame({'price_bmk': self.benchmark_full, 'price_y': self.y_full, 'price_x': self.x_full,'full_resid': self.OU_PARAMS_DICT['full_resid'], 'b': self.OU_PARAMS_DICT['b']})
        raw['return_y'] = np.log(raw['price_y'] / raw['price_y'].shift(1)).fillna(0)
        raw['return_x'] = np.log(raw['price_x'] / raw['price_x'].shift(1)).fillna(0)
        raw['return_spread'] = raw['return_y'] - raw['b'] * raw['return_x']
        raw['return_benchmark'] = np.log(raw['price_bmk'] / raw['price_bmk'].shift(1)).fillna(0)
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

        #  # Add background color for in-sample range (white)
        # ax_stockX.axvspan(in_sample_start, self.end_train, color='white', alpha=0.5)
        # ax_spread.axvspan(in_sample_start, self.end_train, color='white', alpha=0.5)
        # ax_position.axvspan(in_sample_start, self.end_train, color='white', alpha=0.5)

        # Add background color for out-sample range (light green)
        ax_stockX.axvspan(self.end_train, self.end_full, color='lightgreen', alpha=0.5, label='Out-sample')
        ax_spread.axvspan(self.end_train, self.end_full, color='lightgreen', alpha=0.5, label='Out-sample')
        ax_position.axvspan(self.end_train, self.end_full, color='lightgreen', alpha=0.5, label='Out-sample')

        in_sample_patch = mpatches.Patch(color='white', alpha=0.5, label='In-sample')
        out_sample_patch = mpatches.Patch(color='lightgreen', alpha=0.5, label='Out-sample')
        ax_stockX.legend(handles=[in_sample_patch, out_sample_patch])
        ax_spread.legend(handles=[in_sample_patch, out_sample_patch])
        ax_position.legend(handles=[in_sample_patch, out_sample_patch])

        ax_stockX.legend()
        ax_stockY.legend()
        ax_spread.legend()
      
        ax_position.legend()


        fig.set_size_inches(18.5, 10.5, forward=True)

        plt.savefig(self.name + '.jpg', dpi=400)



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

        self.sd = z
        self.slip = slip
        data = self.data
        spread = self.OU_PARAMS_DICT['full_resid']
        mu = self.OU_PARAMS_DICT['mu_e']
        sigma = self.OU_PARAMS_DICT['sigma_eq']
        data['mu'] = mu
        data['sma'] = mu
        data['dist'] = data['full_resid'] - data['mu']

        results = self.data
        self.sd = z
        self.slip = slip
        data = self.data

        upper_bound = mu + sigma * z
        lower_bound = mu - sigma * z

        # add stop loss

        print(upper_bound)
        print(lower_bound)
        data['lower_bound'] = lower_bound
        data['upper_bound'] = upper_bound
        data = data.copy().dropna()
        # # # positions
        # positions
        data['position'] = np.where(data['dist'].shift(1) * data['dist'] < 0, 0, np.nan)
        data['position'] = np.where(data['full_resid'] > data['upper_bound'], -1, data['position'])  # sell signals
        data['position'] = np.where(data['full_resid'] < data['lower_bound'], 1, data['position'])  # buy signals
        data['position'].ffill(inplace=True)  # fill forward na values.
        data['position'].shift(self.slip)  # enter slippage assumption
        data['position'].fillna(0, inplace=True)  # fill na gaps.
        # returns:
        data['strategy_returns'] = (data['position'] * data['return_spread']) + 0  # add zero to avoid negative zeros
        data['fees'] = np.where(data['position'] == data['position'].shift(1), 0, self.trading_fee)
        data['fees'] = data['fees'].fillna(0)
        data['fees'][0] = int(0)  # first obs as it was a bug from the former rule.
        data['net_strategy_returns'] = (data['strategy_returns'] - data['fees']) + 0  # add zero to avoid negative zeros.
        data['creturns'] = data['return_spread'].cumsum().apply(np.exp)  # long spread buy-and-hold
        data['cbmkreturns'] = data['return_benchmark'].cumsum().apply(np.exp)  # long bmk buy-and-hold
        data['cstrategy'] = data['strategy_returns'].cumsum().apply(np.exp)  # our strategy
        data['cnetstrategy'] = data['net_strategy_returns'].cumsum().apply(np.exp)  # our strategy after fees.
        self.results = data
        # absolute performance of the strategy
        aperf = data['cstrategy'].iloc[-1] - 1
        aperf_net = data['cnetstrategy'].iloc[-1] - 1
        # out-/underperformance vs bmk
        operf = aperf - (data['cbmkreturns'].iloc[-1] - 1)
        operf_net = aperf_net - (data['cbmkreturns'].iloc[-1] - 1)
        # vol and sharpe - todo remove
        a_vol = self.an_vol(data['net_strategy_returns'], self.freq)
        sharpe_ratio = self.sharpe(data['net_strategy_returns'], self.freq, self.rf)
        return {'strat cum P&L': round(aperf_net, 4), 'cum perf vs bmk': round(operf_net, 4), 'strat vol': round(a_vol, 4), 'strat sharpe': round(sharpe_ratio, 4)}

    ###just add to plot pair - both of below
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')

        title = '%s | slip = %d, sd = %d' % ('Cumulative Performance', self.slip, self.sd)
        self.results[['creturns', 'cstrategy', 'cnetstrategy', 'cbmkreturns']].plot(title=title, figsize=(10, 6))
        self.results[['return_benchmark', 'net_strategy_returns']].hist(bins=50, figsize=(10, 6))

    def plot_drawdown(self):
        ''' Plots Drawdown metrics (DD and DD Periods) along with a rolling DD line chart and histogram
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')

        cumret = self.results['net_strategy_returns'].cumsum().apply(np.exp).dropna()
        cummax = cumret.cummax()
        drawdown = 100 * abs(cumret / cummax - 1).dropna()
        temp = drawdown[drawdown == 0]
        periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())

        print(pd.DataFrame({'3 Worst DD': sorted(drawdown, reverse=True)[0:3],
                            '3 Worst DD Periods': sorted(periods, reverse=True)[0:3]}))
        print('#' * 50)
        print(pd.DataFrame({'Rolling DrawDown Stats': drawdown}).describe())
        print('#' * 50)
        title = 'Rolling Drawdown | slip = %d, sd = %d' % (self.slip, self.sd)
        drawdown.plot(figsize=(10, 6), title=title)
        pd.DataFrame({'Rolling Drawdown': drawdown}).hist(bins=50, figsize=(10, 6))

    def pl_CAGR(self,x):
        '''% P&L CAGR Return or Annualised Return'''
        y = float((x.index[-1]-x.index[0]).days)/365
        return 100*(x.cumsum().apply(np.exp)[-1]**(1/y)-1)

    def an_vol(self,x,freq):
        ''' Annualised Vol
        '''        
        if freq == 'daily':
            an = 365
        elif freq == 'weekly':
            an = 52
        elif freq == 'monthly':
            an = 12
        elif freq == 'quarterly':
            an = 4
        else:
            an = 1
        return 100*(np.std(x)*np.sqrt(an))


    def sharpe(self, x,freq,rf):
        ''' Sharpe Ratio'''
        if freq == 'daily':
            an = 365
        elif freq == 'weekly':
            an = 52
        elif freq == 'monthly':
            an = 12
        elif freq == 'quarterly':
            an = 4
        else:
            an = 1
        return (self.pl_CAGR(x)-rf)/(self.an_vol(x,freq))
    

    def an_vol(self, x,freq):
        ''' Annualised Vol
        '''        
        if freq == 'daily':
            an = 365
        elif freq == 'weekly':
            an = 52
        elif freq == 'monthly':
            an = 12
        elif freq == 'quarterly':
            an = 4
        else:
            an = 1
        return 100*(np.std(x)*np.sqrt(an))

    def run_strategy_mine(self, z=1, slip=1):
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
        self.sd = z
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

     


        data['spread_return'] = data['return_y'] - data['b'] * data['return_x']
        # Apply the signals to the returns
        # Multiply the returns by the signals to get the strategy's returns
        data['strategy_returns2'] = data['position'].shift(1) * data['spread_return'] + 0
        # The shift accounts for the fact that you can only trade at the next period's open
        # after you observe the signal at the close
        # The first value of 'strategy_returns' will be NaN because of the shift, so fill it with 0
        data['strategy_returns2'].fillna(0, inplace=True)

        # Calculate the cumulative returns of the strategy
        data['cumulative_strategy_returns'] = (1 + data['strategy_returns2']).cumprod() - 1

        # # Initialize the column where the final accumulation will be stored
        # data['accumulated_return'] = 0.0

        # Track the start of a new accumulation period
        start_new_accumulation = True
        accumulated_return = 0.0



        # Now 'aggregated_return_on_change' has the aggregated return on the day of position change, and zero otherwise
        # print(data)
        
        # data['trade_return'] = data['position'] * data['spread_return'] + 0
        # data['cumulative_return'] = data['trade_return'].cumsum()
        
        # raw['return'] = raw['return_y'] - raw['b'] * raw['return_x']

        # data['upper_cross'] = self.__level_crosses(spread, level=upper_bound)
        # data['lower_cross'] = self.__level_crosses(spread, level=lower_bound)
        # # self.upper_stop = self.__level_crosses(spread, level=self.stop_loss)
        # # self.lower_stop = self.__level_crosses(spread, level=-self.stop_loss)
        # data['mean_cross'] = self.__level_crosses(spread, level=mu)
        # # self.mean_crosses = self.mean.count(1) + self.mean.count(-1)

        # returns:
        data['strategy_returns'] = (data['position'] * data['spread_return']) + 0  # add zero to avoid negative zeros
        data['strategy_returns'].fillna(0, inplace=True)
        # data['fees'] = np.where(data['position'] == data['position'].shift(1), 0, self.trading_fee)
        # data['fees'] = data['fees'].fillna(0)
        # data['fees'][0] = int(0)  # first obs as it was a bug from the former rule.
        # data['net_strategy'] = (data['strategy'] - data['fees']) + 0  # add zero to avoid negative zeros.
        data['creturns'] = data['spread_return'].cumsum().apply(np.exp)  # long spread buy-and-hold
        # data['cbmkreturns'] = data['bmk_return'].cumsum().apply(np.exp)  # long bmk buy-and-hold
        data['cstrategy'] = data['strategy_returns'].cumsum().apply(np.exp)  # our strategy
        # data['cnetstrategy'] = data['net_strategy'].cumsum().apply(np.exp)  # our strategy after fees.
        # self.results = data
        # # absolute performance of the sßtrategy
        aperf = data['cstrategy'].iloc[-1] - 1
        # aperf_net = data['cnetstrategy'].iloc[-1] - 1

        # # out-/underperformance vs bmkß
        # operf = aperf - (data['cbmkreturns'].iloc[-1] - 1)
        # operf_net = aperf_net - (data['cbmkreturns'].iloc[-1] - 1)
        # # vol and sharpe:
        # a_vol = an_vol(data['net_strategy'], self.freq)
        # sharpe_ratio = sharpe(data['net_strategy'], self.freq, self.rf)
        # return {'strat cum P&L': round(aperf_net, 4), 'cum perf vs bmk': round(operf_net, 4), 'strat vol': round(a_vol, 4), 'strat sharpe': round(sharpe_ratio, 4)}
        # return {'strat cum P&L': round(aperf_net, 4)}
        print('ab perf', aperf)
        self.results = data
        return data

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
        self.OU_PARAMS_DICT = {'price': full_resid, 'full_resid': full_resid, 'mu_e': mu, 'tau': tau, 'theta': theta,'sigma_OU': sigmaOU, 'sigma_eq': sigmaeq, 'b': self.b, 'half life': half_life, 'days': days}
     

    def _stationarity_check(self, residuals):  
        adf_result, adf_df = ad_fuller_to_df(residuals)
        self.adf_p_value = adf_result[1]
        critical_values = adf_result[4]
        adf_statistic = adf_result[0]
        self.AFF_DICT = {'adf_result': adf_result, 'adf_p_value': self.adf_p_value,'1%': critical_values['1%'], '5%': critical_values['5%'], '10%': critical_values['10%'], 'df': adf_df, 'adf_statistic': adf_statistic}

    def _normalize_start_1(self, prices):
        return prices/prices.iloc[0]
    
    def __repr__(self):
        s = f"Pair [{self.x_train.name}, {self.y_train.name}]"
        s += "\n\tWindow: " + self.window_string
        s += f"\n\tp-value: {self.adf_p_value}"
        s += f"\n\tPair eligible: {self.is_valid_pair()}"
        return s



  