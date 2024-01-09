from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

from src.pairs.Pair import Pair


class Portfolio:
    def __init__(self, stocks_df):
        if not isinstance(stocks_df, pd.core.frame.DataFrame):
            raise Exception("Symbols must be provided in a Pandas DataFrame")

        self.time_series_df = stocks_df
        self.time_series_df.dropna(inplace=True)
        self.symbols_list = self.time_series_df.columns
        self.all_possible_pairs = list(combinations(self.symbols_list, 2))
        self.selected_pairs = list()

        all_possible_pairs_number = len(self.all_possible_pairs)

        for i, pair_symbols in enumerate(self.all_possible_pairs):
            print(f"{i}/{all_possible_pairs_number}")
            pair = Pair(self.time_series_df[pair_symbols[0]],
                        self.time_series_df[pair_symbols[1]], trading_period=365 * 2)
            if pair.eligible():
                self.selected_pairs.append(pair)

        self.calculate_portfolio_return()

    def calculate_portfolio_return(self):
        data = dict()
        for pair in self.selected_pairs:
            data[pair.name] = pair.returns_series
        df_return = pd.DataFrame(data=data)
        df_return['Return'] = df_return.mean(axis=1)
        df_return['Cumulative Return'] = df_return['Return'].cumsum()
        self.cum_return = df_return['Cumulative Return']

    def plot_portfolio(self):
        plt.title(f"Return of a random portfolio ({len(self.selected_pairs)} stocks)")
        plt.plot(self.cum_return)
        plt.ylabel("Percentage")
        plt.show()