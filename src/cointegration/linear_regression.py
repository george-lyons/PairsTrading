import pandas as pd
import statsmodels.tsa.stattools as ts 
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

class   regresion_ols:

    def __init__(self, Y, X, constant=True):
        self.Y=Y
        self.X=X
        self.fitted = False
        self.beta = None
        self.c = None
        self.constant = constant
        if constant :
            X = sm.add_constant(X)  # Adding a constant for the intercept
        self.model = OLS(Y, X)

    def fit(self):
        self.model = self.model.fit()
        self.df_results = pd.DataFrame({
            'Estimate': self.model.params,
            'SD of Estimate': self.model.bse,
            't-Statistic': self.model.tvalues,
            'p-value': self.model.pvalues
        })
        self.summary = self.model.summary()
        self.residuals =  pd.DataFrame(self.model.resid)
        self.residuals.columns = ['Residuals'] 
        self.fitted = True
        self.c = self.model.params[0]
        self.beta = self.model.params[1]
        print('Beta', self.beta, 'Constant', self.c)

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model is not fitted.")
        if self.constant :
            x_predict = sm.add_constant(X)  # Adding a constant term to the predictor
        predicted_y = self.model.predict(x_predict)
        return predicted_y

    def predict_residuals(self, Y_test, X_test):
        Y_pred = self.predict(X_test)   
        res_pred = Y_test - Y_pred
        return res_pred
    
    def direction_accuracy(Y_test, Y_pred):
        return np.sum(np.sign(Y_pred) == np.sign(Y_test)) / len(Y_test) * 100
    
    def r2_score(Y_test, Y_pred):
        return r2_score(Y_test, Y_pred)


    