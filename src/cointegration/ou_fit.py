import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

class OUFit:
    def __init__(self, residuals, tau):
        self.tau = tau
        self.residuals = residuals
        self.half_life = None
        self.theta = None
        self.days = None
        self.sse = None
        self.denom = None
        self.sigmaeq = None
        self.sigmaOU = None
        self.mu = None

    def fit(self, lag=1):
        # Add a constant to the residuals DataFrame for the intercept term
        residuals_df = sm.add_constant(self.residuals.shift(lag).fillna(0))
        # Fit the AR(1) model using OLS
        cointresid_AR1 = OLS(self.residuals, residuals_df).fit()
        # print(cointresid_AR1.summary())
        C, B = cointresid_AR1.params
        self.mu = C / (1 - B)
        self.theta = -np.log(B) / self.tau
        # Calculate the half-life of mean reversion
        self.half_life = np.log(2) / self.theta
        self.days = self.half_life / self.tau
        self.sse = np.sum(cointresid_AR1.resid ** 2)
        self.denom = (1 - np.exp(-2 * self.theta * self.tau))
        self.sigmaeq = np.sqrt(self.sse * self.tau / self.denom)
        self.sigmaOU = self.sigmaeq * np.sqrt(2 * self.theta)

    def getRangeBoundFrame(self, Z=1):
        cointpair_sigma = self.sigmaeq
        OU_res = {
            'mu_e': self.mu,
            'upper': self.mu + Z * cointpair_sigma,
            'lower': self.mu - Z * cointpair_sigma
        }
        cointresid_OUFit = pd.DataFrame(self.residuals, columns=['Residuals'])
        cointresid_OUFit = cointresid_OUFit.assign(**OU_res)
        return cointresid_OUFit, OU_res