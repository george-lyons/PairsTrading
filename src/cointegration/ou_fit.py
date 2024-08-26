import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


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

    def fit(self):
        # Add a constant to the residuals DataFrame for the intercept term
        residuals_df = sm.add_constant(self.residuals.shift(1).fillna(0))
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
        self.OU_PARAMS_DICT = {'mu_e': self.mu_e, 'tau': self.tau, 'theta': self.theta,'sigma_OU': self.sigma_OU, 'sigma_eq': self.sigma_eq, 'half life': self.half_l, 'days': self.days}

    def fit2(self):
         # OLS regression: OU SDE Solution Regression: e_t = C + B*et_1 + eps_t_tau
        res_t = self.residuals [1:]
        res_t_1 = self.residuals.shift(1).dropna()
        x = add_constant(res_t_1)  # add intercept = columns of 1s to x_t
        x.rename(columns={0: 'res_t_1'}, inplace=True)
        ols_r = OLS(res_t, x).fit()
        # Backtesting Parameters
        self.mu_e = ols_r.params[0] / (1 - ols_r.params[1])  # equilibrium level = C/(1-B)
        self.theta = - np.log(ols_r.params[1]) / self.tau  # speed of reversion = - log(B)/tau
        self.half_l = np.log(2) / self.theta  # half life
        
        self.days = self.half_l / self.tau
        self.sse = np.sum(ols_r.resid ** 2)
        self.denom = (1 - np.exp(-2 * self.theta * self.tau))
        self.sigma_eq = np.sqrt(self.sse * self.tau / self.denom)
        self.sigma_OU = self.sigma_eq * np.sqrt(2 * self.theta)
        self.OU_PARAMS_DICT2 = {'mu_e': self.mu_e, 'tau': self.tau, 'theta': self.theta,'sigma_OU': self.sigma_OU, 'sigma_eq': self.sigma_eq, 'half life': self.half_l, 'days': self.days}

    def getRangeBoundFrame(self, Z=1):
        cointpair_sigma = self.sigmaeq
        self.OU_res = {
            'mu_e': self.mu,
            'upper': self.mu + Z * cointpair_sigma,
            'lower': self.mu - Z * cointpair_sigma
        }
        cointresid_OUFit = pd.DataFrame(self.residuals, columns=['Residuals'])
        cointresid_OUFit = cointresid_OUFit.assign(**self.OU_res)
        return cointresid_OUFit, self.OU_res
    