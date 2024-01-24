import numpy as np
from .tool import runRBGA, runRBSLSQP
from arch import arch_model


class RBModel:
    def __init__(self, tmp_close, weight_constraints, actual_data, solution_method=0):
        self.tmp_close = tmp_close
        self.weight_constraints = weight_constraints
        self.actual_data = actual_data
        self.solution_method = solution_method

    def optimize(self):
        cov = (self.tmp_close.pct_change().dropna()).cov() * 252
        
        risk_alloc_data = self.tmp_close
        daily_returns = risk_alloc_data.pct_change().dropna()
        volatility = self.predict_volatility(daily_returns)
        risk_alloc = volatility / volatility.sum()
        
        new_weight = self.get_weight(cov, risk_alloc)
        return new_weight
    
    def get_weight(self, cov, risk_alloc):
        if self.solution_method == 0:
            new_weight, _ = runRBSLSQP(cov=cov, risk_alloc=risk_alloc, bounds=self.weight_constraints)
        elif self.solution_method == 1:
            new_weight, _ = runRBGA(cov=cov, risk_alloc=risk_alloc, bounds=self.weight_constraints)
            new_weight = new_weight[0]
        
        return new_weight
    
    def predict_volatility(self, returns):
        """
        using arch package:
        https://arch.readthedocs.io/en/latest/univariate/introduction.html#arch.univariate.arch_model.vol
        https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html#GARCH-(with-a-Constant-Mean)
        """
        num_assets = returns.shape[1]
        future_volatilities = []
        
        for i in range(num_assets):
            asset_returns = returns.iloc[:, i]
            am = arch_model(asset_returns, vol='GARCH', p=1, q=1)
            res = am.fit(update_freq=10, disp='off')
            forecast = res.forecast(horizon=1)
            future_volatility = forecast.variance.iloc[-1,-1]**0.5
            future_volatilities.append(future_volatility)
        
        volatility = np.array(future_volatilities)
        return volatility
