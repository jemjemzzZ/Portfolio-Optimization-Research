import numpy as np
from .tool import get_risk_budget
from arch import arch_model


class RBModel:
    def __init__(self, tmp_close, weight_constraints, actual_data, risk_alloc_method=0):
        self.tmp_close = tmp_close
        self.weight_constraints = weight_constraints
        self.actual_data = actual_data
        self.risk_alloc_method = risk_alloc_method

    def optimize(self):
        S = (self.tmp_close.pct_change().dropna()).cov() * 252
        risk_alloc = self.get_risk(self.risk_alloc_method)
        new_weight = get_risk_budget(cov=S, risk_alloc=risk_alloc, bounds=self.weight_constraints)
        return new_weight
    
    def get_risk(self, risk_alloc_method):
        if risk_alloc_method == 1: # use historical data
            risk_alloc_data = self.tmp_close
            daily_returns = risk_alloc_data.pct_change().dropna()
            volatility = daily_returns.std()
        elif risk_alloc_method == 2: # use garch model
            risk_alloc_data = self.tmp_close
            daily_returns = risk_alloc_data.pct_change().dropna()
            volatility = self.predict_volatility(daily_returns)
        else: # use future data
            risk_alloc_data = self.actual_data
            daily_returns = risk_alloc_data.pct_change().dropna()
            volatility = daily_returns.std()
        
        # risk normalisation from volatility
        normalized_risk = volatility / volatility.sum()
        return normalized_risk
    
    def predict_volatility(self, returns):
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
