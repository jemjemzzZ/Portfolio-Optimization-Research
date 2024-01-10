import numpy as np
from .tool import get_risk_budget


class RBModel:
    def __init__(self, tmp_close, weight_constraints, actual_data, risk_alloc_method=0):
        self.tmp_close = tmp_close
        self.weight_constraints = weight_constraints
        self.actual_data = actual_data
        self.risk_alloc_method = 0

    def optimize(self):
        S = (self.tmp_close.pct_change().dropna()).cov() * 252
        risk_alloc = self.get_risk(self.risk_alloc_method)
        
        new_weight = get_risk_budget(cov=S, risk_alloc=risk_alloc, bounds=self.weight_constraints)
        print(new_weight)
        return new_weight
    
    def get_risk(self, risk_alloc_method):
        if risk_alloc_method == 1:
            risk_alloc_data = self.tmp_close
        else:
            risk_alloc_data = self.actual_data
        daily_returns = risk_alloc_data.pct_change().dropna()
        volatility = daily_returns.std()
        normalized_risk = volatility / volatility.sum()
        return normalized_risk
