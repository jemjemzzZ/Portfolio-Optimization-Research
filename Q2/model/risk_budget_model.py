import numpy as np
from .tool import get_risk_budget


class RBModel:
    def __init__(self, tmp_close, weight_constraints, actual_data):
        self.tmp_close = tmp_close
        self.weight_constraints = weight_constraints
        self.actual_data = actual_data

    def optimize(self):
        S = (self.tmp_close.pct_change().dropna()).cov() * 252
        risk_alloc = self.get_risk()
        
        new_weight = get_risk_budget(cov=S, risk_alloc=risk_alloc, bounds=self.weight_constraints)
        return new_weight
    
    def get_risk(self):
        daily_returns = self.actual_data.pct_change().dropna()
        volatility = daily_returns.std()
        normalized_risk = volatility / volatility.sum()
        return normalized_risk
