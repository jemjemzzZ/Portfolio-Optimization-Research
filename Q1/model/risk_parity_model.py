import numpy as np
from .tool import get_risk_parity


class RPModel:
    def __init__(self, tmp_close, weight_constraints):
        self.tmp_close = tmp_close
        self.weight_constraints = weight_constraints

    def optimize(self):
        S = (self.tmp_close.pct_change().dropna()).cov() * 252
        new_weight = get_risk_parity(cov=S, bounds=self.weight_constraints)
        return new_weight
