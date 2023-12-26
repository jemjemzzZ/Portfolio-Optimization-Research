import numpy as np
from .tool import get_max_return, get_min_sd, get_max_sharpe


class MVOModel:
    def __init__(self, tmp_close, target_method, target_return, weight_constraints, risk_free_rate):
        self.tmp_close = tmp_close
        self.target_method = target_method
        self.target_return = target_return
        self.weight_constraints = weight_constraints
        self.risk_free_rate = risk_free_rate

    def optimize(self):
        new_weight = None

        mu = np.array(self.tmp_close.tail(1).div(self.tmp_close.iloc[0], axis=1)
                      ** (1 / (len(self.tmp_close) / 252)) - 1)[0]
        S = (self.tmp_close.pct_change().dropna()).cov() * 252

        if self.target_method == "min_volatility":
            new_weight = \
                get_min_sd(target_return=self.target_return, history_return=mu, cov=S, bounds=self.weight_constraints)
        elif self.target_method == "max_return":
            new_weight = \
                get_max_return(target_variance=self.target_return, history_return=mu, cov=S, bounds=self.weight_constraints)
        elif self.target_method == "max_sharpe":
            new_weight = \
                get_max_sharpe(history_return=mu, cov=S, bounds=self.weight_constraints, risk_free_rate=self.risk_free_rate)

        return new_weight

