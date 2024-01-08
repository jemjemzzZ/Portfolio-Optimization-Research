import numpy as np
from pypfopt import black_litterman
from pypfopt import BlackLittermanModel
from .tool import get_max_return, get_min_sd, get_max_sharpe


class BLModel:
    def __init__(self, tmp_close, target_method, target_return, weight_constraints, risk_free_rate, actual_data):
        self.tmp_close = tmp_close
        self.target_method = target_method
        self.target_return = target_return
        self.weight_constraints = weight_constraints
        self.risk_free_rate = risk_free_rate
        self.actual_data = actual_data

    def optimize(self):
        new_weight = None

        mu = np.array(self.tmp_close.tail(1).div(self.tmp_close.iloc[0], axis=1)
                      ** (1 / (len(self.tmp_close) / 252)) - 1)[0]
        S = (self.tmp_close.pct_change().dropna()).cov() * 252
        P, Q = self.generate_pq(self.actual_data)
        
        mcaps ={i:1 for i in list(S.index)}
        delta = 1
        market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
        bl = BlackLittermanModel(S, pi=market_prior,P=P,Q=Q,)
        ret_bl = bl.bl_returns()
        S_bl = bl.bl_cov()

        if self.target_method == "min_volatility":
            new_weight = \
                get_min_sd(target_return=self.target_return, history_return=ret_bl, cov=S_bl, bounds=self.weight_constraints)
        elif self.target_method == "max_return":
            new_weight = \
                get_max_return(target_variance=self.target_return, history_return=ret_bl, cov=S_bl, bounds=self.weight_constraints)
        elif self.target_method == "max_sharpe":
            new_weight = \
                get_max_sharpe(history_return=ret_bl, cov=S_bl, bounds=self.weight_constraints, risk_free_rate=self.risk_free_rate)

        return new_weight
    
    def generate_pq(self, df):
        asset_info = {}
        for asset_code in df.columns:
            first_price = df[asset_code].iloc[0]
            last_price = df[asset_code].iloc[-1]
            asset_info[asset_code] = (first_price, last_price)
        
        views = {asset: (final - initial) / initial for asset, (initial, final) in asset_info.items() if final != initial}

        P = np.zeros((len(views), len(df.columns)))
        Q = np.zeros(len(views))

        for i, (asset, view) in enumerate(views.items()):
            asset_index = list(df.columns).index(asset)
            P[i, asset_index] = 1
            Q[i] = view

        return P, Q