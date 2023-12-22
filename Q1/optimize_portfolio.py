from model.mvo_model import MVOModel
import numpy as np


def optimize_portfolio(model_type, index_list, asset_index):
    # dataset processed
    asset_index = asset_index.loc[:"20230928"]
    asset_index = asset_index.pivot(index='TRADE_DT', columns='INDEX_CODE', values='CLOSE').ffill()[index_list]
    n = asset_index.shape[1]

    # look back period
    backtest_day = 750
    tmp_close = asset_index.tail(backtest_day)

    # weight
    last_weight = [1 / n for _ in range(n)]  # equally weighed
    index_min_weight = [0 for _ in range(n)]
    index_max_weight = [1 for _ in range(n)]
    weight_constraints = list(zip(index_min_weight, index_max_weight))

    # target
    target_method = 'min_volatility'
    target_return = 0.2

    # risk-free rate
    risk_free_rate = 0.02

    # Model Initialization
    if model_type == 'MVO':
        model = MVOModel(tmp_close, target_method, target_return, weight_constraints, risk_free_rate)

    # New Weight
    new_weight = model.optimize()
    if new_weight is None:
        new_weight = last_weight

    # Evaluate Result
    evaluation = evaluate(new_weight, tmp_close, risk_free_rate)

    return new_weight, evaluation


def evaluate(new_weight, tmp_close, risk_free_rate):
    new_weight = np.array(new_weight)
    mu = np.array(tmp_close.tail(1).div(tmp_close.iloc[0], axis=1) ** (1 / (len(tmp_close) / 252)) - 1)[0]

    # expected_return = np.sum(new_weight * mu)
    expected_return = np.dot(mu, new_weight)
    volatility = np.sqrt(
        np.dot(new_weight.T, np.dot(tmp_close.cov(), new_weight)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility

    return expected_return, sharpe_ratio, volatility

