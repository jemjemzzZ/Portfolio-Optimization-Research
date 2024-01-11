from model.mvo_model import MVOModel
from model.risk_parity_model import RPModel
from model.risk_budget_model import RBModel
import numpy as np


def optimize_portfolio(
        model_type, tmp_close, target_return, risk_free_rate):

    # Weight bounds
    n = tmp_close.shape[1]
    last_weight = [1 / n for _ in range(n)]  # equally weighed
    index_min_weight = [0 for _ in range(n)]
    index_max_weight = [1 for _ in range(n)]
    weight_constraints = list(zip(index_min_weight, index_max_weight))

    # Model initialization
    if model_type == 'MVO':
        target_method = 'min_volatility'
        model = MVOModel(tmp_close, target_method, target_return, weight_constraints, risk_free_rate)
    if model_type == 'RP':
        model = RPModel(tmp_close, weight_constraints)
    if model_type == 'RB': # use historical data for risk allocation
        model = RBModel(tmp_close, weight_constraints)

    # Run model
    new_weight = model.optimize()

    # No optimal solution
    if new_weight is None:
        new_weight = last_weight  # Method 1: use equal weight
        # return None, (-99999, -99999, 99999)  # Method 2: set to extreme value
        # Method 3: use max return/sharpe

    # Evaluate solution
    evaluation = evaluate(new_weight, tmp_close, risk_free_rate)

    return new_weight, evaluation


def evaluate(new_weight, tmp_close, risk_free_rate):
    # Preprocess
    new_weight = np.array(new_weight)
    mu = np.array(tmp_close.tail(1).div(tmp_close.iloc[0], axis=1) ** (1 / (len(tmp_close) / 252)) - 1)[0]

    # Evaluation (return, sharpe, volatility)
    expected_return = np.dot(mu, new_weight)
    volatility = np.sqrt(
        np.dot(new_weight.T, np.dot(tmp_close.cov(), new_weight)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility

    return expected_return, sharpe_ratio, volatility

