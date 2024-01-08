import numpy as np
from model.mvo_model import MVOModel
from model.risk_parity_model import RPModel
from model.bl_model import BLModel


def evaluate(historical_data, 
          future_data, 
          weight_constraints,
          model_type='MVO', 
          target_return=0.1, 
          risk_free_rate=0.02):
    
    n = historical_data.shape[1]
    last_weight = [1 / n for _ in range(n)]
    
    if model_type == 'MVO':
        target_method = 'min_volatility'
        model = MVOModel(historical_data, target_method, target_return, weight_constraints, risk_free_rate)
    if model_type == 'RP':
        model = RPModel(historical_data, weight_constraints)
    if model_type == 'BL':
        target_method = 'min_volatility'
        model = BLModel(historical_data, target_method, target_return, weight_constraints, risk_free_rate, future_data)
    
    new_weight = model.optimize()

    if new_weight is None:
        new_weight = last_weight

    predict = check(new_weight, historical_data, risk_free_rate)
    actual = check(new_weight, future_data, risk_free_rate)
    
    return predict, actual


def check(weight, tmp_close, risk_free_rate):
    weight = np.array(weight)
    mu = np.array(tmp_close.tail(1).div(tmp_close.iloc[0], axis=1) ** (1 / (len(tmp_close) / 252)) - 1)[0]

    expected_return = np.dot(mu, weight)
    volatility = np.sqrt(
        np.dot(weight.T, np.dot(tmp_close.cov(), weight)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility

    return expected_return, volatility, sharpe_ratio