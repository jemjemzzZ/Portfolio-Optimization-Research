import numpy as np
from model.mvo_model import MVOModel
from model.risk_parity_model import RPModel
from model.bl_model import BLModel
from model.risk_budget_model import RBModel


def evaluate(historical_data, 
          future_data, 
          weight_constraints,
          model_type='MVO', 
          target_return=0.1, 
          risk_free_rate=0.02):
    
    n = historical_data.shape[1]
    last_weight = [1 / n for _ in range(n)]
    
    if model_type == 'MVO': # mean-variance optimization model
        target_method = 'min_volatility'
        model = MVOModel(historical_data, target_method, target_return, weight_constraints, risk_free_rate)
    if model_type == 'RP': # risk parity model
        model = RPModel(historical_data, weight_constraints)
    if model_type == 'BL': # Black-Litterman model
        target_method = 'min_volatility'
        model = BLModel(historical_data, target_method, target_return, weight_constraints, risk_free_rate, future_data)
    if model_type == 'RB': # use future data to calculate risk allocation
        model = RBModel(historical_data, weight_constraints, future_data, risk_alloc_method=0)
    if model_type == 'RB-H': # use historical data to calculate risk allocation
        model = RBModel(historical_data, weight_constraints, future_data, risk_alloc_method=1)
    if model_type == 'RB-G': # use GARCH model to predict risk allocation
        model = RBModel(historical_data, weight_constraints, future_data, risk_alloc_method=2)
    
    try:
        new_weight = model.optimize()
    except Exception as e:
        print(e)
        print(f'{model_type} model fails!')
        new_weight = None

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