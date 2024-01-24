import numpy as np
from model.rb_model import RBModel


def evaluate(historical_data, 
          future_data, 
          weight_constraints,
          model_type='RB', 
          target_return=0.1, 
          risk_free_rate=0.02):
    
    n = historical_data.shape[1]
    last_weight = [1 / n for _ in range(n)]
    
    if model_type == 'RB-SLSQP':
        model = RBModel(historical_data, weight_constraints, future_data, solution_method=0)
    if model_type == 'RB-GA':
        model = RBModel(historical_data, weight_constraints, future_data, solution_method=1)
    
    try:
        new_weight = model.optimize()
    except Exception as e:
        print(e)
        print(f'{model_type} model fails!')
        new_weight = None

    if new_weight is None:
        new_weight = last_weight

    predict = check(new_weight, historical_data, risk_free_rate) # model prediction
    actual = check(new_weight, future_data, risk_free_rate) # actual with model output weights
    
    return predict, actual


def check(weight, tmp_close, risk_free_rate):
    weight = np.array(weight)
    mu = np.array(tmp_close.tail(1).div(tmp_close.iloc[0], axis=1) ** (1 / (len(tmp_close) / 252)) - 1)[0]

    expected_return = np.dot(mu, weight)
    volatility = np.sqrt(
        np.dot(weight.T, np.dot(tmp_close.cov(), weight)))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility

    return expected_return, volatility, sharpe_ratio