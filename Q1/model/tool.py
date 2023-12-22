import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import cvxpy as cp
from scipy.optimize import minimize


def get_min_sd(target_return, history_return, cov, bounds):
    """
    目标收益率下求最小波动组合
    :params:target_return:目标收益率
    :params:history_return:历史收益率
    :params:cov:方差协方差矩阵
    :param:bounds:约束条件
    """
    bounds_min = [i[0] for i in bounds]
    bounds_max = [i[-1] for i in bounds]
    cov_matrix = np.array(cov)
    num_assets = len(history_return)
    # Define the portfolio weights as variables
    weights = cp.Variable(num_assets)

    # Define the objective function - minimize variance
    port_variance = cp.quad_form(weights, cov_matrix)
    objective = cp.Minimize(port_variance)
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},  # 各类资产比例之和为1
                   {'type': 'eq', 'fun': lambda x: sum(x * history_return) - target_return}  # 达到目标收益率
                   ]
    # Set constraints: expected return and budget constraints
    returns = np.array(history_return)
    expected_return = returns @ weights
    constraints = [expected_return >= target_return, cp.sum(weights) == 1, weights >= bounds_min, weights <= bounds_max]
    # Create the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # Check if the optimization was successful
    if problem.status == cp.OPTIMAL:
        optimal_weights = weights.value
    else:
        optimal_weights = None
    return optimal_weights


def get_max_return(target_variance, history_return, cov, bounds):
    """
    目标波动率下求最大收益组合
    :params:target_variance:目标波动率
    :params:history_return:历史收益率
    :params:cov:方差协方差矩阵
    :param:bounds:约束条件
    """
    bounds_min = [i[0] for i in bounds]
    bounds_max = [i[-1] for i in bounds]
    cov_matrix = np.array(cov)
    num_assets = len(history_return)
    # Define the portfolio weights as variables
    weights = cp.Variable(num_assets)
    returns = np.array(history_return)
    expected_return = -(returns @ weights)
    objective = cp.Minimize(expected_return)
    # Set constraints: expected return and budget constraints
    constraints = [cp.quad_form(weights, cov_matrix) <= target_variance, cp.sum(weights) == 1, weights >= bounds_min,
                   weights <= bounds_max]
    # Create the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # Check if the optimization was successful
    if problem.status == cp.OPTIMAL:
        optimal_weights = weights.value
    else:
        optimal_weights = None
    return optimal_weights


def get_max_sharpe(history_return, cov, bounds, risk_free_rate):
    """
    获得最大夏普组合权重
    """
    cov_matrix = np.array(cov)
    n = len(cov_matrix)
    x0 = [1 / n for _ in range(n)]
    history_return = np.array(history_return)

    def max_sharpe(weights):
        weights = np.array(weights)  # weights为一维数组
        sigma = np.sqrt(np.dot(weights, np.dot(cov, weights)))  # 获取组合标准差
        portfolio_return = np.dot(history_return, weights)  # 获取组合收益
        sharpe = (portfolio_return - risk_free_rate) / sigma
        return -sharpe

    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]  # 各类资产比例之和为1
    solution = minimize(max_sharpe, x0=x0, bounds=bounds, constraints=constraints, method='SLSQP')
    if solution.success:
        final_weights = solution.x  # 权重
    else:
        final_weights = None
    return final_weights
