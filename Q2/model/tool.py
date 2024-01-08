import numpy as np
import cvxpy as cp
import warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')


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
    # problem.solve(solver=cp.SCS, eps=1e-8)  # Adjust the tolerance level
    problem.solve
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


def get_risk_parity(cov, bounds):
    """
    获得风险平价模型最优组合权重
    """
    cov_matrix = np.array(cov)
    n = len(cov_matrix)
    x0 = [1 / n for _ in range(n)]

    def risk_budget_objective(weights):
        weights = np.array(weights)  # weights为一维数组
        sigma = np.sqrt(np.dot(weights, np.dot(cov, weights)))  # 获取组合标准差
        # sigma = np.sqrt(weights@cov@weights)
        MRC = np.dot(cov_matrix, weights) / sigma  # MRC = cov@weights/sigma 资产i边际风险贡献
        # MRC = np.dot(weights,cov)/sigma
        TRC = weights * MRC  # 资产i总风险贡献
        delta_TRC = [sum((i - TRC) ** 2) for i in TRC]
        return sum(delta_TRC)

    # cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]  # 各类资产比例之和为1
    solution = minimize(risk_budget_objective, x0=x0, bounds=bounds, constraints=constraints, method='SLSQP')
    if solution.success:
        final_weights = solution.x  # 权重
    else:
        final_weights = None
    return final_weights


def get_risk_budget(cov, risk_alloc, bounds):
    """
    获得风险预算模型最优组合权重
    :params:cov:方差协方差矩阵
    :params:risk_alloc:目标风险贡献比例列表
    :bounds:资产权重比例限制
    """
    cov_matrix = np.array(cov)
    n = len(cov_matrix)
    x0 = [1 / n for _ in range(n)]

    def volatility(weights):
        # 返回组合波动率
        weights = np.array(weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return vol

    def risk_contribution(weights):
        # 返回边际风险贡献度
        weights = np.array(weights)
        vol = volatility(weights)
        mrc = np.dot(cov_matrix, weights) / vol
        trc = np.multiply(mrc, weights)
        return trc

    def risk_parity(weights):
        # 返回最小化函数
        vol = volatility(weights)
        risk_target_pct = np.array(risk_alloc)
        risk_target = np.multiply(vol, risk_target_pct)
        trc = risk_contribution(weights)
        J = np.sqrt(sum(np.square(trc - risk_target)))
        return J

    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]  # 各类资产比例之和为1
    solution = minimize(risk_parity, x0=x0,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    if solution.success:
        final_weights = solution.x  # 权重
    else:
        final_weights = None
    return final_weights
