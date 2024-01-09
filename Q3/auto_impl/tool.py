import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import black_litterman
from pypfopt import BlackLittermanModel
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


"""
Find turning point
"""
def find_turning_point(df, 
                       direction, 
                       n_days=7, 
                       fraction_movement=0.01, 
                       threshold_day=22, 
                       threshold_rate=0.05):
    
    """寻找资产CLOSE曲线的显著转折点，包括上涨下跌。
    其核心逻辑为从某一时间点开始，持续向后判断是否有连续变化，直到不满足输入条件。

    :param df: 资产数据
    :param direction: 变化方向为上升(1)或者下降(-1)
    :param n_days: n天内有存在变化, defaults to 7
    :param fraction_movement: n天内的变化幅度阈值, defaults to 0.01
    :param threshold_day: 整体变化时间长度最低限制, defaults to 22
    :param threshold_rate: 整体变化幅度最低限制, defaults to 0.05
    :return: 返回List，其包含多段变化时间区间，以数组形式呈现，(start_date, end_date)
    """    
    
    df = df.reset_index(drop=True)
    record = []
    
    i = 0  # start index
    while i < len(df.index):
        record_i = i
        pattern_find = False
        
        for n in range(1, n_days+1):
            try:
                current_close = df.loc[i, 'CLOSE']
                ndays_close = df.loc[i+n, 'CLOSE']
                if (ndays_close - current_close) * direction >= (fraction_movement * current_close):
                    pattern_find = True
                    i += n
                    n = 0
            except:
                pass
            
        if pattern_find:
            start_date = df.loc[record_i, 'TRADE_DT']
            end_date = df.loc[i, 'TRADE_DT']
            start_close = df.loc[record_i, 'CLOSE']
            end_close = df.loc[i, 'CLOSE']
            if (i - record_i) >= threshold_day \
                or (end_close - start_close) * direction >= (threshold_rate * start_close):
                record.append((start_date, end_date))
            else:
                i = record_i
        i += 1
        
    return record


"""
Filter qualified asset period
"""
def select_asset(ird_code, 
                 df, 
                 fraction_movement=0.01, 
                 threshold_day=22, 
                 n_days=7, 
                 threshold_rate=0.05, 
                 start_date='20230901',
                 tolerance_days=5):
    
    """选择满足要求的变化区间，通过使用find_turning_point()

    :param ird_code: 资产代码(未使用)
    :param df: 资产数据
    :param fraction_movement: 见find_turning_point(), defaults to 0.01
    :param threshold_day: 见find_turning_point(), defaults to 22
    :param n_days: 见find_turning_point(), defaults to 7
    :param threshold_rate: 见find_turning_point(), defaults to 0.05
    :param start_date: 变化的起始时间, defaults to '20230901'
    :param tolerance_days: 起始时间的可调整范围，+-N天, defaults to 5
    :return: 返回两个List，第一个列表为上升变化的多个时间区间，第二个列表为下降变化的多个时间区间
    """    
    
    date_record_up = find_turning_point(
        df, direction=1, fraction_movement=fraction_movement, threshold_day=threshold_day, n_days=n_days, threshold_rate=threshold_rate)
    
    date_record_down = find_turning_point(
        df, direction=-1, fraction_movement=fraction_movement, threshold_day=threshold_day, n_days=n_days, threshold_rate=threshold_rate)
    
    def filter_periods_by_start(periods, target_start, tolerance_days=5):
        target_start = pd.to_datetime(target_start)
        start_range = target_start - pd.Timedelta(days=tolerance_days)
        end_range = target_start + pd.Timedelta(days=tolerance_days)
        
        qualified_periods = []
        for start, end in periods:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            if start_range <= start <= end_range:
                qualified_periods.append((start, end))
        
        return qualified_periods
    
    up_periods = filter_periods_by_start(date_record_up, target_start=start_date, tolerance_days=tolerance_days)
    down_periods = filter_periods_by_start(date_record_down, target_start=start_date, tolerance_days=tolerance_days)
    
    def add_close_to_periods(df, periods):
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
        if df.index.name != 'TRADE_DT':
            df = df.set_index('TRADE_DT', drop=False)
        
        enhanced_periods = []
        for start, end in periods:
            start_close = df.at[start, 'CLOSE'] if start in df.index else None
            end_close = df.at[end, 'CLOSE'] if end in df.index else None
            start_str = start.strftime('%Y-%m-%d') if start_close is not None else None
            end_str = end.strftime('%Y-%m-%d') if end_close is not None else None
            enhanced_period = ((start_str, start_close), (end_str, end_close))
            enhanced_periods.append(enhanced_period)
        
        return enhanced_periods
    
    enhanced_up_periods = add_close_to_periods(df=df, periods=up_periods)
    enhanced_down_periods = add_close_to_periods(df=df, periods=down_periods)
    
    return enhanced_up_periods, enhanced_down_periods


"""
Generate random qualified assets with number limit
"""
def generate_random_qualified_assets(asset_dfs, 
                                     num_limit=[5,10], 
                                     fraction_movement=0.01, 
                                     threshold_day=22, 
                                     n_days=7, 
                                     threshold_rate=0.05, 
                                     start_date='20230901',
                                     tolerance_days=5):
    
    """随机生成多个符合变化要求的资产的变化时间段

    :param asset_dfs: 多个资产数据
    :param num_limit: 随机生成数量范围, defaults to [5,10]
    :param fraction_movement: 见select_asset(), defaults to 0.01
    :param threshold_day: 见select_asset(), defaults to 22
    :param n_days: 见select_asset(), defaults to 7
    :param threshold_rate: 见select_asset(), defaults to 0.05
    :param start_date: 见select_asset(), defaults to '20230901'
    :param tolerance_days: 见select_asset(), defaults to 5
    :return: 返回字典，key为每个资产代码，value为每个资产的变化信息：（start_close_price, end_close_price）
    """    
    
    # random generate
    asset_info = {}
    count = 1
    count_limit = random.randint(*num_limit)
    ird_codes = list(asset_dfs.keys())
    while count <= count_limit:
        ird_code = random.choice(ird_codes)
        df = asset_dfs[ird_code]
        up, down = select_asset(ird_code=ird_code, df=df, 
                                fraction_movement=fraction_movement, threshold_day=threshold_day, 
                                n_days=n_days, threshold_rate=threshold_rate, start_date=start_date, tolerance_days=tolerance_days)
        if len(up) != 0 or len(down) != 0:
            if len(up) > 0:
                start_close = up[0][0][1]
                end_close = up[0][1][1]
                asset_info[ird_code] = (start_close, end_close)
            elif len(down) > 0:
                start_close = down[0][0][1]
                end_close = down[0][1][1]
                asset_info[ird_code] = (start_close, end_close)
            count += 1
    
    return asset_info


"""
Generate random assets without any limitation on views
"""
def generate_random_assets(asset_dfs,
                           num_limit=[5,10],
                           start_date='20230901',
                           end_date='20230928'):
    
    """随机挑选多个资产

    :param asset_dfs: 多个资产数据
    :param num_limit: 随机生成数量范围_, defaults to [5,10]
    :param start_date: 变化起点时间, defaults to '20230901'
    :param end_date: 变化截止时间, defaults to '20230928'
    :return: 返回字典，key为每个资产代码，value为每个资产的变化信息：（start_close_price, end_close_price）
    """    
    
    asset_info = {}
    count = 1
    count_limit = random.randint(*num_limit)
    ird_codes = list(asset_dfs.keys())
    
    while count <= count_limit:
        ird_code = random.choice(ird_codes)
        df = asset_dfs[ird_code]
        
        try:
            start_close = df.loc[df['TRADE_DT'] == start_date, 'CLOSE'].values[0]
            end_close = df.loc[df['TRADE_DT'] == end_date, 'CLOSE'].values[0]
        except Exception as e:
            continue
        
        asset_info[ird_code] = (start_close, end_close)
        count += 1
    
    return asset_info


"""
Genearte random P matrix
"""
def generate_random_p_matrix(num_assets, num_views):
    
    """随机生成规范的观点矩阵P

    :param num_assets: 资产数量
    :param num_views: 观点数量
    :return: 观点矩阵P
    """    
    
    P = np.zeros((num_views, num_assets))
    
    for i in range(num_views):
        row_sum = random.choice([0, 1])
        
        if row_sum == 1:
            idx = random.randint(0, num_assets-1)
            P[i, idx] = 1
        else:
            idx1, idx2 = random.sample(range(num_assets), 2)
            P[i, idx1] = 1
            P[i, idx2] = -1

    return P


"""
Matrix orthogonalisation (gram schmidt)
"""
def gram_schmidt(V):
    
    """矩阵正交化处理

    :param V: 矩阵
    :return: 正交化后结果
    """    
    
    U = np.copy(V).astype(np.float64)
    
    for i in range(1, V.shape[0]):
        for j in range(i):
            U[i] -= np.dot(U[j], U[i]) / np.dot(U[j], U[j]) * U[j]
            
    for i in range(V.shape[0]):
        U[i] /= np.linalg.norm(U[i])
        
    return U


"""
Check if matrix can be orthogonalised
"""
def can_be_orthogonalised(P):
    
    """判断矩阵是否可以被正交化

    :param P: 待检验矩阵
    :return: 是否可被正交化
    """    
    
    return ~np.isnan(P).any()


"""
Check if the matrix is orthogonal
"""
def is_orthogonal(matrix, tol=1e-6):
    
    """判断矩阵是否是正交矩阵（测试用）

    :param matrix: 待验矩阵
    :param tol: tolerance, defaults to 1e-6
    :return: 是否正交
    """    
    
    if matrix.shape[0] != matrix.shape[1]:
        return False  # Must be a square matrix

    identity = np.eye(matrix.shape[0])  # Create an identity matrix of the same size
    return np.allclose(matrix.T @ matrix, identity, atol=tol) and np.allclose(matrix @ matrix.T, identity, atol=tol)


"""
Generate qualified PQ (can be orthogonal or not)
"""
def generate_pq(num_assets, num_views, can_be_orthogonal, actual_returns):
    
    """根据实际回报随机生成合理的P,Q矩阵

    :param num_assets: 资产数量
    :param num_views: 观点数量
    :param can_be_orthogonal: P是否需要可被正交化
    :param actual_returns: 资产的实际汇报
    :return: P,Q矩阵
    """    
    
    P = generate_random_p_matrix(num_assets, num_views)
    
    # keep looping
    while can_be_orthogonalised(gram_schmidt(P)) != can_be_orthogonal:
        P = generate_random_p_matrix(num_assets, num_views)
    
    Q = np.array([np.dot(row, actual_returns) for row in P])
    
    return P, Q


"""
Generate orthogonal P with Q
"""
def generate_orthogonal_pq(num_assets, num_views, actual_returns):
    
    """生成正交P,Q矩阵，根据实际回报（测试用）

    :param num_assets: 资产数量
    :param num_views: 观点数量
    :param actual_returns: 实际回报
    :return: 正交的P，以及对应Q
    """    
    
    P = generate_random_p_matrix(num_assets, num_views)
    
    # keep looping
    while is_orthogonal(P) == False:
        P = generate_random_p_matrix(num_assets, num_views)
    
    Q = np.array([np.dot(row, actual_returns) for row in P])
    
    return P, Q


"""
Genearte actual BL table based on asset info
"""
def generate_bl_matrix(asset_info):
    
    """根据资产实际信息，生成标准P，Q.
    其中P为单位矩阵，即每一资产的独立变化信息。

    :param asset_info: 多个资产变化信息，字典
    :return: 标准P,Q,以及一个无相关性P(用于测试)
    """    
    
    views = {asset: (final - initial) / initial for asset, (initial, final) in asset_info.items()}
    
    P_orthogonal = np.eye(len(views))  # individual return & orthogonal
    Q = np.array(list(views.values())).ravel()  # 1D array
    P_non_orthogonal = np.full((len(views), len(views)), 0.5)  # Example where each view partially affects each asset (non-relative)
    
    return P_orthogonal, P_non_orthogonal, Q


"""
Auto MSE calculation
"""
def mse_calculation(asset_info, S, mcaps, delta, num_assets, num_views):
    
    """计算各种情况P，Q在经过BL计算的，与实际回报的MSE比较

    :param asset_info: 多个资产信息
    :param S: 历史波动
    :param mcaps: 市场权重
    :param delta: risk aversion parameter
    :param num_assets: 资产数量
    :param num_views: 观点数量
    :return: 各种不同情况的P对应的MSE数值
    """    
    
    # generate P_orthogonal_actual, Q_actual (actual returns)
    P_orthogonal_actual, P_random, Q_actual = generate_bl_matrix(asset_info)
    
    # generate non-orthogonal PQ
    P, Q = generate_pq(num_assets, num_views, can_be_orthogonal=False, actual_returns=Q_actual)
    
    # generate can-be-orthogonalised PQ
    P_can_orthogonal, Q_can_orthogonal = generate_pq(num_assets, num_views, can_be_orthogonal=True, actual_returns=Q_actual)
    
    # orthogonalisation
    P_orthogonal = gram_schmidt(P_can_orthogonal)
    Q_orthogonal = np.array([np.dot(row, Q_actual) for row in P_orthogonal])
    
    # BL model
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    
    # Orthogonoal, Actual
    bl_orthogonal_actual = BlackLittermanModel(S, pi=market_prior, P=P_orthogonal_actual, Q=Q_actual)
    ret_bl_orthogonal_actual = bl_orthogonal_actual.bl_returns()
    mse_orthogonal_actual = mean_squared_error(ret_bl_orthogonal_actual, Q_actual)

    # Non-orthogonal
    bl_non_orthogonal = BlackLittermanModel(S, pi=market_prior, P=P, Q=Q)
    ret_bl_non_orthogonal = bl_non_orthogonal.bl_returns()
    mse_non_orthogonal = mean_squared_error(ret_bl_non_orthogonal, Q_actual)

    # Orthogonal
    bl_orthogonal = BlackLittermanModel(S, pi=market_prior, P=P_orthogonal, Q=Q_orthogonal)
    ret_bl_orthogonal = bl_orthogonal.bl_returns()
    mse_orthogonal = mean_squared_error(ret_bl_orthogonal, Q_actual)

    # Can be orthogonalised
    bl_can_orthogonal = BlackLittermanModel(S, pi=market_prior, P=P_can_orthogonal, Q=Q_can_orthogonal)
    ret_bl_can_orthogonal = bl_can_orthogonal.bl_returns()
    mse_can_orthogonal = mean_squared_error(ret_bl_can_orthogonal, Q_actual)

    # Random
    bl_random = BlackLittermanModel(S, pi=market_prior, P=P_random, Q=Q_actual)
    ret_bl_random = bl_random.bl_returns()
    mse_random = mean_squared_error(ret_bl_random, Q_actual)
    
    return (mse_orthogonal_actual, mse_non_orthogonal, mse_orthogonal, mse_can_orthogonal, mse_random)


## simple test
if __name__ == "__main__":
    
    print("Test1")
    P = generate_random_p_matrix(5, 5)
    print(P)
    
    print("Test2")
    P = generate_random_p_matrix(3, 4)
    print(P)
    
    print("Test3")
    P, Q = generate_pq(5, 5, True, [0.1, 0.1, 0.1, 0.1, 0.1])
    print(P)
    print(Q)
    
    print("Test4")
    P_gs = gram_schmidt(P)
    Q_gs = np.array([np.dot(row, [0.1, 0.1, 0.1, 0.1, 0.1]) for row in P_gs])
    print(P_gs)
    print(Q_gs)
    print(is_orthogonal(P_gs))
    
    print("Test5")
    P, Q = generate_pq(5, 5, False, [0.1, 0.2, 0.3, 0.4, 0.5])
    print(P)
    print(Q)
    P_gs = gram_schmidt(P)
    print(P_gs)
    print(is_orthogonal(P_gs))
    
    print("Test6")
    P, Q = generate_orthogonal_pq(5, 5, [0.1, 0.2, 0.3, 0.4, 0.5])
    print(P)
    print(Q)
    print(is_orthogonal(P))
    
    print("Test7")
    asset_info = {
        'h20025.CSI': (5669.2288, 4705.2669),
        '399244.SZ': (640.136, 600.1125),
        '931646CNY01.CSI': (1344.0437, 1247.5575),
        'h40006.SH': (5912.7329, 6335.0402),
        '983087.CNI': (4835.5539, 4471.1056)
    }
    P, P_random, Q = generate_bl_matrix(asset_info)
    print(P)
    print(P_random)
    print(Q)
    
    pass