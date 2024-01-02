import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import black_litterman
from pypfopt import BlackLittermanModel
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def find_turning_point_self(df, direction, n_days=7, fraction_movement=0.01, threshold_day=14, threshold_rate=0.05):
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


def select_asset(ird_code, df, fraction_movement=0.01, threshold_day=22, n_days=7, threshold_rate=0.05, start_date='20230901'):
    date_record_up = find_turning_point_self(
        df, direction=1, fraction_movement=fraction_movement, threshold_day=threshold_day, n_days=n_days, threshold_rate=threshold_rate)
    
    date_record_down = find_turning_point_self(
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
    
    up_periods = filter_periods_by_start(date_record_up, target_start=start_date)
    down_periods = filter_periods_by_start(date_record_down, target_start=start_date)
    
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


def generate_random_qualified_assets(asset_index, num_limit=[5,10]):
    # data format
    grouped_asset = asset_index.groupby("S_IRDCODE")
    asset_dfs = {ird_code: group for ird_code, group in grouped_asset if len(group) >= 200}
    for ird_code, grouped_df in asset_dfs.items():
        grouped_df['TRADE_DT'] = pd.to_datetime(grouped_df['TRADE_DT'], format='%Y%m%d')
        grouped_df.sort_values(by='TRADE_DT', inplace=True)

    # random generate
    asset_info = {}
    count = 1
    count_limit = random.randint(*num_limit)
    ird_codes = list(asset_dfs.keys())
    while count <= count_limit:
        ird_code = random.choice(ird_codes)
        df = asset_dfs[ird_code]
        up, down = select_asset(ird_code=ird_code, df=df)
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


def generate_random_p_matrix(num_assets, num_views):
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


def gram_schmidt(V):
    U = np.copy(V).astype(np.float64)
    for i in range(1, V.shape[0]):
        for j in range(i):
            U[i] -= np.dot(U[j], U[i]) / np.dot(U[j], U[j]) * U[j]
    for i in range(V.shape[0]):
        U[i] /= np.linalg.norm(U[i])
    return U


def can_be_orthogonalised(P):
    return ~np.isnan(P).any()


def generate_pq(num_assets, num_views, is_orthogonal, actual_returns):
    
    P = generate_random_p_matrix(num_assets, num_views)
    while can_be_orthogonalised(gram_schmidt(P)) != is_orthogonal:
        P = generate_random_p_matrix(num_assets, num_views)
    
    Q = np.array([np.dot(row, actual_returns) for row in P])
    
    return P, Q


def generate_bl_matrix(asset_info):
    views = {asset: (final - initial) / initial for asset, (initial, final) in asset_info.items()}
    P_orthogonal = np.eye(len(views)) # individual return / orthogonal
    Q = np.array(list(views.values())).ravel() # 1D array
    P_non_orthogonal = np.full((len(views), len(views)), 0.5) #  Example where each view partially affects each asset
    
    return P_orthogonal, P_non_orthogonal, Q


def my_test(asset_index, backtest_day, end_date):
    # generate qualified assets for testing
    asset_info = generate_random_qualified_assets(asset_index)
    index_list = list(asset_info.keys())
    print(index_list)

    # initialisation
    num_assets = len(index_list)
    num_views = num_assets
    
    asset_index_copy = asset_index.copy()
    asset_index_copy['TRADE_DT'] = pd.to_datetime(asset_index_copy['TRADE_DT'], format='%Y%m%d')
    asset_index_copy.sort_values(by='TRADE_DT', inplace=True)
    asset_index_copy.set_index('TRADE_DT', inplace=True)
    asset_index_copy = asset_index_copy.loc[:end_date]
    asset_index_copy = asset_index_copy.pivot(columns='S_IRDCODE', values='CLOSE').ffill()[index_list]
    tmp_close = asset_index_copy.tail(backtest_day)
    
    # generate P_orthogonal_actual, Q_actual (actual returns)
    P_orthogonal_actual, P_random, Q_actual = generate_bl_matrix(asset_info)
    
    # generate non-orthogonal PQ
    P, Q = generate_pq(num_assets, num_views, is_orthogonal=False, actual_returns=Q_actual)
    
    # generate orthogonal-enable PQ
    P_can_orthogonal, Q_can_orthogonal = generate_pq(num_assets, num_views, is_orthogonal=True, actual_returns=Q_actual)
    
    # orthogonalisation
    P_orthogonal = gram_schmidt(P_can_orthogonal)
    Q_orthogonal = np.array([np.dot(row, Q_actual) for row in P_orthogonal])
    
    # BL model
    S = (tmp_close.pct_change().dropna()).cov()
    mcaps = {i:1 for i in list(S.index)}
    delta = 1
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


if __name__ == "__main__":
    # read data
    asset_index = pd.read_csv("data/AIDX.csv", encoding='gbk')
    
    # parameters
    backtest_day = 30
    end_date = '20230901'
    num_iteration = 100
    
    # iteration
    results = []
    for i in range(0, num_iteration):
        result = my_test(asset_index, backtest_day, end_date)
        print(result)
        results.append(result)
    
    # visualisation
    line_names = ['Orthogonal/Actual', 'Non-orthogonal', 'Orthogonal', 'Can be orthogonal', 'Not-related(Random)']
    transposed_list = list(zip(*results))

    # Plot each list as a separate line
    plt.figure()
    for line_data, line_name in zip(transposed_list, line_names):
        avg_value = np.mean(line_data)
        plt.plot(line_data, label=f'{line_name} (avg: {avg_value:.5f})')

    # Adding labels and title
    plt.xlabel('Tests')
    plt.ylabel('MSE value')
    plt.title('MSE with actual return')
    plt.legend()

    # Show the plot
    plt.show()
