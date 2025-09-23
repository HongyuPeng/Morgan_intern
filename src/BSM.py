import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import sys
import os
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)
from src.param_calculator import calculate_jpm_metrics

def bsm_vectorized(s0, r, q, sigma, t, size):
    z = np.random.normal(0, 1, size)
    return s0 * np.exp((r - q - 0.5 * sigma**2) * t + z * sigma * np.sqrt(t))

def simulation(simulation_time=10, base_params = None):
    """
    使用Black-Scholes模型模拟期权价格，计算分析价值和数值价值
    
    参数:
    simulation_time: int, 可选
        模拟次数，默认为10
    s0: float, 可选
        初始股票价格，默认为156.7
    r: float, 可选
        无风险利率，默认为0.0015
    q: float, 可选
        股息率，默认为0.0233
    sigma: float, 可选
        波动率，默认为0.282
    t: float, 可选
        第一个到期时间点（年），默认为0.5
    sp: float, 可选
        执行价格，默认为150
    T: float, 可选
        总到期时间（年），默认为1
    
    返回:
    tuple: (ct_list, pt_list)
        ct_list: ndarray
            看涨期权数据数组，包含以下列:
            - 时间t的股票价格
            - 期权类型('CALL')
            - 时间T的股票价格
            - 数值计算的价值
            - 分析计算的价值
        
        pt_list: ndarray
            看跌期权数据数组，包含以下列:
            - 时间t的股票价格
            - 期权类型('PUT')
            - 时间T的股票价格
            - 数值计算的价值
            - 分析计算的价值
    
    功能说明:
    1. 使用Black-Scholes模型生成时间t时的股票价格
    2. 根据股票价格与执行价格的关系确定期权类型（看涨或看跌）
    3. 使用Black-Scholes公式计算期权的分析价值
    4. 使用蒙特卡洛模拟生成时间T时的股票价格并计算数值价值
    5. 将看涨和看跌期权数据分别整理并返回
    
    注意:
    - 该函数假设股票价格遵循几何布朗运动
    - 分析价值使用Black-Scholes公式直接计算
    - 数值价值通过蒙特卡洛模拟未来收益并折现得到
    """
    # 默认参数设置
    default_params = {
        's0': 156.7,
        'r': 0.0015,
        'q': 0.0233,
        'sigma': 0.282,
        't': 0.5,
        'sp': 150,
        'T': 1
    }
    
    # 更新默认参数
    if base_params is not None:
        default_params.update(base_params)

    s0 = default_params['s0']
    r = default_params['r']
    q = default_params['q']
    sigma = default_params['sigma']
    sp = default_params['sp']
    t = default_params['t']
    T = default_params['T']

    st = bsm_vectorized(s0, r, q, sigma, t, simulation_time)
    
    option_type = np.where(st > sp, 'CALL', 'PUT')

    dt = T - t
    d1 = (np.log(st / sp) + (r - q + 0.5 * sigma**2) * dt / (sigma * np.sqrt(dt)))
    d2 = d1 - sigma * np.sqrt(dt)

    call_mask = (option_type == 'CALL')
    put_mask = (option_type == 'PUT')

    analytical_values = np.zeros(simulation_time)
    analytical_values[call_mask] = st[call_mask] * np.exp(-q * dt) * norm.cdf(d1[call_mask]) - sp * np.exp(-r * dt) * norm.cdf(d2[call_mask])
    analytical_values[put_mask] = sp * np.exp(-r * dt) * norm.cdf(-d2[put_mask]) - st[put_mask] * np.exp(-q * dt) * norm.cdf(-d1[put_mask])

    st_2 = bsm_vectorized(st, r, q, sigma, t, simulation_time)

    future_payoff_call = np.maximum(st_2[call_mask] - sp, 0)
    future_payoff_put = np.maximum(sp - st_2[put_mask], 0)
    
    numerical_values = np.zeros(simulation_time)
    numerical_values[call_mask] = future_payoff_call * np.exp(-r * T)
    numerical_values[put_mask] = future_payoff_put * np.exp(-r * T)

    ct_list = np.column_stack([st[call_mask], option_type[call_mask], st_2[call_mask], numerical_values[call_mask], analytical_values[call_mask]])
    pt_list = np.column_stack([st[put_mask], option_type[put_mask], st_2[put_mask], numerical_values[put_mask], analytical_values[put_mask]])

    return (ct_list, pt_list)

def option_value_vectorized(st_list):
    if st_list.size == 0:
        return 0
    return np.mean(st_list[:, -1].astype(float))

def sensitivity_analysis(param_name, param_range, simulation_time=100000, base_params=None,
                         ct_color='#ea7e36', pt_color='#fcc01e', 
                         figsize=(10, 6), foldername=None):
    """
    执行期权价格的敏感性分析
    
    参数:
    param_name: str
        要分析的参数名称，可选值: 'sigma', 'sp', 'r', 'q', 's0', 't', 'T'
    param_range: array-like
        参数值的范围，用于分析该参数对期权价格的影响
    simulation_time: int, 可选
        每次参数模拟的次数，默认为100000
    base_params: dict, 可选
        基础参数设置，包含所有其他参数的值。如果为None，则使用默认参数
    ct_color: str, 可选
        看涨期权在图表中的颜色，默认为'#ea7e36'
    pt_color: str, 可选
        看跌期权在图表中的颜色，默认为'#fcc01e'
    figsize: tuple, 可选
        图表大小，默认为(10, 6)
    foldername: str, 可选
        保存图表的文件夹路径。如果为None，则根据参数名称自动生成默认路径
    
    返回:
    tuple: (ct_option_prices, pt_option_prices)
        ct_option_prices: ndarray
            看涨期权在不同参数值下的价格数组
        pt_option_prices: ndarray
            看跌期权在不同参数值下的价格数组
    
    功能说明:
    1. 对指定参数在一定范围内进行敏感性分析
    2. 使用蒙特卡洛模拟计算每个参数值对应的看涨和看跌期权价格
    3. 生成并保存敏感性分析图表
    4. 返回两种期权在不同参数值下的价格数组
    """

    # 默认参数设置
    default_params = {
        's0': 156.7,
        'r': 0.0015,
        'q': 0.0233,
        'sigma': 0.282,
        't': 0.5,
        'sp': 150,
        'T': 1
    }
    
    # 更新默认参数
    if base_params is not None:
        default_params.update(base_params)
    
    labels = {
        'sigma': ('Volatility (sigma)', 'Sensitivity Analysis on Sigma'),
        'sp': ('Strike Price', 'Sensitivity Analysis on the Strike Price'),
        'r': ('Risk-free Interest Rate', 'Sensitivity Analysis on the Risk-free Rate'),
        'q': ('Dividend Rate', 'Sensitivity Analysis on the Dividend Rate'),
        's0': ('Initial Stock Price', 'Sensitivity Analysis on Initial Stock Price'),
        't': ('Time to Maturity (t)', 'Sensitivity Analysis on Time to Maturity'),
        'T': ('Total Time (T)', 'Sensitivity Analysis on Total Time')
    }
    
    if param_name not in labels:
        raise ValueError("参数名称必须是 'sigma', 'sp', 'r', 'q', 's0', 't' 或 'T'")
    
    xlabel, title = labels[param_name]
    
    # 生成文件名（不包含路径）
    filename = f"{title.replace(' ', '_')}.png"
    
    # 确定完整保存路径
    if foldername is not None:
        # 确保文件夹路径以斜杠结尾
        if not foldername.endswith('/'):
            foldername += '/'
        save_path = f"{foldername}{filename}"
    else:
        save_path = f"data/output/images/default/{filename}"
    
    ct_option_prices, pt_option_prices = [], []
    
    for value in tqdm(param_range, desc=f"{param_name} 敏感性分析", unit="参数"):
        # 创建参数字典副本
        params = default_params.copy()
        # 更新当前分析的参数值
        params[param_name] = value
        
        # 直接传入simulation_time参数
        ct_list, pt_list = simulation(simulation_time=simulation_time, base_params=params)
        
        ct_option_price = option_value_vectorized(ct_list)
        ct_option_prices.append(ct_option_price)
        
        pt_option_price = option_value_vectorized(pt_list)
        pt_option_prices.append(pt_option_price)
    
    # 绘制图形
    plt.figure(figsize=figsize)
    plt.scatter(x=param_range, y=ct_option_prices, c=ct_color, label='Call Option')
    plt.scatter(x=param_range, y=pt_option_prices, c=pt_color, label='Put Option')
    plt.xlabel(xlabel)
    plt.ylabel('Option Value')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.clf()
    
    return np.array(ct_option_prices), np.array(pt_option_prices)

if __name__ == '__main__':
    # 参数设置
    simulation_time = 1000000

    # The 2nd Six-month Stock Price vs. Payoff
    ct_list, pt_list = simulation(simulation_time=simulation_time)
    plt.figure(figsize=(10, 6))
    plt.scatter(x=ct_list[:, 2].astype(float), y=ct_list[:, 3].astype(float), c='#ea7e36', label='Call Option')
    plt.scatter(x=pt_list[:, 2].astype(float), y=pt_list[:, 3].astype(float), c='#fcc01e', label='Put Option')
    plt.xlabel('Stock Price at Maturity ($)')
    plt.ylabel('Option Payoff ($)')
    plt.title('The 2nd Six-month Stock Price vs. Payoff')
    plt.legend()
    plt.savefig('data/output/images/default/The_2nd_Six-month_Stock_Price_vs._Payoff.png')
    plt.clf()
            
    # 对sigma进行敏感性分析     
    sigma_range = [i/100 for i in range(10, 90)]
    ct_sigma, pt_sigma = sensitivity_analysis('sigma', sigma_range, simulation_time)

    # 对strike price进行敏感性分析
    sp_range = [i for i in range(50, 450)]
    ct_sp, pt_sp = sensitivity_analysis('sp', sp_range, simulation_time)

    # 对risk-free rate进行敏感性分析
    r_range = [i/1000 for i in range(0, 80)]
    ct_r, pt_r = sensitivity_analysis('r', r_range, simulation_time)

    # 对dividend rate进行敏感性分析
    q_range = [i/1000 for i in range(0, 80)]
    ct_q, pt_q = sensitivity_analysis('q', q_range, simulation_time)


    # Alterative Data
    params = calculate_jpm_metrics('2023-8-23', '2024-8-23')
    params['sp'] = round(params['s0'], -1)

    # The 2nd Six-month Stock Price vs. Payoff
    ct_list, pt_list = simulation(simulation_time=simulation_time, base_params=params)
    plt.figure(figsize=(10, 6))
    plt.scatter(x=ct_list[:, 2].astype(float), y=ct_list[:, 3].astype(float), c='#ea7e36', label='Call Option')
    plt.scatter(x=pt_list[:, 2].astype(float), y=pt_list[:, 3].astype(float), c='#fcc01e', label='Put Option')
    plt.xlabel('Stock Price at Maturity ($)')
    plt.ylabel('Option Payoff ($)')
    plt.title('The 2nd Six-month Stock Price vs. Payoff')
    plt.legend()
    plt.savefig('data/output/images/2023-2024/The_2nd_Six-month_Stock_Price_vs._Payoff.png')
    plt.clf()

    # 对sigma进行敏感性分析     
    sigma_range = [i/100 for i in range(10, 90)]
    ct_sigma, pt_sigma = sensitivity_analysis('sigma', sigma_range, simulation_time, base_params=params,foldername='data/output/images/2023-2024')

    # 对strike price进行敏感性分析
    sp_range = [i for i in range(50, 450)]
    ct_sp, pt_sp = sensitivity_analysis('sp', sp_range, simulation_time, base_params=params,foldername='data/output/images/2023-2024')

    # 对risk-free rate进行敏感性分析
    r_range = [i/1000 for i in range(0, 80)]
    ct_r, pt_r = sensitivity_analysis('r', r_range, simulation_time, base_params=params,foldername='data/output/images/2023-2024')

    # 对dividend rate进行敏感性分析
    q_range = [i/1000 for i in range(0, 80)]
    ct_q, pt_q = sensitivity_analysis('q', q_range, simulation_time, base_params=params,foldername='data/output/images/2023-2024')