import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

def bsm_vectorized(s0, r, q, sigma, t, size):
    z = np.random.normal(0, 1, size)
    return s0 * np.exp((r - q - 0.5 * sigma**2) * t + z * sigma * np.sqrt(t))

def simulation_vectorized(simulation_time=10, s0=156.7, r=0.0015, q=0.0233, sigma=0.282, t=0.5, sp=150, T=1):

    st = bsm_vectorized(s0, r, q, sigma, t, simulation_time)
    
    option_type = np.where(st > sp, 'CALL', 'PUT')

    d1 = (np.log(st / sp) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    call_mask = (option_type == 'CALL')
    put_mask = (option_type == 'PUT')

    analytical_values = np.zeros(simulation_time)
    analytical_values[call_mask] = st[call_mask] * np.exp(-q * t) * norm.cdf(d1[call_mask]) - sp * np.exp(-r * t) * norm.cdf(d2[call_mask])
    analytical_values[put_mask] = sp * np.exp(-r * t) * norm.cdf(-d2[put_mask]) - st[put_mask] * np.exp(-q * t) * norm.cdf(-d1[put_mask])

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

def sensitivity_analysis_vectorized(param_name, param_range, simulation_time=100000, 
                                     ct_color='#ea7e36', pt_color='#fcc01e', figsize=(10, 6), filename=None):
    labels = {
        'sigma': ('Volatility (sigma)', 'Sensitivity Analysis on Sigma'),
        'sp': ('Strike Price', 'Sensitivity Analysis on the Strike Price'),
        'r': ('Risk-free Interest Rate', 'Sensitivity Analysis on the Risk-free Rate'),
        'q': ('Dividend Rate', 'Sensitivity Analysis on the Dividend Rate')
    }
    
    if param_name not in labels:
        raise ValueError("参数名称必须是 'sigma', 'sp', 'r' 或 'q'")
    
    xlabel, title = labels[param_name]
    if filename == None:
        filename = f'{title.replace(" ", "_")}.png'
    
    ct_option_prices, pt_option_prices = [], []
    
    for value in tqdm(param_range, desc=f"{param_name} 敏感性分析", unit="参数"):
        # 创建参数字典
        params = {param_name: value}
        ct_list, pt_list = simulation_vectorized(simulation_time=simulation_time, **params)
        
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
    plt.savefig(f'figs/{filename}')
    plt.clf()
    
    return np.array(ct_option_prices), np.array(pt_option_prices)

# 参数设置
simulation_time = 1000000

# The 2nd Six-month Stock Price vs. Payoff
ct_list, pt_list = simulation_vectorized(simulation_time=simulation_time)
plt.figure(figsize=(10, 6))
plt.scatter(x=ct_list[:, 2].astype(float), y=ct_list[:, 3].astype(float), c='#ea7e36', label='Call Option')
plt.scatter(x=pt_list[:, 2].astype(float), y=pt_list[:, 3].astype(float), c='#fcc01e', label='Put Option')
plt.xlabel('Stock Price at Maturity ($)')
plt.ylabel('Option Payoff ($)')
plt.title('The 2nd Six-month Stock Price vs. Payoff')
plt.legend()
plt.savefig('figs/The_2nd_Six-month_Stock_Price_vs._Payoff.png')
plt.clf()

# 对sigma进行敏感性分析
sigma_range = [i/100 for i in range(10, 90)]
ct_sigma, pt_sigma = sensitivity_analysis_vectorized('sigma', sigma_range, simulation_time)

# 对strike price进行敏感性分析
sp_range = [i for i in range(50, 450)]
ct_sp, pt_sp = sensitivity_analysis_vectorized('sp', sp_range, simulation_time)

# 对risk-free rate进行敏感性分析
r_range = [i/1000 for i in range(0, 80)]
ct_r, pt_r = sensitivity_analysis_vectorized('r', r_range, simulation_time)

# 对dividend rate进行敏感性分析
q_range = [i/1000 for i in range(0, 80)]
ct_q, pt_q = sensitivity_analysis_vectorized('q', q_range, simulation_time)