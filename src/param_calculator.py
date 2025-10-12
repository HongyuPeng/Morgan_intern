import pandas as pd
import numpy as np
from datetime import datetime

def calculate_jpm_metrics(start_date=None, end_date=None):
    """
    计算JPM股价的波动率、股息率和无风险利率
    
    参数:
    start_date (str): 开始日期，格式为'YYYY-MM-DD', 默认为None表示使用所有数据
    end_date (str): 结束日期，格式为'YYYY-MM-DD', 默认为None表示使用所有数据
    
    返回:
    dict: 包含波动率、股息率、无风险利率和相关数据的字典
    """
    # 读取股价数据（日期降序排列）
    price_data = pd.read_csv('data\input\excel\JPM_ historial_data.csv')
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    
    # 读取股息数据（日期降序排列）
    dividend_data = pd.read_csv('data\input\excel\JPM_dividend.csv')
    dividend_data['Date'] = pd.to_datetime(dividend_data['Date'])
    
    # 读取美债数据（日期降序排列）
    treasury_data = pd.read_csv('data\input\excel\DGS1.csv')
    treasury_data['Date'] = pd.to_datetime(treasury_data['Date'])
    
    # 将所有数据按日期升序排列
    price_data = price_data.sort_values('Date').reset_index(drop=True)
    dividend_data = dividend_data.sort_values('Date').reset_index(drop=True)
    treasury_data = treasury_data.sort_values('Date').reset_index(drop=True)
    
    # 设置日期索引
    price_data.set_index('Date', inplace=True)
    dividend_data.set_index('Date', inplace=True)
    treasury_data.set_index('Date', inplace=True)
    
    # 筛选指定日期范围
    if start_date:
        start_date = pd.to_datetime(start_date)
        price_data = price_data[price_data.index >= start_date]
        dividend_data = dividend_data[dividend_data.index >= start_date]
        treasury_data = treasury_data[treasury_data.index >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        price_data = price_data[price_data.index <= end_date]
        dividend_data = dividend_data[dividend_data.index <= end_date]
        treasury_data = treasury_data[treasury_data.index <= end_date]
    
    # 检查是否有足够的数据
    if len(price_data) < 2:
        return {"error": "所选日期范围内没有足够的价格数据"}
    
    if len(treasury_data) == 0:
        return {"error": "所选日期范围内没有国债数据"}
    
    # 计算波动率（年化）
    returns = np.log(price_data['Close'] / price_data['Close'].shift(1))
    volatility = returns.std() * np.sqrt(252)  # 年化波动率
    
    # 获取最新股价和最新股息
    latest_price = price_data['Close'].iloc[-1]  # 数据现在是按日期升序排列的，取最后一个
    latest_dividend = dividend_data['Dividend'].iloc[-1] if len(dividend_data) > 0 else 0
    
    # 计算股息率
    dividend_yield = (latest_dividend * 4) / latest_price if latest_dividend > 0 else 0
    
    # 计算无风险利率（年化，转换为小数形式）
    risk_free_rate = treasury_data['DGS1'].iloc[-1] / 100
    
    return {
        'sigma': round(volatility, 4),
        'q': round(dividend_yield, 4),
        's0': latest_price,
        'r': round(risk_free_rate, 4)
    }

# 示例用法
if __name__ == "__main__":
    # 计算特定时间范围的指标
    print("\n2023-2024年结果:")
    results = calculate_jpm_metrics('2023-8-23', '2024-8-23')
    print(f"年化波动率: {results['sigma']}")
    print(f"年化股息率: {results['q']*100:.2f}%")
    print(f"无风险利率: {results['r']*100:.2f}%")