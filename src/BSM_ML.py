import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
from tqdm import tqdm
from datetime import datetime

# ===============================
# Monte Carlo Path Generator
# ===============================
def generate_paths(s0, r, q, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    z = np.random.normal(0, 1, (n_paths, n_steps))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    paths = s0 * np.exp(log_paths)
    return paths

def asian_option_value(paths, K, r, T, option_type='call', start_step=0):
    averages = np.mean(paths[:, start_step:], axis=1)
    if option_type.lower() == 'call':
        payoffs = np.maximum(averages - K, 0)
    else:
        payoffs = np.maximum(K - averages, 0)
    remaining_time = T * (paths.shape[1] - 1 - start_step) / (paths.shape[1] - 1)
    return np.exp(-r * remaining_time) * payoffs

def lookback_option_value(paths, K, r, T, option_type='call', start_step=0):
    if option_type.lower() == 'call':
        max_prices = np.max(paths[:, start_step:], axis=1)
        payoffs = np.maximum(max_prices - K, 0)
    else:
        min_prices = np.min(paths[:, start_step:], axis=1)
        payoffs = np.maximum(K - min_prices, 0)
    remaining_time = T * (paths.shape[1] - 1 - start_step) / (paths.shape[1] - 1)
    return np.exp(-r * remaining_time) * payoffs

def generate_training_data(s0, K, r, q, sigma, T, choice_date, n_steps, n_paths, option_type='asian'):
    paths = generate_paths(s0, r, q, sigma, T, n_steps, n_paths)
    choice_step = int(choice_date / T * n_steps)
    if option_type == 'asian':
        call_values = asian_option_value(paths, K, r, T, 'call', start_step=choice_step)
        put_values = asian_option_value(paths, K, r, T, 'put', start_step=choice_step)
    else:
        call_values = lookback_option_value(paths, K, r, T, 'call', start_step=choice_step)
        put_values = lookback_option_value(paths, K, r, T, 'put', start_step=choice_step)
    final_payoffs = np.maximum(call_values, put_values)
    return paths, final_payoffs

# ===============================
# MLP 模型
# ===============================
def build_mlp(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                 loss=Huber(delta=1.0), 
                 metrics=['mae'])
    return model

# ===============================
# 特征工程改进
# ===============================
def create_enhanced_features(paths, s0, K):
    """创建增强特征"""
    n_samples = paths.shape[0]
    n_steps = paths.shape[1]
    
    # 1. 关键统计量
    current_price = paths[:, -1] / s0
    mean_price = np.mean(paths, axis=1) / s0
    max_price = np.max(paths, axis=1) / s0
    min_price = np.min(paths, axis=1) / s0
    volatility = np.std(paths, axis=1) / s0
    
    # 2. 期权相关特征
    moneyness = current_price / (K / s0)  # S/K
    log_moneyness = np.log(moneyness)
    
    # 3. 时间衰减特征（假设均匀时间步）
    time_steps = np.linspace(0, 1, n_steps).reshape(1, -1)
    time_steps = np.repeat(time_steps, n_samples, axis=0)
    
    # 组合所有特征
    enhanced_features = np.column_stack([
        paths,  # 标准化路径
        current_price.reshape(-1, 1),
        mean_price.reshape(-1, 1),
        max_price.reshape(-1, 1),
        min_price.reshape(-1, 1),
        volatility.reshape(-1, 1),
        moneyness.reshape(-1, 1),
        log_moneyness.reshape(-1, 1)
    ])
    
    return enhanced_features

# ===============================
# Pipeline
# ===============================
if __name__ == "__main__":
    # 参数

    # Define parameter ranges for randomization
    param_ranges = {
        's0': [100, 200],
        'K_std_ratio': [0.1, 0.2],  # 保守设置
        'r': [0.001, 0.05],
        'q': [0.0, 0.04],
        'sigma': [0.15, 0.45],
        'T': [0.5, 2.0],
        'choice_date_frac': [0.25, 0.75] # As a fraction of T
    }

    n_total_paths = 2000000
    n_batches = 100
    paths_per_batch = n_total_paths // n_batches
    
    all_X_enhanced = []
    all_paths = []
    all_payoffs = []
    all_params_list = [] # To store the parameters for each path

    for i in tqdm(range(n_batches), desc="Generating batch", unit= 'batch'):
        # Randomly sample parameters for this batch
        s0 = np.random.uniform(*param_ranges['s0'])
        std_ratio = np.random.uniform(*param_ranges['K_std_ratio'])
        K_std = s0 * std_ratio
        
        # 生成K，截断确保合理性
        K = np.random.normal(loc=s0, scale=K_std)
        K = np.clip(K, s0 * 0.5, s0 * 1.5)  # 限制在s0的50%-150%范围内
        T = np.random.uniform(*param_ranges['T'])
        params = {
            's0': s0,
            'K': K,
            'r': np.random.uniform(*param_ranges['r']),
            'q': np.random.uniform(*param_ranges['q']),
            'sigma': np.random.uniform(*param_ranges['sigma']),
            'T': T,
            'n_steps': int(252*T),
            'n_paths': paths_per_batch,
            'option_type': 'asian'
        }
        params['choice_date'] = params['T'] * np.random.uniform(*param_ranges['choice_date_frac'])

        # Generate data with these randomized parameters
        paths, payoffs = generate_training_data(**params)
        X_enhanced = create_enhanced_features(paths, params['s0'], params['K'])

        all_X_enhanced.append(X_enhanced)
        all_paths.append(paths)
        all_payoffs.append(payoffs)
        
        # Store the parameters used for each path in this batch
        for _ in range(paths_per_batch):
            all_params_list.append(params)

    all_X_enhanced = np.concatenate(all_X_enhanced, axis=0).astype(np.float32)
    all_paths = np.concatenate(all_paths, axis=0).astype(np.float32)
    all_payoffs = np.concatenate(all_payoffs, axis=0)

    # Step1: Monte Carlo 数据
    X_train, X_test, y_train, y_test = train_test_split(all_X_enhanced, all_payoffs, test_size=0.2, random_state=42)

    # 初始化scaler
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # 分批进行partial_fit
    batch_size = 50000  # 根据你的数据量调整批次大小

    # 对X进行分批partial_fit
    for i in range(0, len(X_train), batch_size):
        end_idx = min(i + batch_size, len(X_train))
        X_batch = X_train[i:end_idx]
        X_scaler.partial_fit(X_batch)

    # 对y进行分批partial_fit  
    for i in range(0, len(y_train), batch_size):
        end_idx = min(i + batch_size, len(y_train))
        y_batch = y_train[i:end_idx].reshape(-1, 1)
        y_scaler.partial_fit(y_batch)

    # 转换数据
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    joblib.dump(X_scaler, 'scalers/X_scaler.save')
    joblib.dump(y_scaler, 'scalers/y_scaler.save')

    # Step2: 端到端 MLP
    mlp = build_mlp(input_dim=all_X_enhanced.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    mlp.fit(X_train_scaled, y_train_scaled, epochs=300, batch_size=256, validation_split=0.1, verbose=1, callbacks = [es, rlr])
    timestamp = datetime.now().strftime("%m%d_%H%M")
    mlp.save(f'models/chooser_option_mlp_model_{timestamp}.keras')

    # Step3: 预测 & 评估
    y_pred_scaled = mlp.predict(X_test_scaled).flatten()
    y_pred_corrected = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    mse = mean_squared_error(y_test, y_pred_corrected)
    r2 = r2_score(y_test, y_pred_corrected)
    print(f"Test MSE: {mse:.6f}")
    print(f"Test R²: {r2:.4f}")
    print(f"y_test mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")
    print(f"y_pred mean: {y_pred_corrected.mean():.4f}, std: {y_pred_corrected.std():.4f}")

    # Step4: 可视化
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred_corrected, alpha=0.3)
    lims = [min(y_test.min(), y_pred_corrected.min()), max(y_test.max(), y_pred_corrected.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel("Monte Carlo Payoff")
    plt.ylabel("ML Predicted Payoff")
    plt.title("Chooser Option: ML MLP Prediction")
    plt.show()
