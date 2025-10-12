import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input, BatchNormalization, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers
import joblib
from tqdm import tqdm


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

# -------------------------
# 修改 generate_training_data：返回每条路径对应的参数向量
# -------------------------
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

    # create per-path params array (shape: n_paths x n_param_features)
    # normalize raw params to reasonable scales? we'll StandardScale them later
    choice_frac = choice_date / T
    params_array = np.tile(np.array([s0, K, r, q, sigma, T, choice_frac], dtype=np.float32), (n_paths, 1))
    return paths, final_payoffs, params_array


# ===============================
# MLP 模型
# ===============================
def build_mlp(input_dim):
    weight_decay = 1e-5
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(256, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.08))

    model.add(Dense(128, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.06))

    model.add(Dense(64, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(32, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(1, activation='linear'))
    # 用稍大起点 lr，配合 ReduceLROnPlateau
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=Huber(delta=1.0),
                  metrics=['mae'])
    return model

# ===============================
# 特征工程改进
# ===============================
# ---------- 替换：更鲁棒的特征工程 ----------
def create_enhanced_features(paths, s0, K):
    """增强特征：保留原始路径 + 多个统计量 + 最近returns + max drawdown + time-weighted avg"""
    n_samples, n_steps = paths.shape
    # 标准统计量（相对 s0）
    current_price = paths[:, -1] / s0
    mean_price = np.mean(paths, axis=1) / s0
    max_price = np.max(paths, axis=1) / s0
    min_price = np.min(paths, axis=1) / s0

    # moneyness (safe)
    moneyness = (paths[:, -1] / s0) / (K / s0)  # = S/K
    # clip to avoid log(0) or extreme
    moneyness_clipped = np.clip(moneyness, 1e-6, 1e6)
    log_moneyness = np.log(moneyness_clipped)

    # recent returns: last 1, 5, 20 steps (relative)
    def rel_return(arr, lag):
        return (arr[:, -1] - arr[:, max(0, n_steps-lag)]) / arr[:, max(0, n_steps-lag)]
    ret_1 = rel_return(paths, 1)
    ret_5 = rel_return(paths, 5 if n_steps >= 5 else 1)
    ret_20 = rel_return(paths, 20 if n_steps >= 20 else n_steps-1)

    # max drawdown (fraction)
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (running_max - paths) / running_max
    max_drawdown = np.max(drawdowns, axis=1)
    max_drawdown = np.nan_to_num(max_drawdown)

    # time-weighted avg (more weight to recent)
    weights = np.linspace(0.1, 1.0, n_steps)  # small->large
    tw_avg = (paths * weights).sum(axis=1) / weights.sum()

    # normalize tw_avg by s0 for scale-consistency
    tw_avg = tw_avg / s0

    # Optionally include normalized raw path slice (e.g., every 4th step to reduce dim)
    stride = max(1, n_steps // 64)  # limit raw path features to ~64 cols
    path_subsample = paths[:, ::stride] / s0

    enhanced_features = np.column_stack([
        path_subsample,  # subsampled path
        current_price,
        mean_price,
        max_price,
        min_price,
        moneyness,
        log_moneyness,
        ret_1, ret_5, ret_20,
        max_drawdown,
        tw_avg
    ])
    return enhanced_features.astype(np.float32)


# ===============================
# Pipeline
# ===============================
# ------------------ 替换版主流程脚本 ------------------
if __name__ == '__main__':
    import tensorflow as tf
    import torch
    from datetime import datetime
    import os

    # GPU 检查（保留你的版本）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # 修改：generate_training_data 返回 params_array（见上文）
    # 确保你已经用上文中修改过的 generate_training_data 定义

    # 参数范围
    param_ranges = {
        's0': [100, 200],
        'K_std_ratio': [0.1, 0.2],
        'r': [0.001, 0.05],
        'q': [0.0, 0.04],
        'sigma': [0.15, 0.45],
        'T': [0.5, 2.0],
        'choice_date_frac': [0.25, 0.75]
    }

    # 小规模示例先跑通（建议先用 200k，再扩展）
    n_total_paths = 2000000  # <<-- 为测试先用小规模，正式可增大
    n_batches = 100
    paths_per_batch = n_total_paths // n_batches

    all_X_enhanced = []
    all_payoffs = []
    all_params_list = []

    for i in range(n_batches):
        s0 = float(np.random.uniform(*param_ranges['s0']))
        std_ratio = float(np.random.uniform(*param_ranges['K_std_ratio']))
        K_std = s0 * std_ratio
        K = float(np.random.normal(loc=s0, scale=K_std))
        K = float(np.clip(K, s0 * 0.5, s0 * 1.5))
        r = float(np.random.uniform(*param_ranges['r']))
        q = float(np.random.uniform(*param_ranges['q']))
        sigma = float(np.random.uniform(*param_ranges['sigma']))
        T = float(np.random.uniform(*param_ranges['T']))
        choice_frac = float(np.random.uniform(*param_ranges['choice_date_frac']))
        choice_date = T * choice_frac

        params = {
            's0': s0,
            'K': K,
            'r': r,
            'q': q,
            'sigma': sigma,
            'T': T,
            'n_steps': 252,
            'n_paths': paths_per_batch,
            'option_type': 'asian',
            'choice_date': choice_date
        }

        paths, payoffs, params_array = generate_training_data(**params)
        X_enhanced = create_enhanced_features(paths, s0, K)  # subsampled path + stats

        all_X_enhanced.append(X_enhanced)
        all_payoffs.append(payoffs)
        all_params_list.append(params_array)  # array shape (paths_per_batch, 7)

    # 合并
    all_X_enhanced = np.concatenate(all_X_enhanced, axis=0).astype(np.float32)
    all_payoffs = np.concatenate(all_payoffs, axis=0).astype(np.float32)
    all_params_array = np.concatenate(all_params_list, axis=0).astype(np.float32)  # (N,7)

    # 将 params 拼入特征
    X_with_params = np.hstack([all_X_enhanced, all_params_array])  # shape (N, d)
    print("Feature shape:", X_with_params.shape, "Labels shape:", all_payoffs.shape)

    # 目标变换：log1p
    y_transformed = np.log1p(all_payoffs)

    # 简单 OOD: 把高 sigma (>0.40) 留作 OOD 测试（示例）
    sigma_vals = all_params_array[:, 4]
    ood_mask = sigma_vals > 0.40
    if ood_mask.sum() < 1000:
        # 若 OOD 样本太少，使用随机拆分
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_with_params, y_transformed, test_size=0.2, random_state=42)
        X_val = X_test[:len(X_test)//2]; y_val = y_test[:len(y_test)//2]
        X_test = X_test[len(X_test)//2:]; y_test = y_test[len(y_test)//2:]
    else:
        X_ood = X_with_params[ood_mask]
        y_ood = y_transformed[ood_mask]
        X_rest = X_with_params[~ood_mask]
        y_rest = y_transformed[~ood_mask]
        X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.1, random_state=42)
        X_test, y_test = X_ood, y_ood

    # Scalers: 对 X 做 StandardScaler（分批 partial_fit 以节省内存/适应大数据）
    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler()
    batch_size = 50000
    for i in range(0, X_train.shape[0], batch_size):
        X_scaler.partial_fit(X_train[i:i+batch_size])

    X_train_scaled = X_scaler.transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)

    # 对 y (log-space) 可选地使用 scaler；这里直接使用 log1p 结果，不再额外 scale
    y_train_scaled = y_train
    y_val_scaled = y_val
    y_test_scaled = y_test

    # 构建并训练模型（输入维度按 X_train_scaled）
    input_dim = X_train_scaled.shape[1]
    mlp = build_mlp(input_dim=input_dim)

    # 增强正则化以提升泛化（可调）
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7)

    history = mlp.fit(X_train_scaled, y_train_scaled, epochs=200, batch_size=256,
                      validation_data=(X_val_scaled, y_val_scaled),
                      verbose=1, callbacks=[es, rlr])

    # 预测并反变换
    y_pred_t = mlp.predict(X_test_scaled).flatten()
    y_pred = np.expm1(y_pred_t)   # 回到 payoff 原始尺度
    y_test_orig = np.expm1(y_test_scaled)

    # 评估
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    print(f"[RESULT] Test MSE: {mse:.6f}, R2: {r2:.4f}")
    print(f"y_test mean: {y_test_orig.mean():.4f}, std: {y_test_orig.std():.4f}")
    print(f"y_pred mean: {y_pred.mean():.4f}, std: {y_pred.std():.4f} ")

    # 保存 scalers & model
    timestamp = datetime.now().strftime("%m%d_%H%M")
    os.makedirs('scalers', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    X_scaler_path = f'scalers/X_scaler_{timestamp}.save'
    joblib.dump(X_scaler, X_scaler_path)
    print(f"✅ X_scaler saved to {X_scaler_path}")

    model_path = f'models/chooser_option_mlp_model_{timestamp}.h5'
    mlp.save(model_path)  # 使用 .h5 格式兼容性最强
    print(f"✅ Model saved to {model_path}")

    # Step 4: 可视化（两张图并排显示）
    import matplotlib.pyplot as plt

    # ---- 反变换到原始尺度 ----
    y_true = np.expm1(y_test_scaled)
    y_pred_final = np.expm1(mlp.predict(X_test_scaled).flatten())

    # ---- 创建画布 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ===== 左图：预测 vs Monte Carlo =====
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred_final, alpha=0.25, s=8)
    lims = [min(y_true.min(), y_pred_final.min()), max(y_true.max(), y_pred_final.max())]
    ax1.plot(lims, lims, 'r--', lw=1)
    ax1.set_xlabel("Monte Carlo Payoff (True)")
    ax1.set_ylabel("ML Predicted Payoff")
    ax1.set_title("Chooser Option: MLP Prediction vs Monte Carlo")
    ax1.axis('equal')
    ax1.grid(alpha=0.3)

    # ===== 右图：误差分布 =====
    ax2 = axes[1]
    errors = y_pred_final - y_true
    ax2.hist(errors, bins=50, alpha=0.7, color='steelblue')
    ax2.set_xlabel("Prediction Error (Pred - True)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    ax2.grid(alpha=0.3)

    # ---- 调整布局并显示 ----
    plt.tight_layout()
    plt.show()
