import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys
from time import time

# ==== 路径与环境设置 / Path & Environment Setup ====
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定 GPU 设备 (Set CUDA device)

# ==== 自定义模块导入 / Import Custom Modules ====
from src.param_calculator import calculate_rolling_periods, calculate_jpm_metrics


# ==========================================================
# 1️⃣ 基础函数 (Monte Carlo Path Simulation & Payoff)
# ==========================================================
def generate_paths(s0, r, q, sigma, T, n_steps, n_paths):
    """生成股票价格路径 (几何布朗运动)."""
    dt = T / n_steps
    z = np.random.normal(0, 1, (n_paths, n_steps))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
    return s0 * np.exp(log_paths)


def asian_option_value(paths, K, r, T, option_type='call', start_step=0):
    """计算亚洲期权价值（平均价格期权）."""
    averages = np.mean(paths[:, start_step:], axis=1)
    if option_type.lower() == 'call':
        payoffs = np.maximum(averages - K, 0)
    else:
        payoffs = np.maximum(K - averages, 0)

    remaining_time = T * (paths.shape[1] - 1 - start_step) / (paths.shape[1] - 1)
    return np.exp(-r * remaining_time) * payoffs


def generate_training_data(s0, K, r, q, sigma, T, choice_date, n_steps, n_paths, option_type='asian'):
    """生成训练样本 (路径 + payoff + 参数矩阵)."""
    paths = generate_paths(s0, r, q, sigma, T, n_steps, n_paths)
    choice_step = int(choice_date / T * n_steps)

    if option_type == 'asian':
        call_values = asian_option_value(paths, K, r, T, 'call', start_step=choice_step)
        put_values = asian_option_value(paths, K, r, T, 'put', start_step=choice_step)
    else:
        call_values = asian_option_value(paths, K, r, T, 'call', start_step=choice_step)
        put_values = asian_option_value(paths, K, r, T, 'put', start_step=choice_step)

    final_payoffs = np.maximum(call_values, put_values)
    choice_frac = choice_date / T
    params_array = np.tile(np.array([s0, K, r, q, sigma, T, choice_frac], dtype=np.float32), (n_paths, 1))
    return paths, final_payoffs, params_array


# ==========================================================
# 2️⃣ 特征工程模块
# ==========================================================
def create_robust_features(paths, s0, K, params_array):
    """提取鲁棒性特征，包括统计特征、动量指标和时间加权价格."""
    n_samples, n_steps = paths.shape

    # --- 基础价格特征 ---
    current_price = paths[:, -1] / s0
    moneyness = current_price / (K / s0)
    log_moneyness = np.log(np.clip(moneyness, 1e-6, 1e6))

    # --- 路径统计 ---
    mean_price = np.mean(paths, axis=1) / s0
    max_price = np.max(paths, axis=1) / s0
    min_price = np.min(paths, axis=1) / s0

    # --- 实现波动率 ---
    path_returns = np.diff(np.log(np.clip(paths, 1e-6, None)), axis=1)
    realized_vol = np.std(path_returns, axis=1) * np.sqrt(252)

    # --- 技术指标 ---
    lookback_short = min(10, n_steps)
    sma_short = np.mean(paths[:, -lookback_short:], axis=1) / s0
    sma_long = np.mean(paths, axis=1) / s0
    momentum = (paths[:, -1] - paths[:, max(0, n_steps - lookback_short)]) / paths[:, max(0, n_steps - lookback_short)]

    # --- 最大回撤 ---
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (running_max - paths) / (running_max + 1e-8)
    max_drawdown = np.max(drawdowns, axis=1)

    # --- 时间加权均价 ---
    weights = np.linspace(0.1, 1.0, n_steps)
    time_weighted_avg = np.average(paths, axis=1, weights=weights) / s0

    # --- 关键时间点价格 ---
    if n_steps >= 5:
        time_indices = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
        key_prices = paths[:, time_indices] / s0
    else:
        key_prices = paths / s0

    # --- 汇总所有特征 ---
    features = np.column_stack([
        current_price, moneyness, log_moneyness, mean_price,
        max_price, min_price, realized_vol, sma_short,
        sma_long, momentum, max_drawdown, time_weighted_avg,
        key_prices, params_array
    ])

    return features.astype(np.float32)


# ==========================================================
# 3️⃣ 数据生成策略
# ==========================================================

def generate_training_data_with_modes(n_total_paths=400000, 
                                     banks=['GS', 'BAC', 'WFC', 'C', 'MS'],
                                     data_mode='mixed',
                                     enhanced=False):
    """
    精简版多银行数据生成器
    
    参数:
    n_total_paths: 总路径数
    banks: 银行列表
    data_mode: 数据模式 - 'historical_only', 'synthetic_only', 'mixed'
    enhanced: 是否使用增强历史数据（添加扰动）
    """
    n_batches = 100
    paths_per_batch = n_total_paths // n_batches
    
    # 确定批次分配
    if data_mode == 'historical_only':
        hist_batches, rand_batches = 100, 0
    elif data_mode == 'synthetic_only':
        hist_batches, rand_batches = 0, 100
    else:  # mixed
        hist_batches, rand_batches = 50, 50
    
    print(f"模式: {data_mode}, 历史批次: {hist_batches}, 随机批次: {rand_batches}")
    
    all_X, all_y, all_params = [], [], []
    
    # 加载历史数据
    historical_batches = []
    if hist_batches > 0:
        for bank in banks:
            try:
                data = calculate_rolling_periods(bank)
                if not data.empty:
                    # 简单采样，每个银行至少分配一些批次
                    samples = data.sample(min(len(data), hist_batches//len(banks)+1), 
                                         replace=True, random_state=42)
                    for _, row in samples.iterrows():
                        historical_batches.append((bank, row))
                    print(f"✅ {bank}: {len(samples)}条数据")
            except Exception as e:
                print(f"❌ {bank}加载失败: {e}")
    
    # 生成历史数据批次
    for i in range(min(hist_batches, len(historical_batches))):
        bank, row = historical_batches[i]
        
        if enhanced:
            # 增强模式：添加扰动
            s0 = float(row['s0']) * np.random.uniform(0.95, 1.05)
            sigma = max(0.05, float(row['sigma']) * np.random.uniform(0.9, 1.1))
            q = max(0.0, float(row['q']) * np.random.uniform(0.8, 1.2))
            r = max(0.001, float(row['r']) * np.random.uniform(0.9, 1.1))
        else:
            # 普通模式：直接使用历史数据
            s0 = float(row['s0'])
            sigma = float(row['sigma'])
            q = float(row['q'])
            r = float(row['r'])
        
        # 随机化其他参数
        K = s0 * np.random.uniform(0.9, 1.1)
        T = float(np.random.uniform(0.8, 1.2))
        choice_date = T * np.random.uniform(0.4, 0.6)
        
        params = dict(s0=s0, K=K, r=r, q=q, sigma=sigma, T=T,
                      n_steps=252, n_paths=paths_per_batch,
                      option_type='asian', choice_date=choice_date)
        
        try:
            paths, payoffs, params_array = generate_training_data(**params)
            features = create_robust_features(paths, s0, K, params_array)
            all_X.append(features)
            all_y.append(payoffs)
            all_params.append(params_array)
        except Exception as e:
            print(f"历史批次 {i} 失败: {e}")
            continue
    
    # 生成随机数据批次
    for i in range(rand_batches + max(0, hist_batches - len(historical_batches))):
        s0 = float(np.random.uniform(90, 250))
        K = s0 * np.random.uniform(0.9, 1.1)
        r = float(np.random.uniform(0.0001, 0.055))
        
        # 随机股息率
        rand = np.random.random()
        if rand < 0.1:
            q = 0.0
        elif rand < 0.7:
            q = float(np.random.uniform(0.0, 0.04))
        else:
            q = float(np.random.uniform(0.04, 0.1))
            
        sigma = float(np.random.uniform(0.1, 0.5))
        T = float(np.random.uniform(0.8, 1.2))
        choice_date = T * np.random.uniform(0.4, 0.6)
        
        params = dict(s0=s0, K=K, r=r, q=q, sigma=sigma, T=T,
                      n_steps=252, n_paths=paths_per_batch,
                      option_type='asian', choice_date=choice_date)
        
        try:
            paths, payoffs, params_array = generate_training_data(**params)
            features = create_robust_features(paths, s0, K, params_array)
            all_X.append(features)
            all_y.append(payoffs)
            all_params.append(params_array)
        except Exception as e:
            print(f"随机批次 {i} 失败: {e}")
            continue

    if not all_X:
        raise ValueError("未成功生成任何数据")

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    params = np.concatenate(all_params)
    
    print(f"✅ 成功生成 {X.shape[0]} 条样本")
    return X, y, params

# ==========================================================
# 4️⃣ 自定义变换与损失函数
# ==========================================================
class SignedLogTransform:
    """对数符号变换: T(y)=sign(y)*log(1+|y|/c)"""
    def __init__(self, c=1.0):
        self.c = float(c)
    def transform(self, y):
        return tf.sign(y) * tf.math.log(1.0 + tf.abs(y) / self.c)
    def inverse(self, z):
        return tf.sign(z) * (self.c * (tf.math.exp(tf.abs(z)) - 1.0))
    def log_abs_det_jacobian(self, y):
        return -tf.math.log(self.c + tf.abs(y) + 1e-8)


def gaussian_nll_transformed(transform):
    """高斯负对数似然损失 (带变换)."""
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, (-1,))
        mu, log_sigma = y_pred[:, 0], y_pred[:, 1]
        sigma = tf.nn.softplus(log_sigma) + 1e-6
        z = transform.transform(y_true)
        nll_z = 0.5 * (tf.math.log(2.0 * np.pi) + 2.0 * tf.math.log(sigma) + tf.square((z - mu) / sigma))
        log_jac = transform.log_abs_det_jacobian(y_true)
        return tf.reduce_mean(nll_z - log_jac)
    return loss


def mae_mu(transform):
    """MAE (逆变换后的均值预测误差)."""
    def metric(y_true, y_pred):
        mu_z = y_pred[:, 0]
        y_hat = transform.inverse(mu_z)
        return tf.reduce_mean(tf.abs(y_true - y_hat))
    return metric


def mse_mu(transform):
    """MSE (逆变换后的均值预测误差)."""
    def metric(y_true, y_pred):
        mu_z = y_pred[:, 0]
        y_hat = transform.inverse(mu_z)
        return tf.reduce_mean(tf.square(y_true - y_hat))
    return metric


# ==========================================================
# 5️⃣ 模型结构定义
# ==========================================================
def build_improved_mlp(input_dim):
    """构建改进版 MLP 模型 (输出μ与logσ)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256), BatchNormalization(), LeakyReLU(0.1), Dropout(0.05),
        Dense(256), BatchNormalization(), LeakyReLU(0.1), Dropout(0.05),
        Dense(128), BatchNormalization(), LeakyReLU(0.1),
        Dense(2, activation='linear')
    ])
    return model


# ==========================================================
# 6️⃣ 训练函数
# ==========================================================
def train_model(data_type='mixed', enhanced=True, n_total_paths=400000):
    """训练模型的主函数"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ 使用 GPU 训练")
    else:
        print("⚙️ 使用 CPU 训练")

    try:
        print(f"\n=== Step 1: 数据生成 (模式: {data_type}, 增强: {enhanced}) ===")
        X, y, params = generate_training_data_with_modes(
            n_total_paths=n_total_paths, 
            data_mode=data_type,
            enhanced=enhanced
        )

        print("\n=== Step 2: 数据集划分 ===")
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

        print("\n=== Step 3: 特征标准化 ===")
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        print("\n=== Step 4: 构建与编译模型 ===")
        transform = SignedLogTransform()
        model = build_improved_mlp(X_train_s.shape[1])
        model.compile(
            optimizer=Adam(learning_rate=5e-5, amsgrad=True, clipnorm=1.0),
            loss=gaussian_nll_transformed(transform),
            metrics=[mae_mu(transform), mse_mu(transform)]
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, min_delta=5e-5),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=5e-7, verbose=1)
        ]

        print("\n=== Step 5: 模型训练 ===")
        timestamp = datetime.now().strftime("%m%d_%H%M")

        history = model.fit(
            X_train_s, y_train,
            validation_data=(X_val_s, y_val),
            epochs=400,
            batch_size=512,
            shuffle=True,
            verbose=1,
            callbacks=callbacks
        )

        print("\n=== Step 6: 模型评估 ===")
        y_pred = model.predict(X_test_s)
        y_pred = transform.inverse(y_pred[:, 0]).numpy()

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n✅ Test MSE: {mse:.6f}, R²: {r2:.4f}")

        # 保存模型与Scaler
        os.makedirs('scalers', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(scaler, f'scalers/X_scaler_{data_type}_{timestamp}.pkl')
        model.save(f'models/chooser_option_mlp_model_{data_type}_{timestamp}.h5')
        print("📁 模型与Scaler已保存")

        # 绘制Loss曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.yscale('log')
        plt.title(f"Data: {data_type}, Enhanced: {enhanced}\nR²={r2:.4f}, true_mean={y_test.mean():.4f}, pred_mean={y_pred.mean():.4f}")
        plt.legend()
        plt.show()

        return model, scaler, history

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ==========================================================
# 7️⃣ 评估函数
# ==========================================================
def evaluate_model(model_path, data_type='mixed', n_paths=500000):
    """评估模型性能的主函数"""
    print(f"🎯 开始评估模式 - 模型: {model_path}")
    
    # ==== 基础参数设置 / Base Parameters ====
    base_params = {
        's0': 156.7,           # 初始股价 / Initial stock price
        'K': 150,              # 执行价 / Strike price
        'r': 0.0015,           # 无风险利率 / Risk-free rate
        'q': 0.0233,           # 股息率 / Dividend yield
        'sigma': 0.282,        # 波动率 / Volatility
        'T': 1.0,              # 到期时间 / Time to maturity
        'choice_date': 0.5,    # 选择日期 / Choice date (e.g., half-year)
        'n_steps': 252,        # 时间步数 / Number of time steps
        'n_paths': n_paths,    # 模拟路径数 / Number of Monte Carlo paths
        'option_type': 'asian' # 期权类型 / Option type ('asian' or 'lookback')
    }

    # ==== 动态更新参数 / Update Parameters Dynamically ====
    try:
        params = calculate_jpm_metrics('2021-8-23', '2022-8-23')
        params['K'] = round(params['s0'], -1)
        base_params.update(params)
        print("✅ JPM参数更新成功")
    except Exception as e:
        print(f"⚠️ JPM参数更新失败，使用默认参数: {e}")

    # ==== Step 1: 生成 Monte Carlo 数据 / Generate Monte Carlo Data ====
    print("\n=== Step 1: 生成 Monte Carlo 数据 ===")
    t1 = time()
    paths, payoffs, params_array = generate_training_data(**base_params)
    t2 = time()

    # ==== 加载模型与Scaler / Load Model & Scaler ====
    print("\n=== Step 2: 加载模型与Scaler ===")
    try:
        print("开始加载模型...")
        model = load_model(model_path, compile=False)
        print("✅ 模型加载成功")

        # 从模型路径推断scaler路径 - 修改这部分
        scaler_dir = 'scalers'
        model_filename = os.path.basename(model_path)
        
        # 尝试多种可能的scaler命名规则
        possible_scaler_paths = [
            # 规则1: 直接替换模型名为scaler名
            model_path.replace('models/', 'scalers/').replace('_model_', '_scaler_').replace('.h5', '.pkl'),
            # 规则2: 使用固定的scaler文件名（根据你的实际文件名）
            os.path.join(scaler_dir, 'X_scaler_mixed.pkl'),
            # 规则3: 从模型名提取数据类型
            os.path.join(scaler_dir, f"X_scaler_{data_type}.pkl"),
            # 规则4: 简单的文件名替换
            model_path.replace('.h5', '.pkl').replace('models', 'scalers')
        ]
        
        X_scaler = None
        used_path = None
        
        for scaler_path in possible_scaler_paths:
            if os.path.exists(scaler_path):
                X_scaler = joblib.load(scaler_path)
                used_path = scaler_path
                print(f"✅ Scaler加载成功: {used_path}")
                break
        
        if X_scaler is None:
            # 如果所有规则都失败，列出可用的scaler文件
            available_scalers = [f for f in os.listdir(scaler_dir) if f.endswith('.pkl')]
            print(f"❌ 无法自动找到对应的scaler文件")
            print(f"📁 可用的scaler文件: {available_scalers}")
            raise FileNotFoundError("请手动指定正确的scaler文件路径")
            
    except Exception as e:
        print(f"❌ 加载过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("✅ 模型与Scaler加载完成")

    # ==== 特征构建 / Feature Construction ====
    print("\n=== Step 3: 特征工程 ===")
    base_params_stack = np.tile(
        np.array([
            base_params['s0'],
            base_params['K'],
            base_params['r'],
            base_params['q'],
            base_params['sigma'],
            base_params['T'],
            base_params['choice_date'] / base_params['T']
        ], dtype=np.float32),
        (base_params['n_paths'], 1)
    )

    X_enhanced = create_robust_features(paths, base_params['s0'], base_params['K'], base_params_stack)

    # ==== 特征缩放 / Feature Scaling ====
    X_test_scaled = X_scaler.transform(X_enhanced)

    # ==== 模型预测 / Model Prediction ====
    print("\n=== Step 4: 模型预测 ===")
    y_pred_transformed = model.predict(X_test_scaled)

    # 兼容多输出模型 / Handle different output formats
    if y_pred_transformed.ndim == 2 and y_pred_transformed.shape[1] == 2:
        y_pred = y_pred_transformed[:, 0]
    else:
        y_pred = y_pred_transformed.reshape(-1)

    # ==== 逆变换预测结果 / Inverse Transform ====
    transform = SignedLogTransform()
    y_pred = transform.inverse(y_pred)
    y_pred = y_pred.numpy()

    t3 = time()

    print(f"⏱️ 蒙特卡洛耗时: {t2 - t1:.4f}s")
    print(f"⏱️ 模型预测耗时: {t3 - t2:.4f}s")

    # ==== 性能指标计算 / Compute Metrics ====
    print("\n=== Step 5: 性能评估 ===")
    y_test = payoffs
    print(f"📊 y_test mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")
    print(f"📊 y_pred mean: {y_pred.mean():.4f}, std: {y_pred.std():.4f}")

    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    print(f"✅ MAE: {mae:.4f}")
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ R²: {r2:.4f}")

    # ==== 极端值误差分析 / Extreme Value Error Analysis ====
    def analyze_extremes(y_true, y_pred, percentile=90):
        """计算极端样本与正常样本的误差 / Compare errors for extreme vs normal samples."""
        threshold = np.percentile(y_true, percentile)
        mask_extreme = y_true > threshold
        extreme_mae = np.mean(np.abs(y_true[mask_extreme] - y_pred[mask_extreme]))
        normal_mae = np.mean(np.abs(y_true[~mask_extreme] - y_pred[~mask_extreme]))
        return extreme_mae, normal_mae

    extreme_mae, normal_mae = analyze_extremes(y_test, y_pred)
    print(f"📈 极端值MAE: {extreme_mae:.4f}")
    print(f"📉 正常值MAE: {normal_mae:.4f}")

    # ==== 可视化分析 / Visualization ====
    print("\n=== Step 6: 生成可视化图表 ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 左图：预测值 vs 蒙特卡洛真值 / Left: Predicted vs True ---
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.25, s=8)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, 'r--', lw=1)
    ax1.set_xlabel("Monte Carlo Payoff (True)")
    ax1.set_ylabel("ML Predicted Payoff")
    ax1.set_title("Chooser Option: Prediction vs Monte Carlo")
    ax1.axis('equal')
    ax1.grid(alpha=0.3)

    # --- 中图：误差分布 / Middle: Error Distribution ---
    ax2 = axes[1]
    errors = y_pred - y_test
    ax2.hist(errors, bins=50, alpha=0.7, color='steelblue')
    ax2.set_xlabel("Prediction Error (Pred - True)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    ax2.grid(alpha=0.3)

    # --- 右图：预测分布 vs 真值分布 / Right: Value Distribution ---
    ax3 = axes[2]
    ax3.hist(y_test, bins=50, alpha=0.7, label='True Values', color='blue')
    ax3.hist(y_pred, bins=50, alpha=0.7, label='Predicted Values', color='orange')
    ax3.legend()
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution Comparison')
    ax3.grid(alpha=0.3)

    # --- 图标题与布局 / Overall Title & Layout ---
    fig.suptitle(
        f'Chooser Option MLP Evaluation (MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f})',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()    
    plt.show()

    return {
        'mae': mae,
        'rmse': rmse, 
        'r2': r2,
        'extreme_mae': extreme_mae,
        'normal_mae': normal_mae,
        'mc_time': t2 - t1,
        'inference_time': t3 - t2
    }

# ==========================================================
# 8️⃣ 预测函数
# ==========================================================
def predict_option_price(model_path, s0, K, r, q, sigma, T, choice_date, n_paths=100000):
    """预测选择期权价格的主函数"""
    print(f"🔮 开始预测模式")
    print(f"📊 输入参数:")
    print(f"  s0 (初始股价): {s0}")
    print(f"  K (执行价): {K}")
    print(f"  r (无风险利率): {r}")
    print(f"  q (股息率): {q}")
    print(f"  sigma (波动率): {sigma}")
    print(f"  T (到期时间): {T}")
    print(f"  choice_date (选择日期): {choice_date}")
    print(f"  n_paths (模拟路径数): {n_paths}")

    # ==== 参数验证 ====
    if choice_date >= T:
        raise ValueError(f"选择日期 ({choice_date}) 必须小于到期时间 ({T})")
    
    if s0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("股价、执行价和到期时间必须为正数")

    # ==== 加载模型与Scaler ====
    print("\n=== Step 1: 加载模型与Scaler ===")
    try:
        model = load_model(model_path, compile=False)
        print("✅ 模型加载成功")

        # 从模型路径推断scaler路径 - 修改这部分
        scaler_dir = 'scalers'
        model_filename = os.path.basename(model_path)
        
        # 尝试多种可能的scaler命名规则
        possible_scaler_paths = [
            # 使用你提供的固定文件名
            os.path.join(scaler_dir, 'X_scaler_mixed.pkl'),
            # 其他可能的命名规则
            model_path.replace('models/', 'scalers/').replace('_model_', '_scaler_').replace('.h5', '.pkl'),
            model_path.replace('.h5', '.pkl').replace('models', 'scalers')
        ]
        
        X_scaler = None
        used_path = None
        
        for scaler_path in possible_scaler_paths:
            if os.path.exists(scaler_path):
                X_scaler = joblib.load(scaler_path)
                used_path = scaler_path
                print(f"✅ Scaler加载成功: {used_path}")
                break
        
        if X_scaler is None:
            # 如果所有规则都失败，列出可用的scaler文件
            available_scalers = [f for f in os.listdir(scaler_dir) if f.endswith('.pkl')]
            print(f"❌ 无法自动找到对应的scaler文件")
            print(f"📁 可用的scaler文件: {available_scalers}")
            raise FileNotFoundError("请手动指定正确的scaler文件路径")

    except Exception as e:
        print(f"❌ 加载过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ==== 生成测试数据 ====
    print("\n=== Step 2: 生成测试数据 ===")
    try:
        t1 = time()
        paths, payoffs, params_array = generate_training_data(
            s0=s0, K=K, r=r, q=q, sigma=sigma, T=T,
            choice_date=choice_date, n_steps=252, n_paths=n_paths,
            option_type='asian'
        )
        t2 = time()
        print(f"✅ 成功生成 {n_paths} 条路径，耗时: {t2-t1:.2f}s")
    except Exception as e:
        print(f"❌ 数据生成失败: {e}")
        return None

    # ==== 特征工程 ====
    print("\n=== Step 3: 特征工程 ===")
    base_params_stack = np.tile(
        np.array([s0, K, r, q, sigma, T, choice_date/T], dtype=np.float32),
        (n_paths, 1)
    )

    X_enhanced = create_robust_features(paths, s0, K, base_params_stack)

    # ==== 特征缩放 ====
    X_scaled = X_scaler.transform(X_enhanced)

    # ==== 模型预测 ====
    print("\n=== Step 4: 模型预测 ===")
    try:
        y_pred_transformed = model.predict(X_scaled, batch_size=1024, verbose=1)
        
        # 处理模型输出
        if y_pred_transformed.ndim == 2 and y_pred_transformed.shape[1] == 2:
            y_pred = y_pred_transformed[:, 0]
        else:
            y_pred = y_pred_transformed.reshape(-1)

        # 逆变换
        transform = SignedLogTransform()
        y_pred_payoffs = transform.inverse(y_pred).numpy()
        
        t3 = time()
        print(f"✅ 预测完成，耗时: {t3-t2:.2f}s")

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return None

    # ==== 结果计算 ====
    print("\n=== Step 5: 结果计算 ===")
    
    # 计算蒙特卡洛基准价格
    mc_price = np.mean(payoffs)
    mc_std = np.std(payoffs) / np.sqrt(n_paths)
    
    # 计算模型预测价格
    ml_price = np.mean(y_pred_payoffs)
    ml_std = np.std(y_pred_payoffs) / np.sqrt(n_paths)
    
    # 计算预测区间
    confidence = 0.95
    z_score = 1.96  # 95% 置信区间
    
    mc_ci_lower = mc_price - z_score * mc_std
    mc_ci_upper = mc_price + z_score * mc_std
    
    ml_ci_lower = ml_price - z_score * ml_std
    ml_ci_upper = ml_price + z_score * ml_std
    
    # 计算相对误差
    relative_error = abs(ml_price - mc_price) / mc_price * 100

    # ==== 输出结果 ====
    print("\n" + "="*60)
    print("🎯 选择期权定价结果")
    print("="*60)
    print(f"📊 蒙特卡洛基准价格: {mc_price:.6f}")
    print(f"  95% 置信区间: [{mc_ci_lower:.6f}, {mc_ci_upper:.6f}]")
    print(f"  标准误差: {mc_std:.6f}")
    print()
    print(f"🤖 机器学习预测价格: {ml_price:.6f}")
    print(f"  95% 置信区间: [{ml_ci_lower:.6f}, {ml_ci_upper:.6f}]")
    print(f"  标准误差: {ml_std:.6f}")
    print()
    print(f"📈 相对误差: {relative_error:.4f}%")
    print(f"⏱️ 总耗时: {t3-t1:.2f}s")
    print("="*60)

    # ==== 可视化结果 ====
    print("\n=== Step 6: 生成可视化图表 ===")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 左上：价格分布对比
    ax1 = axes[0, 0]
    ax1.hist(payoffs, bins=50, alpha=0.7, label='Monte Carlo', color='blue', density=True)
    ax1.hist(y_pred_payoffs, bins=50, alpha=0.7, label='ML Prediction', color='orange', density=True)
    ax1.axvline(mc_price, color='blue', linestyle='--', linewidth=2, label=f'MC Mean: {mc_price:.4f}')
    ax1.axvline(ml_price, color='orange', linestyle='--', linewidth=2, label=f'ML Mean: {ml_price:.4f}')
    ax1.set_xlabel('Option Payoff')
    ax1.set_ylabel('Density')
    ax1.set_title('Payoff Distribution Comparison')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 右上：预测vs真实散点图
    ax2 = axes[0, 1]
    ax2.scatter(payoffs, y_pred_payoffs, alpha=0.5, s=10)
    lims = [min(payoffs.min(), y_pred_payoffs.min()), max(payoffs.max(), y_pred_payoffs.max())]
    ax2.plot(lims, lims, 'r--', alpha=0.8, label='Perfect Prediction')
    ax2.set_xlabel('Monte Carlo Payoff')
    ax2.set_ylabel('ML Predicted Payoff')
    ax2.set_title('Prediction vs True Payoff')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 左下：价格对比条形图
    ax3 = axes[1, 0]
    methods = ['Monte Carlo', 'ML Prediction']
    prices = [mc_price, ml_price]
    errors = [mc_std * z_score, ml_std * z_score]
    bars = ax3.bar(methods, prices, yerr=errors, capsize=10, alpha=0.7, 
                   color=['blue', 'orange'], edgecolor='black')
    ax3.set_ylabel('Option Price')
    ax3.set_title('Price Comparison with 95% Confidence Intervals')
    ax3.grid(alpha=0.3, axis='y')
    
    # 在条形上添加数值标签
    for bar, price in zip(bars, prices):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + errors[0],
                f'{price:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 右下：参数摘要
    ax4 = axes[1, 1]
    ax4.axis('off')
    param_text = (
        f"Parameters Summary:\n\n"
        f"S₀ = {s0:.2f}\n"
        f"K = {K:.2f}\n"
        f"r = {r:.4f}\n"
        f"q = {q:.4f}\n"
        f"σ = {sigma:.4f}\n"
        f"T = {T:.2f}\n"
        f"choice_date = {choice_date:.2f}\n"
        f"n_paths = {n_paths:,}\n\n"
        f"Results:\n"
        f"MC Price = {mc_price:.6f}\n"
        f"ML Price = {ml_price:.6f}\n"
        f"Error = {relative_error:.4f}%"
    )
    ax4.text(0.1, 0.9, param_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.suptitle(f'Chooser Option Pricing Prediction\n(Relative Error: {relative_error:.4f}%)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return {
        'monte_carlo_price': mc_price,
        'ml_predicted_price': ml_price,
        'relative_error_percent': relative_error,
        'monte_carlo_std': mc_std,
        'ml_std': ml_std,
        'total_time': t3 - t1
    }

# ==========================================================
# 9️⃣ 命令行接口主函数
# ==========================================================

def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description='训练、评估和预测选择期权定价模型')
    
    parser.add_argument('--mode', '-m', type=str, required=True,
                       choices=['train', 'eval', 'predict'],
                       help='运行模式: train (训练), eval (评估), predict (预测)')
    
    parser.add_argument('--data_type', '-d', type=str, 
                       choices=['synthetic', 'historical', 'mixed'],
                       help='数据类型: synthetic (合成数据), historical (历史数据), mixed (混合数据)')
    
    parser.add_argument('--enhanced', '-e', action='store_true',
                       help='是否使用增强历史数据（添加扰动）')
    
    parser.add_argument('--n_total_paths', '-n', type=int, default=400000,
                       help='总路径数 (默认: 400000)')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径 (用于评估或预测模式)')
    
    parser.add_argument('--eval_paths', type=int, default=500000,
                       help='评估时使用的路径数 (默认: 500000)')
    
    parser.add_argument('--scaler_path', type=str, default=None,
                       help='Scaler文件路径 (用于评估或预测模式，如不指定则自动推断)')
    
    # 预测模式专用参数
    parser.add_argument('--s0', type=float, help='初始股价')
    parser.add_argument('--K', type=float, help='执行价')
    parser.add_argument('--r', type=float, help='无风险利率')
    parser.add_argument('--q', type=float, help='股息率')
    parser.add_argument('--sigma', type=float, help='波动率')
    parser.add_argument('--T', type=float, help='到期时间')
    parser.add_argument('--choice_date', type=float, help='选择日期')
    parser.add_argument('--n_paths_predict', type=int, default=100000,
                       help='预测时使用的路径数 (默认: 100000)')
    
    args = parser.parse_args()
    
    print(f"🚀 开始运行选择期权定价模型")
    print(f"📊 模式: {args.mode}")
    
    if args.mode == 'train':
        if args.data_type is None:
            print("❌ 训练模式必须指定 --data_type 参数")
            sys.exit(1)
            
        # 映射数据类型的参数
        data_mode_mapping = {
            'synthetic': 'synthetic_only',
            'historical': 'historical_only', 
            'mixed': 'mixed'
        }
        
        data_mode = data_mode_mapping.get(args.data_type, 'mixed')
        
        print(f"📈 数据类型: {args.data_type} -> {data_mode}")
        print(f"🔧 增强模式: {args.enhanced}")
        print(f"📊 总路径数: {args.n_total_paths}")
        
        print("\n🎯 开始训练模式...")
        model, scaler, history = train_model(
            data_type=data_mode,
            enhanced=args.enhanced,
            n_total_paths=args.n_total_paths
        )
        
        if model is not None:
            print("✅ 训练完成!")
        else:
            print("❌ 训练失败!")
            sys.exit(1)
            
    elif args.mode == 'eval':
        if args.model_path is None and args.data_type is None:
            print("❌ 评估模式必须指定 --model_path 或 --data_type 参数")
            sys.exit(1)
            
        if args.model_path is None:
            # 如果没有指定模型路径，使用默认命名规则
            args.model_path = f'models/chooser_option_mlp_model_{args.data_type}.h5'
            print(f"🔍 使用默认模型路径: {args.model_path}")
        
                # 如果手动指定了scaler路径，优先使用
        if args.scaler_path:
            scaler_path = args.scaler_path
        else:
            scaler_path = None  # 让函数自动推断

        print("\n📊 开始评估模式...")
        results = evaluate_model(
            model_path=args.model_path,
            data_type=args.data_type if args.data_type else 'mixed',
            n_paths=args.eval_paths
        )
        
        if results is not None:
            print("✅ 评估完成!")
            print(f"📋 评估结果:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print("❌ 评估失败!")
            sys.exit(1)
        
    elif args.mode == 'predict':
        # 检查必需的预测参数
        required_params = ['s0', 'K', 'r', 'q', 'sigma', 'T', 'choice_date']
        missing_params = [param for param in required_params if getattr(args, param) is None]
        
        if missing_params:
            print(f"❌ 预测模式缺少必需参数: {', '.join(missing_params)}")
            print("ℹ️  预测模式需要以下参数:")
            print("  --s0: 初始股价")
            print("  --K: 执行价") 
            print("  --r: 无风险利率")
            print("  --q: 股息率")
            print("  --sigma: 波动率")
            print("  --T: 到期时间")
            print("  --choice_date: 选择日期")
            sys.exit(1)
            
        if args.model_path is None:
            print("❌ 预测模式必须指定 --model_path 参数")
            sys.exit(1)

        # 如果手动指定了scaler路径，优先使用
        if args.scaler_path:
            scaler_path = args.scaler_path
        else:
            scaler_path = None  # 让函数自动推断
        
        results = predict_option_price(
            model_path=args.model_path,
            s0=args.s0,
            K=args.K,
            r=args.r,
            q=args.q,
            sigma=args.sigma,
            T=args.T,
            choice_date=args.choice_date,
            n_paths=args.n_paths_predict
        )
        
        if results is not None:
            print("✅ 预测完成!")
        else:
            print("❌ 预测失败!")
            sys.exit(1)
    
    print("🎉 程序执行完成!")

if __name__ == '__main__':
    main()
