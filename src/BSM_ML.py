import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

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
def simple_improved_data_generation(n_total_paths=400000):
    """批量生成带参数随机化的训练数据."""
    n_batches = 50
    paths_per_batch = n_total_paths // n_batches

    all_X, all_y, all_params = [], [], []

    for i in range(n_batches):
        s0 = float(np.random.uniform(90, 250))
        moneyness = np.random.uniform(0.9, 1.1)
        K = s0 * moneyness
        r = float(np.random.uniform(0.0001, 0.055))

        # 随机股息率分布
        rand = np.random.random()
        if rand < 0.1:
            q = 0.0
        elif rand < 0.7:
            q = float(np.random.uniform(0.0, 0.04))
        else:
            q = float(np.random.uniform(0.04, 0.1))

        sigma = float(np.random.uniform(0.1, 0.5))
        T = float(np.random.uniform(0.8, 1.2))
        choice_frac = float(np.random.uniform(0.4, 0.6))
        choice_date = T * choice_frac

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
            print(f"批次 {i} 失败: {e}")
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
# 6️⃣ 主训练流程
# ==========================================================
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ 使用 GPU 训练")
    else:
        print("⚙️ 使用 CPU 训练")

    try:
        print("\n=== Step 1: 数据生成 ===")
        X, y, params = simple_improved_data_generation(300000)

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
        os.makedirs(f'models/{timestamp}', exist_ok=True)

        history = model.fit(
            X_train_s, y_train,
            validation_data=(X_val_s, y_val),
            epochs=200,
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
        joblib.dump(scaler, f'scalers/X_scaler_{timestamp}.pkl')
        model.save(f'models/chooser_option_mlp_model_{timestamp}.h5')
        print("📁 模型与Scaler已保存")

        # 绘制Loss曲线
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.yscale('log')
        plt.title(f"R²={r2:.4f}, true_mean={y_test.mean():.4f}, pred_mean={y_pred.mean():.4f}")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
