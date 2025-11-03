import matplotlib.pyplot as plt
import numpy as np
import joblib
import sys
import os
from time import time
from tensorflow.keras.models import load_model

# ==== 路径与环境设置 / Path & Environment Setup ====
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定 GPU 设备 (Set CUDA device)

# ==== 自定义模块导入 / Import Custom Modules ====
from src.param_calculator import calculate_jpm_metrics
from src.BSM_ML import (
    generate_training_data,
    create_robust_features,
    SignedLogTransform
)

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
    'n_paths': 500000,     # 模拟路径数 / Number of Monte Carlo paths
    'option_type': 'asian' # 期权类型 / Option type ('asian' or 'lookback')
}

# ==== 动态更新参数 / Update Parameters Dynamically ====
params = calculate_jpm_metrics('2021-8-23', '2022-8-23')
params['K'] = round(params['s0'], -1)
base_params.update(params)

# ==== Step 1: 生成 Monte Carlo 数据 / Generate Monte Carlo Data ====
mode = 'mixed'

t1 = time()
paths, payoffs, params_array = generate_training_data(**base_params)
t2 = time()

# ==== 加载模型与Scaler / Load Model & Scaler ====
model_path = f'models/chooser_option_mlp_model_{mode}.h5'

try:
    print("开始加载模型 / Loading model...")
    model = load_model(model_path, compile=False)
    print("模型加载成功 / Model loaded successfully")

    print("开始加载Scaler / Loading scaler...")
    X_scaler = joblib.load(f'scalers/X_scaler_{mode}.pkl')
    print("Scaler加载成功 / Scaler loaded successfully")

except Exception as e:
    print(f"加载过程中出错 / Error during loading: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("模型与Scaler加载完成 / All models and scalers loaded")

# ==== 特征构建 / Feature Construction ====
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

print(f"蒙特卡洛耗时 / Monte Carlo time: {t2 - t1:.4f}s")
print(f"模型预测耗时 / Model inference time: {t3 - t2:.4f}s")

# ==== 性能指标计算 / Compute Metrics ====
y_test = payoffs
print(f"y_test mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")
print(f"y_pred mean: {y_pred.mean():.4f}, std: {y_pred.std():.4f}")

mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# ==== 极端值误差分析 / Extreme Value Error Analysis ====
def analyze_extremes(y_true, y_pred, percentile=90):
    """计算极端样本与正常样本的误差 / Compare errors for extreme vs normal samples."""
    threshold = np.percentile(y_true, percentile)
    mask_extreme = y_true > threshold
    extreme_mae = np.mean(np.abs(y_true[mask_extreme] - y_pred[mask_extreme]))
    normal_mae = np.mean(np.abs(y_true[~mask_extreme] - y_pred[~mask_extreme]))
    return extreme_mae, normal_mae

extreme_mae, normal_mae = analyze_extremes(y_test, y_pred)
print(f"极端值MAE / Extreme MAE: {extreme_mae:.4f}")
print(f"正常值MAE / Normal MAE: {normal_mae:.4f}")

# ==== 可视化分析 / Visualization ====
print("生成可视化图表 / Generating comprehensive comparison plots...")
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
