import matplotlib.pyplot as plt
import numpy as np
import joblib
import sys
import os
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)
from src.param_calculator import calculate_jpm_metrics
from tests.BSM_ML_overfitting import generate_training_data, RobustScaler, create_robust_features
from tensorflow.keras.models import load_model
from time import time
# 先禁用 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 参数
base_params = {
    's0': 156.7,           # Initial stock price
    'K': 150,            # Strike price
    'r': 0.0015,           # Risk-free rate
    'q': 0.0233,           # Dividend yield
    'sigma': 0.282,        # Volatility
    'T': 1.0,            # Total time to expiration
    'choice_date': 0.5,  # Choice date (choose after half year)
    'n_steps': 252,      # Number of time steps
    'n_paths': 500000,    # Number of simulation paths
    'option_type': 'asian'  # Path-dependent type: 'asian' or 'lookback'
}

params = calculate_jpm_metrics('2023-8-23', '2024-8-23')
params['K'] = round(params['s0'], -1)
base_params.update(params)

# Step1: Monte Carlo 数据
# 重新生成 Monte Carlo 数据
time_stemp = '1012_1641'

time1 = time()
paths, payoffs, params_array = generate_training_data(**base_params)
time2 = time()
# 检查文件是否存在
model_path = f'models/chooser_option_mlp_model_{time_stemp}.h5'

try:
    print("开始加载模型...")
    model = load_model(model_path, compile=False)
    print('模型加载成功')
    
    print("开始加载scaler...")
    X_scaler = joblib.load(f'scalers/X_scaler_{time_stemp}.pkl')
    print('scaler加载成功')
    
except Exception as e:
    print(f"加载过程中出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("所有模型和scaler加载完成")

base_params_stack = np.tile(np.array([base_params['s0'],
                                base_params['K'],
                                base_params['r'],
                                base_params['q'],
                                base_params['sigma'],
                                base_params['T'],
                                base_params['choice_date']/base_params['T']],
                                dtype=np.float32), (base_params['n_paths'], 1))

X_enhanced = create_robust_features(paths, base_params['s0'], base_params['K'], base_params_stack)

# 检查特征维度
print(f"X_enhanced shape: {X_enhanced.shape}")
print(f"params_array shape: {params_array.shape}")



# 调用scaler
X_test_scaled = X_scaler.transform(X_enhanced)

# 预测
y_pred_t = model.predict(X_test_scaled).flatten()
y_pred = np.expm1(y_pred_t)
time3 = time()
print(f'蒙特卡洛模拟耗时{time3-time2:.4f}, 模型预测耗时{time2-time1:.4f}')

y_test = payoffs

# 检查数据分布
print(f"y_test mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")
print(f"y_pred mean: {y_pred.mean():.4f}, std: {y_pred.std():.4f}")

# 计算关键指标
mae = np.mean(np.abs(y_test - y_pred))
print(f"MAE: {mae:.4f}")
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print(f"RMSE: {rmse:.4f}") 
r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
print(f"R²: {r2:.4f}")

# 分析极端值预测
def analyze_extremes(y_test, y_pred, percentile=90):
    threshold = np.percentile(y_test, percentile)
    extreme_mask = y_test > threshold
    
    extreme_mae = np.mean(np.abs(y_test[extreme_mask] - y_pred[extreme_mask]))
    normal_mae = np.mean(np.abs(y_test[~extreme_mask] - y_pred[~extreme_mask]))
    
    return extreme_mae, normal_mae

extreme_mae, normal_mae = analyze_extremes(y_test, y_pred)
print(f"极端值MAE: {extreme_mae:.4f}")
print(f"正常值MAE: {normal_mae:.4f}")

# Create comprehensive comparison plots
print("Generating comprehensive comparison plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ===== Left Subplot: Prediction vs Monte Carlo =====
ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.25, s=8)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax1.plot(lims, lims, 'r--', lw=1)
ax1.set_xlabel("Monte Carlo Payoff (True)")
ax1.set_ylabel("ML Predicted Payoff")
ax1.set_title("Chooser Option: MLP Prediction vs Monte Carlo")
ax1.axis('equal')
ax1.grid(alpha=0.3)

# ===== Middle Subplot: Error Distribution =====
ax2 = axes[1]
errors = y_pred - y_test
ax2.hist(errors, bins=50, alpha=0.7, color='steelblue')
ax2.set_xlabel("Prediction Error (Pred - True)")
ax2.set_ylabel("Frequency")
ax2.set_title("Residual Distribution")
ax2.grid(alpha=0.3)

# ===== Right Subplot: Distribution Comparison =====
ax3 = axes[2]
ax3.hist(y_test, bins=50, alpha=0.7, label='True Values', color='blue')
ax3.hist(y_pred, bins=50, alpha=0.7, label='Predicted Values', color='orange')
ax3.legend()
ax3.set_xlabel('Value')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution Comparison - True vs Predicted')
ax3.grid(alpha=0.3)

# Add overall title with performance metrics
fig.suptitle(f'Chooser Option MLP Model Evaluation (MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f})', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()