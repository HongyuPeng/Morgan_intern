import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
import sys
import os
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)
from src.BSM_ML import create_enhanced_features, generate_training_data
from tqdm import tqdm

# 加载模型和标准化器
model = load_model('models/chooser_option_mlp_model.keras')
X_scaler = joblib.load('scalers/X_scaler.save')
y_scaler = joblib.load('scalers/y_scaler.save')

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
    params = {
        's0': s0,
        'K': K,
        'r': np.random.uniform(*param_ranges['r']),
        'q': np.random.uniform(*param_ranges['q']),
        'sigma': np.random.uniform(*param_ranges['sigma']),
        'T': np.random.uniform(*param_ranges['T']),
        'n_steps': 252,
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

X_train, X_test, y_train, y_test = train_test_split(all_X_enhanced, all_payoffs, test_size=0.2, random_state=42)

print("生成新的训练数据...")

# 使用已有的标准化器（不要重新拟合！）
print("标准化新数据...")
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

# 设置较小的学习率进行微调
optimizer = Adam(learning_rate=1e-5)  # 比原始学习率小10倍
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 回调函数
callbacks = [
    EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-7)
]

print("开始增量训练...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    batch_size=256,
    epochs=1,  # 每个批次只训练1个epoch
    validation_data=(X_batch_val, y_batch_val),
    verbose=0
)

# 保存更新后的模型
model.save('models/chooser_option_mlp_model_finetuned.keras')
print("增量训练完成，模型已保存!")