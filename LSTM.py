import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import pandas as pd
import os
os.chdir('d:/My/清华大学/学习/4.2大四下/2毕业设计/MAPCSS')

import time

start_time = time.time()
df = pd.read_csv("./train_FD001_with_RUL.csv").iloc[:100]
test = pd.read_csv("./test_FD001_with_RUL.csv")

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def build_engine_id_map(df):
    engine_ids = sorted(df['engine_id'].unique())
    return {eid: idx for idx, eid in enumerate(engine_ids)}

def apply_engine_id_encoding(df, engine_id_map):
    df['engine_id_encoded'] = df['engine_id'].map(engine_id_map)
    df = df.dropna(subset=['engine_id_encoded']).copy()
    df['engine_id_encoded'] = df['engine_id_encoded'].astype(int)
    return df

def fit_scalers(df, feature_cols, label_col='RUL'):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaler.fit(df[feature_cols])
    y_scaler.fit(df[[label_col]])
    return X_scaler, y_scaler

def normalize_df(df, X_scaler, y_scaler, feature_cols, label_col='RUL'):
    df[feature_cols] = X_scaler.transform(df[feature_cols])
    df[label_col] = y_scaler.transform(df[[label_col]])
    return df

def generate_sequences(df, seq_len, feature_cols):
    X, y, engine_ids = [], [], []
    for eid, group in df.groupby('engine_id'):
        group = group.sort_values('cycle').reset_index(drop=True)
        if len(group) < seq_len or 'engine_id_encoded' not in group.columns:
            continue
        eid_encoded = group['engine_id_encoded'].iloc[0]
        for i in range(len(group) - seq_len + 1):
            seq = group.loc[i:i+seq_len-1, feature_cols].values
            label = group.loc[i+seq_len-1, 'RUL']
            X.append(seq)
            y.append(label)
            engine_ids.append(eid_encoded)
    return np.array(X), np.array(y), np.array(engine_ids)

def train_model(model, train_loader, epochs=20, lr=1e-3):
    model.train()
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, engine_ids, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb, engine_ids)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_list.append(total_loss)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f}")
    return loss_list

def evaluate_model(model, test_loader, y_scaler):
    model.eval()
    all_preds, all_truths = [], []

    with torch.no_grad():
        for xb, engine_ids, yb in test_loader:
            preds = model(xb, engine_ids)
            all_preds.append(preds.numpy())
            all_truths.append(yb.numpy())

    all_preds = np.concatenate(all_preds).reshape(-1, 1)
    all_truths = np.concatenate(all_truths).reshape(-1, 1)

    # 🔁 反归一化
    all_preds_inv = y_scaler.inverse_transform(all_preds).flatten()
    all_truths_inv = y_scaler.inverse_transform(all_truths).flatten()

    mae = mean_absolute_error(all_truths_inv, all_preds_inv)
    mse = mean_squared_error(all_truths_inv, all_preds_inv)
    mean_pred = np.mean(all_preds_inv)
    var_pred = np.var(all_preds_inv)

    print(f"\n📊 Evaluation on Test Set:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Mean prediction: {mean_pred:.4f}")
    print(f"Variance of prediction: {var_pred:.4f}")

    return all_preds_inv, all_truths_inv

def plot_predictions(y_true, y_pred, title="True vs Predicted RUL"):
    plt.figure(figsize=(8, 6))
    plt.plot(y_true, label="True RUL", alpha=0.8)
    plt.plot(y_pred, label="Predicted RUL", alpha=0.8)
    plt.xlabel("Sample Index")
    plt.ylabel("RUL")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ✅ 生成 engine_id 编码映射（仅用训练集）
def build_engine_id_map(df):
    engine_ids = sorted(df['engine_id'].unique())
    return {eid: idx for idx, eid in enumerate(engine_ids)}

# ✅ 应用映射到 df 中
def apply_engine_id_encoding(df, engine_id_map):
    df['engine_id_encoded'] = df['engine_id'].map(engine_id_map)
    df = df.dropna(subset=['engine_id_encoded']).copy()
    df['engine_id_encoded'] = df['engine_id_encoded'].astype(int)
    return df

# ✅ 序列生成函数（训练 & 测试通用）
def generate_sequences(df, seq_len, feature_cols):
    X, y, engine_ids = [], [], []

    for eid, group in df.groupby('engine_id'):
        group = group.sort_values('cycle').reset_index(drop=True)
        if len(group) < seq_len or 'engine_id_encoded' not in group.columns:
            continue
        eid_encoded = group['engine_id_encoded'].iloc[0]
        for i in range(len(group) - seq_len + 1):
            seq = group.loc[i:i+seq_len-1, feature_cols].values
            label = group.loc[i+seq_len-1, 'RUL']
            X.append(seq)
            y.append(label)
            engine_ids.append(eid_encoded)
    return np.array(X), np.array(y), np.array(engine_ids)

# ✅ 模型训练函数
def train_model(model, train_loader, epochs=20, lr=1e-3):
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, engine_ids, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb, engine_ids)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_list.append(total_loss)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f}")
    return loss_list

# ✅ 模型评估函数
def evaluate_model(model, test_loader, y_scaler):  # ✅ 加上 y_scaler
    model.eval()
    all_preds, all_truths = [], []

    with torch.no_grad():
        for xb, engine_ids, yb in test_loader:
            preds = model(xb, engine_ids)
            all_preds.append(preds.numpy())
            all_truths.append(yb.numpy())

    all_preds = np.concatenate(all_preds).reshape(-1, 1)
    all_truths = np.concatenate(all_truths).reshape(-1, 1)

    # 🔁 反归一化
    all_preds_inv = y_scaler.inverse_transform(all_preds).flatten()
    all_truths_inv = y_scaler.inverse_transform(all_truths).flatten()

    mae = mean_absolute_error(all_truths_inv, all_preds_inv)
    mse = mean_squared_error(all_truths_inv, all_preds_inv)
    mean_pred = np.mean(all_preds_inv)
    var_pred = np.var(all_preds_inv)

    print(f"\n📊 Evaluation on Test Set:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Mean prediction: {mean_pred:.4f}")
    print(f"Variance of prediction: {var_pred:.4f}")

    return all_preds_inv, all_truths_inv

class LSTM_RUL_with_EngineEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_engines, emb_dim=4):
        super().__init__()
        self.engine_embed = nn.Embedding(num_engines, emb_dim)
        self.lstm = nn.LSTM(input_size + emb_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, engine_ids):
        batch_size, seq_len, _ = x.size()
        emb = self.engine_embed(engine_ids).unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, emb], dim=-1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# ✅ 可视化函数
def plot_predictions(y_true, y_pred, title="True vs Predicted RUL"):
    plt.figure(figsize=(8, 6))
    plt.plot(y_true, label="True RUL", alpha=0.8)
    plt.plot(y_pred, label="Predicted RUL", alpha=0.8)
    plt.xlabel("Sample Index")
    plt.ylabel("RUL")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# 设置参数
feature_cols = [f's{i}' for i in range(1, 22)]
seq_len = 15

# 生成编码
engine_id_map = build_engine_id_map(df)
df = apply_engine_id_encoding(df, engine_id_map)
test = apply_engine_id_encoding(test, engine_id_map)

# 🎯 归一化器拟合 & 应用
X_scaler, y_scaler = fit_scalers(df, feature_cols)
df = normalize_df(df, X_scaler, y_scaler, feature_cols)
test = normalize_df(test, X_scaler, y_scaler, feature_cols)

# 序列数据生成
X_train, y_train, engine_id_train = generate_sequences(df, seq_len, feature_cols)
X_test, y_test, engine_id_test = generate_sequences(test, seq_len, feature_cols)

# 转换为 TensorDataset
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                  torch.tensor(engine_id_train, dtype=torch.int64),
                  torch.tensor(y_train, dtype=torch.float32)),
    batch_size=64, shuffle=True
)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                  torch.tensor(engine_id_test, dtype=torch.int64),
                  torch.tensor(y_test, dtype=torch.float32)),
    batch_size=64, shuffle=False
)

# 初始化模型
model = LSTM_RUL_with_EngineEmbedding(
    input_size=len(feature_cols),
    hidden_size=128,         # ↑ 更深隐藏层
    num_layers=3,            # ↑ 增加时间捕捉能力
    num_engines=len(engine_id_map),
    emb_dim=8                # ↑ 提高 engine 表达能力
)

# 训练 + 评估 + 可视化
# train_model(model, train_loader, epochs=20, lr=1e-3)
train_model(model, train_loader, epochs=50, lr=5e-4)
preds, truths = evaluate_model(model, test_loader, y_scaler)
plot_predictions(truths, preds)

end_time = time.time()
print("运行时间为：{:.5f} 秒".format(end_time - start_time))
# import seaborn as sns
# sns.histplot(y_train, kde=True)
# plt.title("Distribution of Normalized RUL (y_train)")
# plt.show()

# sns.histplot(preds - truths, kde=True)
# plt.title("Prediction Error (Pred - True)")
# plt.xlabel("Error")
# plt.show()

# print("预测值范围:", preds.min(), preds.max())
# print("真实值范围:", truths.min(), truths.max())

# plt.plot(X_train[0,:,0], label='s1')
# plt.plot(X_train[0,:,1], label='s2')
# plt.legend()
# plt.title("某一训练样本的传感器变化")
# plt.show()

# sns.histplot(y_train, kde=True)
# plt.title("训练集 RUL 分布")
# plt.show()

# overlap = set(df['engine_id']) & set(test['engine_id'])
# print("重复 engine_id 数量:", len(overlap))  # 应该是 0

# ✅ 清爽强健版：LSTM + 无重复 engine + 只归一化特征 + ReLU 防负预测

# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ==== 1. 读取数据 ====
# df = pd.read_csv("train_FD004_with_RUL.csv")

# # ==== 2. 分割训练 / 测试 ====
# unique_engines = sorted(df['engine_id'].unique())
# train_ids = unique_engines[:80]   # 前 80 个 engine
# test_ids = unique_engines[80:]    # 后面用于测试

# df_train = df[df['engine_id'].isin(train_ids)].copy()
# df_test = df[df['engine_id'].isin(test_ids)].copy()

# # ==== 3. 编码 engine_id ====
# engine_id_map = {eid: idx for idx, eid in enumerate(train_ids)}
# df_train['engine_id_encoded'] = df_train['engine_id'].map(engine_id_map)
# df_test['engine_id_encoded'] = df_test['engine_id'].map(engine_id_map)
# df_test = df_test.dropna(subset=['engine_id_encoded']).copy()
# df_test['engine_id_encoded'] = df_test['engine_id_encoded'].astype(int)

# # ==== 4. 特征归一化（不归一化 RUL） ====
# feature_cols = [f's{i}' for i in range(1, 22)]
# X_scaler = StandardScaler()
# df_train[feature_cols] = X_scaler.fit_transform(df_train[feature_cols])
# df_test[feature_cols] = X_scaler.transform(df_test[feature_cols])

# # ==== 5. 构建时序样本 ====
# def generate_sequences(df, seq_len, feature_cols):
#     X, y, engine_ids = [], [], []
#     for eid, group in df.groupby('engine_id'):
#         group = group.sort_values('cycle').reset_index(drop=True)
#         if len(group) < seq_len:
#             continue
#         eid_encoded = group['engine_id_encoded'].iloc[0]
#         for i in range(len(group) - seq_len + 1):
#             seq = group.loc[i:i+seq_len-1, feature_cols].values
#             label = group.loc[i+seq_len-1, 'RUL']
#             X.append(seq)
#             y.append(label)
#             engine_ids.append(eid_encoded)
#     return np.array(X), np.array(y), np.array(engine_ids)

# seq_len = 15
# X_train, y_train, eid_train = generate_sequences(df_train, seq_len, feature_cols)
# X_test, y_test, eid_test = generate_sequences(df_test, seq_len, feature_cols)

# # ==== 6. 构建 DataLoader ====
# train_loader = DataLoader(TensorDataset(
#     torch.tensor(X_train, dtype=torch.float32),
#     torch.tensor(eid_train, dtype=torch.long),
#     torch.tensor(y_train, dtype=torch.float32)),
#     batch_size=64, shuffle=True
# )

# test_loader = DataLoader(TensorDataset(
#     torch.tensor(X_test, dtype=torch.float32),
#     torch.tensor(eid_test, dtype=torch.long),
#     torch.tensor(y_test, dtype=torch.float32)),
#     batch_size=64, shuffle=False
# )

# # ==== 7. 模型定义（ReLU 防负 RUL） ====
# class LSTM_RUL_with_EngineEmbedding(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_engines, emb_dim=8):
#         super().__init__()
#         self.engine_embed = nn.Embedding(num_engines, emb_dim)
#         self.lstm = nn.LSTM(input_size + emb_dim, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 1),
#             nn.ReLU()
#         )

#     def forward(self, x, engine_ids):
#         emb = self.engine_embed(engine_ids).unsqueeze(1).expand(-1, x.size(1), -1)
#         x = torch.cat([x, emb], dim=-1)
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out.squeeze()

# # ==== 8. 训练函数 ====
# def train_model(model, loader, epochs=30, lr=1e-3):
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.L1Loss()
#     for epoch in range(epochs):
#         total_loss = 0
#         for xb, eids, yb in loader:
#             pred = model(xb, eids)
#             loss = loss_fn(pred, yb)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

# # ==== 9. 评估函数 ====
# def evaluate_model(model, loader):
#     model.eval()
#     preds, truths = [], []
#     with torch.no_grad():
#         for xb, eids, yb in loader:
#             pred = model(xb, eids)
#             preds.append(pred.numpy())
#             truths.append(yb.numpy())
#     preds = np.concatenate(preds)
#     truths = np.concatenate(truths)
#     print("\n📊 Evaluation:")
#     print("MAE:", mean_absolute_error(truths, preds))
#     print("MSE:", mean_squared_error(truths, preds))
#     print("预测值范围:", preds.min(), preds.max())
#     print("真实值范围:", truths.min(), truths.max())
#     return preds, truths

# # ==== 10. 画图函数 ====
# def plot_predictions(y_true, y_pred):
#     plt.figure(figsize=(8, 6))
#     plt.plot(y_true, label='True RUL', alpha=0.8)
#     plt.plot(y_pred, label='Predicted RUL', alpha=0.8)
#     plt.legend()
#     plt.title("True vs Predicted RUL")
#     plt.xlabel("Sample Index")
#     plt.ylabel("RUL")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # ==== 11. 运行全部流程 ====
# model = LSTM_RUL_with_EngineEmbedding(
#     input_size=len(feature_cols),
#     hidden_size=128,
#     num_layers=2,
#     num_engines=len(engine_id_map),
#     emb_dim=8
# )

# train_model(model, train_loader, epochs=40, lr=5e-4)
# preds, truths = evaluate_model(model, test_loader)
# plot_predictions(truths, preds)

