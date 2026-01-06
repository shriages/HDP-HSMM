import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
os.chdir('d:/My/清华大学/学习/4.2大四下/2毕业设计/MAPCSS')
# -----------------------
# 1. 数据读取与预处理
# -----------------------
def load_data(filename):
    df = pd.read_csv(filename, sep=',', engine='python')
    df.set_index('Path', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(how='all', inplace=True)
    return df

def extract_crack_values(df, upper_bound=1.6):
    """提取 crack 数值，分为 non-failure（<=upper_bound）和 failure (>upper_bound)"""
    crack_values = df.values.flatten()
    crack_values = crack_values[~np.isnan(crack_values)]
    lower_bound = crack_values.min()
    failure_values = crack_values[crack_values > upper_bound]
    non_failure_values = crack_values[(crack_values >= lower_bound) & (crack_values <= upper_bound)]
    return lower_bound, failure_values, non_failure_values

# -----------------------
# 2. 根据数据计算各状态的发射概率参数
# -----------------------
def compute_emission_params(non_failure_values, failure_values, lower_bound, upper_bound, n_non_failure_states=6):
    """
    将 non_failure 数据分为 n_non_failure_states 个箱，每个箱对应一个状态，
    同时 failure 数据对应一个状态，共计 n_non_failure_states+1 个状态。
    返回各状态的均值和方差，以及分箱边界（仅供参考）。
    """
    bins = np.linspace(lower_bound, upper_bound, n_non_failure_states+1)
    mu = []
    sigma2 = []
    for i in range(n_non_failure_states):
        # 取区间 [bins[i], bins[i+1])
        bin_vals = non_failure_values[(non_failure_values >= bins[i]) & (non_failure_values < bins[i+1])]
        if len(bin_vals) == 0:
            m = (bins[i] + bins[i+1]) / 2
            s2 = 0.001
        else:
            m = bin_vals.mean()
            s2 = bin_vals.var() if bin_vals.var() > 0 else 0.001
        mu.append(m)
        sigma2.append(s2)
    # failure 状态（最后一个状态），用所有 failure 数据计算
    if len(failure_values) == 0:
        m_fail = upper_bound + 0.05  # 若无数据，随便给个值
        s2_fail = 0.001
    else:
        m_fail = failure_values.mean()
        s2_fail = failure_values.var() if failure_values.var() > 0 else 0.001
    mu.append(m_fail)
    sigma2.append(s2_fail)
    return np.array(mu), np.array(sigma2), bins

# -----------------------
# 3. 高斯概率密度与发射概率计算
# -----------------------
def gaussian_pdf(y, mu, sigma2):
    """计算 y 在高斯分布 N(mu, sigma2) 下的概率密度"""
    return norm.pdf(y, loc=mu, scale=np.sqrt(sigma2))

def emission_prob(y, mu, sigma2):
    """
    对于观测 y，计算每个状态 j 的发射概率
      b_j(y) = N(y; mu_j, sigma2_j)
    返回长度为状态数的数组
    """
    return np.array([gaussian_pdf(y, mu_j, sigma2_j) for mu_j, sigma2_j in zip(mu, sigma2)])

# -----------------------
# 4. Viterbi 算法实现
# -----------------------
def viterbi_algorithm(obs_seq, pi, A, mu, sigma2):
    """
    Viterbi 算法：
    初始化:
      delta_1(i) = pi_i * b_i(O_1)
      phi_1(i) = 0
    递归:
      delta_t(j) = max_{1<=i<=N}[delta_{t-1}(i) * a_{ij}] * b_j(O_t)
      phi_t(j) = argmax_{1<=i<=N}[delta_{t-1}(i) * a_{ij}]
    终止:
      P* = max_{1<=i<=N}[delta_T(i)]
      q*_T = argmax_{1<=i<=N}[delta_T(i)]
    回溯:
      q*_t = phi_{t+1}(q*_{t+1})
    返回最优状态序列以及完整的 delta、phi 矩阵（用于可视化）
    """
    T = len(obs_seq)
    N = len(pi)
    delta = np.zeros((T, N))
    phi = np.zeros((T, N), dtype=int)

    # 初始化
    b = emission_prob(obs_seq[0], mu, sigma2)
    delta[0, :] = pi * b
    phi[0, :] = 0

    # 递归
    for t in range(1, T):
        b = emission_prob(obs_seq[t], mu, sigma2)
        for j in range(N):
            products = delta[t-1, :] * A[:, j]
            delta[t, j] = np.max(products) * b[j]
            phi[t, j] = np.argmax(products)
    
    # 终止与回溯
    q_star = np.zeros(T, dtype=int)
    q_star[T-1] = np.argmax(delta[T-1, :])
    for t in range(T-2, -1, -1):
        q_star[t] = phi[t+1, q_star[t+1]]
    
    # 最优路径概率
    P_star = np.max(delta[T-1, :])
    return q_star, delta, phi, P_star

# -----------------------
# 5. 可视化函数
# -----------------------
def plot_delta_heatmap(delta):
    """绘制 Viterbi 算法中 delta 矩阵的热力图"""
    plt.figure(figsize=(8,4))
    plt.imshow(delta.T, aspect='auto', origin='lower', cmap='plasma')
    plt.xlabel("Time Step")
    plt.ylabel("State")
    plt.title("Viterbi δ (Delta) Matrix")
    cbar = plt.colorbar()
    cbar.set_label("δ value")
    plt.show()

def plot_viterbi_path(obs_seq, state_seq):
    """
    绘制观测序列与 Viterbi 最优状态序列：
      观测数据用折线图，状态序列用阶跃图显示
    """
    T = len(obs_seq)
    plt.figure(figsize=(10,4))
    plt.plot(range(T), obs_seq, marker='o', label="Observations", color='blue')
    plt.step(range(T), state_seq, where='mid', label="Viterbi State", color='red')
    plt.xlabel("Time Step")
    plt.ylabel("Value / State")
    plt.title("Observations & Viterbi Optimal Path")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------
# 6. 主函数：读取数据、构造 HMM 参数、运行 Viterbi 并可视化
# -----------------------
# def main():
#     # 读取数据
#     filename = 'crack_fatigue_data.txt'
#     df = load_data(filename)
    
#     # 提取所有 crack 数值，并分离出 failure 与 non-failure 数据
#     lower_bound, failure_values, non_failure_values = extract_crack_values(df, upper_bound=1.6)
#     print("数据范围：lower_bound =", lower_bound)
#     print("failure 数据个数 (>1.6):", len(failure_values))
#     print("non-failure 数据个数 (<=1.6):", len(non_failure_values))
    
#     # 计算发射参数：6 个非 failure 状态 + 1 failure 状态，共 7 个状态
#     mu, sigma2, bins = compute_emission_params(non_failure_values, failure_values, lower_bound, 1.6, n_non_failure_states=6)
#     print("各状态均值 mu:", mu)
#     print("各状态方差 sigma2:", sigma2)
    
#     # 构造 HMM 参数
#     N = 7  # 状态 0~5：非 failure；状态 6：failure
#     # 初始状态假设从非 failure 状态 0 开始
#     pi = np.zeros(N)
#     pi[0] = 1.0
#     # 构造转移矩阵：假设非 failure 状态按顺序转移，最后从状态 5转移到 failure 状态
#     A = np.zeros((N, N))
#     for i in range(5):
#         A[i, i] = 0.8
#         A[i, i+1] = 0.2
#     # 对于状态 5：保留 70%，转移到 failure 状态 6：30%
#     A[5, 5] = 0.7
#     A[5, 6] = 0.3
#     # failure 状态为吸收状态
#     A[6, 6] = 1.0
    
#     print("初始状态分布 pi:", pi)
#     print("状态转移矩阵 A:\n", A)
    
#     # 选择一条观测序列进行 Viterbi 解码
#     # 这里选取数据中第一行（剔除缺失值）
#     obs_seq = df.iloc[0].dropna().values
#     print("选取的观测序列 (长度 {}):".format(len(obs_seq)), obs_seq)
    
#     # 运行 Viterbi 算法
#     state_seq, delta, phi, P_star = viterbi_algorithm(obs_seq, pi, A, mu, sigma2)
#     print("最优状态路径：", state_seq)
#     print("最优路径概率 P* = ", P_star)
    
#     # 可视化 Viterbi 过程：delta 矩阵热图和最优路径图
#     plot_delta_heatmap(delta)
#     plot_viterbi_path(obs_seq, state_seq)

# if __name__ == "__main__":
#     main()


def decode_all_sequences(df, pi, A, mu, sigma2):
    """
    对 DataFrame 中每一行（样本）运行 Viterbi 算法，
    返回一个二维 numpy 数组，行对应样本，列对应时间步，
    数值为解码得到的状态编号。
    """
    state_matrix = []
    # 遍历每一行样本（剔除缺失值）
    for index, row in df.iterrows():
        obs_seq = row.dropna().values
        state_seq, _, _, _ = viterbi_algorithm(obs_seq, pi, A, mu, sigma2)
        state_matrix.append(state_seq)
        max_len = max(len(seq) for seq in state_matrix)
        padded = [np.pad(seq, (0, max_len - len(seq)), constant_values=-1) for seq in state_matrix]
    return np.array(padded)
    # return np.array(state_matrix)

def plot_state_heatmap(state_matrix):
    """
    绘制状态矩阵的热图，
    横轴为时间步，纵轴为样本索引，颜色表示对应的隐藏状态编号。
    """
    plt.figure(figsize=(10,6))
    plt.imshow(state_matrix, aspect='auto', cmap='tab20', origin='lower')
    plt.xlabel("Time Step")
    plt.ylabel("Sample Index")
    plt.title("Viterbi Decoded States Across All Samples")
    cbar = plt.colorbar()
    cbar.set_label("State")
    plt.show()

def main():
    # 1. 数据读取与预处理
    filename = 'train_FD002_pivoted.csv'
    df = load_data(filename)
    
    # 提取 crack 数值，并分为 non-failure 和 failure 数据
    lower_bound, failure_values, non_failure_values = extract_crack_values(df, upper_bound=1.6)
    print("数据范围：lower_bound =", lower_bound)
    print("failure 数据个数 (>1.6):", len(failure_values))
    print("non-failure 数据个数 (<=1.6):", len(non_failure_values))
    
    # 2. 计算发射概率参数：将 non-failure 数据分成6个箱，再加1个 failure 状态，共7个状态
    mu, sigma2, bins = compute_emission_params(non_failure_values, failure_values, lower_bound, 1.6, n_non_failure_states=6)
    print("各状态均值 mu:", mu)
    print("各状态方差 sigma2:", sigma2)
    
    # 3. 构造 HMM 参数（这里采用与之前类似的简单设定）
    N = 7  # 状态 0~5：non-failure，状态 6：failure
    pi = np.zeros(N)
    pi[0] = 1.0  # 假定所有样本最初都处于状态 0
    A = np.zeros((N, N))
    for i in range(5):
        A[i, i] = 0.8
        A[i, i+1] = 0.2
    A[5, 5] = 0.7
    A[5, 6] = 0.3
    A[6, 6] = 1.0  # failure 状态为吸收状态
    print("初始状态分布 pi:", pi)
    print("状态转移矩阵 A:\n", A)
    
    # 4. 对所有样本运行 Viterbi 算法
    state_matrix = decode_all_sequences(df, pi, A, mu, sigma2)
    print("解码得到的状态矩阵 (每行代表一个样本):\n", state_matrix)
    
    # 5. 可视化所有样本的状态矩阵
    plot_state_heatmap(state_matrix)

if __name__ == "__main__":
    main()