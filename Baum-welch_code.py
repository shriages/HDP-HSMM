# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # from scipy.stats import norm
# import os
# os.chdir('d:/My/清华大学/学习/4.2大四下/2毕业设计/MAPCSS')
# # ---------------------------
# # 1. 数据读取与离散化
# # ---------------------------
# def load_data(filename):
#     df = pd.read_csv(filename, sep=',', engine='python')
#     df.set_index('Path', inplace=True)
#     df = df.apply(pd.to_numeric, errors='coerce')
#     df.dropna(how='all', inplace=True)
#     return df

# def extract_crack_values(df, upper_bound=1.6):
#     """提取所有 crack 数值，并返回 non-failure 与 failure 部分"""
#     crack_values = df.values.flatten()
#     crack_values = crack_values[~np.isnan(crack_values)]
#     lower_bound = crack_values.min()
#     failure_values = crack_values[crack_values > upper_bound]
#     non_failure_values = crack_values[(crack_values >= lower_bound) & (crack_values <= upper_bound)]
#     return lower_bound, failure_values, non_failure_values

# def discretize_sequence(seq, lower_bound, upper_bound, n_non_failure_bins=6, failure_symbol=6):
#     """
#     将连续观测序列离散化：
#       - 对于值 <= upper_bound，使用等宽分箱，将其分为 n_non_failure_bins 个箱，箱索引从 0 到 n_non_failure_bins-1；
#       - 对于值 > upper_bound，赋值为 failure_symbol（例如6）。
#     返回离散化后的序列（列表或 numpy 数组）。
#     """
#     # 生成分箱边界，共 n_non_failure_bins+1 个边界
#     bins = np.linspace(lower_bound, upper_bound, n_non_failure_bins+1)
#     discrete_seq = []
#     for x in seq:
#         if x > upper_bound:
#             discrete_seq.append(failure_symbol)
#         else:
#             # np.digitize 返回从1开始的索引
#             d = np.digitize(x, bins, right=False) - 1
#             # 确保值在 0 到 n_non_failure_bins-1 内
#             d = min(max(d, 0), n_non_failure_bins-1)
#             discrete_seq.append(d)
#     return np.array(discrete_seq, dtype=int)

# # ---------------------------
# # 2. HMM 参数初始化
# # ---------------------------
# def initialize_hmm(n_states=7, n_symbols=7):
#     # 初始状态：假定所有序列从状态 0 开始
#     # pi = np.zeros(n_states)
#     # pi[0] = 1.0
#     pi = np.full(n_states, 1 / n_states)

#     # 状态转移矩阵 A
#     A = np.zeros((n_states, n_states))
#     # 对于状态 0～4：保持 0.8，自转 0.2 向下转移
#     for i in range(5):
#         A[i, i] = 0.8
#         A[i, i+1] = 0.2
#     # 对于状态 5：70% 保持，30% 转到 failure 状态（状态6）
#     # A[5, 5] = 0.7
#     # A[5, 6] = 0.3
#     A[5, 5] = 0.7
#     A[5, 6] = 0.3
#     # failure 状态为吸收状态
#     A[6, 6] = 1.0

#     # 发射概率矩阵 B，维度 (n_states, n_symbols)
#     # B = np.zeros((n_states, n_symbols))
#     B = np.random.dirichlet(np.ones(n_symbols), size=n_states)
#     # 对于非 failure 状态（0～5），初始均匀分布于符号 0～5，符号6设一个很小的概率
#     for i in range(6):
#         B[i, :6] = 1.0 / 6
#         B[i, 6] = 1e-6  # 很小概率
#     # 对于 failure 状态（状态6），主要发射符号6
#     B[6, :] = 0.0
#     B[6, 6] = 1.0
#     return pi, A, B

# # ---------------------------
# # 3. 离散 HMM 前向、后向算法
# # ---------------------------
# def forward_discrete(obs_seq, pi, A, B):
#     T = len(obs_seq)
#     N = len(pi)
#     alpha = np.zeros((T, N))
#     # 初始化
#     alpha[0, :] = pi * B[:, obs_seq[0]]
#     for t in range(1, T):
#         for j in range(N):
#             alpha[t, j] = np.sum(alpha[t-1, :] * A[:, j]) * B[j, obs_seq[t]]
#     return alpha

# def backward_discrete(obs_seq, A, B):
#     T = len(obs_seq)
#     N = A.shape[0]
#     beta = np.zeros((T, N))
#     beta[T-1, :] = 1.0
#     for t in range(T-2, -1, -1):
#         for i in range(N):
#             beta[t, i] = np.sum(A[i, :] * B[:, obs_seq[t+1]] * beta[t+1, :])
#     return beta

# # ---------------------------
# # 4. Baum–Welch 算法（EM）实现
# # ---------------------------
# def baum_welch_discrete(sequences, pi, A, B, n_iter=20, tol=1e-4):
#     """
#     对一组离散观测序列（每个序列为整数数组）运行 Baum-Welch 算法，
#     更新 HMM 参数 λ=(pi,A,B) 并返回参数和每次迭代的对数似然记录。
#     """
#     N = len(pi)
#     M = B.shape[1]
#     loglik_history = []
#     n_seq = len(sequences)

#     for iteration in range(n_iter):
#         # 初始化累积变量
#         pi_accum = np.zeros(N)
#         A_num = np.zeros((N, N))
#         A_den = np.zeros(N)
#         B_num = np.zeros((N, M))
#         B_den = np.zeros(N)
#         total_loglik = 0.0

#         # 对每个序列
#         for obs_seq in sequences:
#             T = len(obs_seq)
#             alpha = forward_discrete(obs_seq, pi, A, B)
#             beta = backward_discrete(obs_seq, A, B)
#             seq_loglik = np.log(np.sum(alpha[-1, :]) + 1e-10)
#             total_loglik += seq_loglik

#             # 计算 gamma 和 xi
#             gamma = np.zeros((T, N))
#             xi = np.zeros((T-1, N, N))
#             denom_seq = np.sum(alpha * beta, axis=1) + 1e-10  # T个标量
#             for t in range(T):
#                 gamma[t, :] = (alpha[t, :] * beta[t, :]) / denom_seq[t]
#             for t in range(T-1):
#                 for i in range(N):
#                     for j in range(N):
#                         xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, obs_seq[t+1]] * beta[t+1, j]
#                 xi[t, :, :] /= (np.sum(xi[t, :, :]) + 1e-10)

#             # 累计初始状态
#             pi_accum += gamma[0, :]

#             # 累计转移概率
#             for i in range(N):
#                 A_den[i] += np.sum(gamma[:-1, i])
#                 for j in range(N):
#                     A_num[i, j] += np.sum(xi[:, i, j])

#             # 累计发射概率
#             for t in range(T):
#                 k = obs_seq[t]
#                 for j in range(N):
#                     B_num[j, k] += gamma[t, j]
#                     B_den[j] += gamma[t, j]

#         # 更新参数
#         pi = pi_accum / n_seq
#         for i in range(N):
#             if A_den[i] > 0:
#                 A[i, :] = A_num[i, :] / A_den[i]
#             if B_den[i] > 0:
#                 B[i, :] = B_num[i, :] / B_den[i]

#         loglik_history.append(total_loglik)
#         print(f"Iteration {iteration+1}, Total Log-Likelihood: {total_loglik:.4f}")
#         if iteration > 0 and abs(loglik_history[-1] - loglik_history[-2]) < tol:
#             break

#     return pi, A, B, loglik_history

# # ---------------------------
# # 5. 可视化收敛情况
# # ---------------------------
# def plot_convergence(loglik_history):
#     plt.figure(figsize=(6,4))
#     plt.plot(loglik_history, marker='o')
#     plt.xlabel("Iteration")
#     plt.ylabel("Total Log-Likelihood")
#     plt.title("Baum-Welch Convergence")
#     plt.grid(True)
#     plt.show()

# # ---------------------------
# # 6. 主函数：整体流程
# # ---------------------------
# def main():
#     # 1. 读取数据
#     filename = 'train_FD002_pivoted.csv'
#     df = load_data(filename)
    
#     # 2. 提取 crack 数值，并分离出 non-failure 和 failure 数据
#     lower_bound, failure_values, non_failure_values = extract_crack_values(df, upper_bound=1.6)
#     print("数据范围: lower_bound =", lower_bound)
#     print("failure 数据个数 (>1.6):", len(failure_values))
#     print("non-failure 数据个数 (<=1.6):", len(non_failure_values))
    
#     # 3. 对每个样本（每一行）进行离散化
#     sequences = []
#     for idx, row in df.iterrows():
#         seq = row.dropna().values
#         disc_seq = discretize_sequence(seq, lower_bound, 1.6, n_non_failure_bins=6, failure_symbol=6)
#         sequences.append(disc_seq)
    
#     # 4. 初始化 HMM 参数（离散 HMM）：7 隐藏状态，7 观测符号
#     n_states = 7
#     n_symbols = 7
#     pi, A, B = initialize_hmm(n_states, n_symbols)
#     print("初始 π:", pi)
#     print("初始 A:\n", A)
#     print("初始 B:\n", B)
    
#     # 5. 使用 Baum-Welch 算法进行参数估计（EM 算法），记录收敛情况
#     pi_hat, A_hat, B_hat, loglik_history = baum_welch_discrete(sequences, pi, A, B, n_iter=50, tol=1e-8)
#     print("估计的 π:", pi_hat)
#     print("估计的 A:\n", A_hat)
#     print("估计的 B:\n", B_hat)
    
#     # 6. 可视化收敛情况
#     plot_convergence(loglik_history)

# if __name__ == "__main__":
#     main()



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# os.chdir('d:/My/清华大学/学习/4.2大四下/2毕业设计/MAPCSS')

# # ---------------------------
# # 1. 数据读取与离散化（保持不变）
# # ---------------------------
# def load_data(filename):
#     df = pd.read_csv(filename, sep=',', engine='python')
#     df.set_index('Path', inplace=True)
#     df = df.apply(pd.to_numeric, errors='coerce')
#     df.dropna(how='all', inplace=True)
#     return df

# def extract_crack_values(df, upper_bound=1.6):
#     crack_values = df.values.flatten()
#     crack_values = crack_values[~np.isnan(crack_values)]
#     lower_bound = crack_values.min()
#     failure_values = crack_values[crack_values > upper_bound]
#     non_failure_values = crack_values[(crack_values >= lower_bound) & (crack_values <= upper_bound)]
#     return lower_bound, failure_values, non_failure_values

# def discretize_sequence(seq, lower_bound, upper_bound, n_non_failure_bins=6, failure_symbol=6):
#     bins = np.linspace(lower_bound, upper_bound, n_non_failure_bins+1)
#     discrete_seq = []
#     for x in seq:
#         if x > upper_bound:
#             discrete_seq.append(failure_symbol)
#         else:
#             d = np.digitize(x, bins, right=False) - 1
#             d = min(max(d, 0), n_non_failure_bins-1)
#             discrete_seq.append(d)
#     return np.array(discrete_seq, dtype=int)

# # ---------------------------
# # 2. HMM 参数初始化（保持不变或根据需要修改）
# # ---------------------------
# def initialize_hmm(n_states=7, n_symbols=7):
#     pi = np.full(n_states, 1 / n_states)
#     A = np.zeros((n_states, n_states))
#     for i in range(5):
#         A[i, i] = 0.8
#         A[i, i+1] = 0.2
#     A[5, 5] = 0.7
#     A[5, 6] = 0.3
#     A[6, 6] = 1.0

#     # 这里用随机初始化，然后覆盖非-failure和failure状态的先验分布
#     B = np.random.dirichlet(np.ones(n_symbols), size=n_states)
#     for i in range(6):
#         B[i, :6] = 1.0 / 6
#         B[i, 6] = 1e-6
#     B[6, :] = 0.0
#     B[6, 6] = 1.0
#     return pi, A, B

# # ---------------------------
# # 3. 修改后的前向、后向算法（加入缩放因子）
# # ---------------------------
# def forward_discrete(obs_seq, pi, A, B):
#     T = len(obs_seq)
#     N = len(pi)
#     alpha = np.zeros((T, N))
#     scale = np.zeros(T)
#     # 初始化：计算 alpha[0]
#     alpha[0] = pi * B[:, obs_seq[0]]
#     scale[0] = alpha[0].sum()
#     alpha[0] /= scale[0]
#     # 递归计算
#     for t in range(1, T):
#         # alpha[t] = (alpha[t-1] @ A) * B[:, obs_seq[t]]
#         # 这里利用矩阵乘法计算上一时刻所有状态转移到当前状态的贡献
#         alpha[t] = (alpha[t-1] @ A) * B[:, obs_seq[t]]
#         scale[t] = alpha[t].sum()
#         # 如果 scale[t]==0, 加一个小常数防止除零
#         if scale[t] == 0:
#             scale[t] = 1e-10
#         alpha[t] /= scale[t]
#     loglik = np.log(scale).sum()
#     return alpha, loglik, scale

# def backward_discrete(obs_seq, A, B, scale):
#     T = len(obs_seq)
#     N = A.shape[0]
#     beta = np.ones((T, N))
#     # 初始化：最后一时刻 beta = 1 / scale[T-1]
#     beta[-1] /= scale[-1]
#     for t in range(T-2, -1, -1):
#         beta[t] = (A @ (B[:, obs_seq[t+1]] * beta[t+1])) / scale[t]
#     return beta

# # ---------------------------
# # 4. 修改后的 Baum-Welch 算法（使用缩放后的 alpha 和 beta）
# # ---------------------------
# def baum_welch_discrete(sequences, pi, A, B, n_iter=20, tol=1e-8):
#     N = len(pi)
#     M = B.shape[1]
#     loglik_history = []
#     n_seq = len(sequences)

#     for iteration in range(n_iter):
#         pi_accum = np.zeros(N)
#         A_num = np.zeros((N, N))
#         A_den = np.zeros(N)
#         B_num = np.zeros((N, M))
#         B_den = np.zeros(N)
#         total_loglik = 0.0

#         for obs_seq in sequences:
#             T = len(obs_seq)
#             alpha, seq_loglik, scale = forward_discrete(obs_seq, pi, A, B)
#             beta = backward_discrete(obs_seq, A, B, scale)
#             total_loglik += seq_loglik

#             # 计算 gamma (无需再归一化，因为缩放已经完成)
#             gamma = alpha * beta  # 每个时间步的 gamma 已经满足归一化条件

#             # 计算 xi
#             xi = np.zeros((T-1, N, N))
#             for t in range(T-1):
#                 obs_next = obs_seq[t+1]
#                 xi[t] = (alpha[t].reshape(-1, 1) * A * B[:, obs_next] * beta[t+1])
#                 # 除以 scale[t+1]（即对应于 alpha[t+1] 的缩放因子）
#                 xi[t] /= scale[t+1]
#                 # 为了确保数值稳定性，可以再归一化xi[t]
#                 xi[t] /= (xi[t].sum() + 1e-10)

#             pi_accum += gamma[0, :]

#             for i in range(N):
#                 A_den[i] += np.sum(gamma[:-1, i])
#                 for j in range(N):
#                     A_num[i, j] += np.sum(xi[:, i, j])
#             for t in range(T):
#                 k = obs_seq[t]
#                 for j in range(N):
#                     B_num[j, k] += gamma[t, j]
#                     B_den[j] += gamma[t, j]

#         pi = pi_accum / n_seq
#         for i in range(N):
#             if A_den[i] > 0:
#                 A[i, :] = A_num[i, :] / A_den[i]
#             if B_den[i] > 0:
#                 B[i, :] = B_num[i, :] / B_den[i]

#         loglik_history.append(total_loglik)
#         print(f"Iteration {iteration+1}, Total Log-Likelihood: {total_loglik:.4f}")
#         if iteration > 0 and abs(loglik_history[-1] - loglik_history[-2]) < tol:
#             break

#     return pi, A, B, loglik_history

# # ---------------------------
# # 5. 可视化收敛情况
# # ---------------------------
# def plot_convergence(loglik_history):
#     plt.figure(figsize=(6,4))
#     plt.plot(loglik_history, marker='o')
#     plt.xlabel("Iteration")
#     plt.ylabel("Total Log-Likelihood")
#     plt.title("Baum-Welch Convergence")
#     plt.grid(True)
#     plt.show()
    
    
# # ---------------------------
# # 6. 主函数：整体流程
# # ---------------------------
# def main():
#     # 读取数据（此处用你预先保存好的CSV文件）
#     filename = 'train_FD002_pivoted.csv'
#     df = load_data(filename)
    
#     lower_bound, failure_values, non_failure_values = extract_crack_values(df, upper_bound=1.6)
#     print("数据范围: lower_bound =", lower_bound)
#     print("failure 数据个数 (>1.6):", len(failure_values))
#     print("non-failure 数据个数 (<=1.6):", len(non_failure_values))
    
#     sequences = []
#     for idx, row in df.iterrows():
#         seq = row.dropna().values
#         disc_seq = discretize_sequence(seq, lower_bound, 1.6, n_non_failure_bins=6, failure_symbol=6)
#         sequences.append(disc_seq)
    
#     n_states = 7
#     n_symbols = 7
#     pi, A, B = initialize_hmm(n_states, n_symbols)
#     print("初始 π:", pi)
#     print("初始 A:\n", A)
#     print("初始 B:\n", B)
    
#     pi_hat, A_hat, B_hat, loglik_history = baum_welch_discrete(sequences, pi, A, B, n_iter=50, tol=1e-8)
#     print("估计的 π:", pi_hat)
#     print("估计的 A:\n", A_hat)
#     print("估计的 B:\n", B_hat)
    
#     plot_convergence(loglik_history)
    
    

# if __name__ == "__main__":
#     main()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('d:/My/清华大学/学习/4.2大四下/2毕业设计/MAPCSS')

EPSILON = 1e-6  # 平滑常数

# ---------------------------
# 1. 数据读取与离散化（保持不变）
# ---------------------------
def load_data(filename):
    df = pd.read_csv(filename, sep=',', engine='python')
    df.set_index('Path', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(how='all', inplace=True)
    return df

def extract_crack_values(df, upper_bound=1.6):
    crack_values = df.values.flatten()
    crack_values = crack_values[~np.isnan(crack_values)]
    lower_bound = crack_values.min()
    failure_values = crack_values[crack_values > upper_bound]
    non_failure_values = crack_values[(crack_values >= lower_bound) & (crack_values <= upper_bound)]
    return lower_bound, failure_values, non_failure_values

def discretize_sequence(seq, lower_bound, upper_bound, n_non_failure_bins=6, failure_symbol=6):
    bins = np.linspace(lower_bound, upper_bound, n_non_failure_bins+1)
    discrete_seq = []
    for x in seq:
        if x > upper_bound:
            discrete_seq.append(failure_symbol)
        else:
            d = np.digitize(x, bins, right=False) - 1
            d = min(max(d, 0), n_non_failure_bins-1)
            discrete_seq.append(d)
    return np.array(discrete_seq, dtype=int)

# ---------------------------
# 2. HMM 参数初始化
# ---------------------------
def initialize_hmm(n_states=7, n_symbols=7):
    # 初始状态分布：这里假定均匀分布
    pi = np.full(n_states, 1 / n_states)
    
    # 状态转移矩阵 A 的构造：
    A = np.zeros((n_states, n_states))
    for i in range(n_states-2):  # 前 n_states-2 状态
        A[i, i] = 0.8
        A[i, i+1] = 0.2
    # 对于倒数第二个状态（状态5）：
    A[n_states-2, n_states-2] = 0.7
    A[n_states-2, n_states-1] = 0.3
    # 最后一个状态为吸收状态
    A[n_states-1, n_states-1] = 1.0
    
    # 发射概率矩阵 B
    # 这里采用 Dirichlet 随机初始化，再覆盖非-failure与failure的先验分布
    B = np.random.dirichlet(np.ones(n_symbols), size=n_states)
    for i in range(n_states-1):
        # 非 failure 状态：均匀分布在 0~n_symbols-2，最后一个符号概率设为极小值
        B[i, :n_symbols-1] = 1.0 / (n_symbols-1)
        B[i, n_symbols-1] = 1e-6
    # 最后一个状态（failure 状态）：只发射最后一个符号
    B[n_states-1, :] = 0.0
    B[n_states-1, n_symbols-1] = 1.0
    
    return pi, A, B

# ---------------------------
# 3. 修改后的前向、后向算法（加入缩放因子）
# ---------------------------
def forward_discrete(obs_seq, pi, A, B):
    T = len(obs_seq)
    N = len(pi)
    alpha = np.zeros((T, N))
    scale = np.zeros(T)
    # 初始化：计算 alpha[0]
    alpha[0] = pi * B[:, obs_seq[0]]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]
    # 递归计算
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, obs_seq[t]]
        scale[t] = alpha[t].sum()
        if scale[t] == 0:
            scale[t] = 1e-10
        alpha[t] /= scale[t]
    loglik = np.log(scale).sum()
    return alpha, loglik, scale

def backward_discrete(obs_seq, A, B, scale):
    T = len(obs_seq)
    N = A.shape[0]
    beta = np.ones((T, N))
    beta[-1] /= scale[-1]
    for t in range(T-2, -1, -1):
        beta[t] = (A @ (B[:, obs_seq[t+1]] * beta[t+1])) / scale[t]
    return beta

# ---------------------------
# 4. 修改后的 Baum-Welch 算法（使用缩放后的 alpha 和 beta，并添加平滑项）
# ---------------------------
def baum_welch_discrete(sequences, pi, A, B, n_iter=20, tol=1e-8):
    N = len(pi)
    M = B.shape[1]
    loglik_history = []
    n_seq = len(sequences)

    for iteration in range(n_iter):
        pi_accum = np.zeros(N)
        A_num = np.zeros((N, N))
        A_den = np.zeros(N)
        B_num = np.zeros((N, M))
        B_den = np.zeros(N)
        total_loglik = 0.0

        for obs_seq in sequences:
            T = len(obs_seq)
            alpha, seq_loglik, scale = forward_discrete(obs_seq, pi, A, B)
            beta = backward_discrete(obs_seq, A, B, scale)
            total_loglik += seq_loglik

            # 计算 gamma（此处 alpha 与 beta 已经归一化）
            gamma = alpha * beta  # 此时 gamma[t] 的和不一定为1，需归一化
            for t in range(T):
                gamma[t] /= (gamma[t].sum() + 1e-10)

            # 计算 xi
            xi = np.zeros((T-1, N, N))
            for t in range(T-1):
                obs_next = obs_seq[t+1]
                xi[t] = (alpha[t].reshape(-1, 1) * A * B[:, obs_next] * beta[t+1])
                xi[t] /= scale[t+1]
                xi[t] /= (xi[t].sum() + 1e-10)

            pi_accum += gamma[0, :]

            for i in range(N):
                A_den[i] += np.sum(gamma[:-1, i])
                for j in range(N):
                    A_num[i, j] += np.sum(xi[:, i, j])
            for t in range(T):
                k = obs_seq[t]
                for j in range(N):
                    B_num[j, k] += gamma[t, j]
                    B_den[j] += gamma[t, j]

        pi = pi_accum / n_seq
        for i in range(N):
            # 在更新 A、B 时添加平滑项 EPSILON
            A[i, :] = (A_num[i, :] + EPSILON) / (A_den[i] + N * EPSILON)
            B[i, :] = (B_num[i, :] + EPSILON) / (B_den[i] + M * EPSILON)

        loglik_history.append(total_loglik)
        print(f"Iteration {iteration+1}, Total Log-Likelihood: {total_loglik:.4f}")
        if iteration > 0 and abs(loglik_history[-1] - loglik_history[-2]) < tol:
            break

    return pi, A, B, loglik_history

# ---------------------------
# 5. 可视化收敛情况
# ---------------------------
def plot_convergence(loglik_history):
    plt.figure(figsize=(6,4))
    plt.plot(loglik_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Total Log-Likelihood")
    plt.title("Baum-Welch Convergence")
    plt.grid(True)
    plt.show()

# ---------------------------
# 6. 主函数：整体流程
# ---------------------------
def main():
    # 读取数据（此处用预先保存好的 CSV 文件）
    filename = 'train_FD002_pivoted.csv'
    df = load_data(filename)
    
    lower_bound, failure_values, non_failure_values = extract_crack_values(df, upper_bound=1.6)
    print("数据范围: lower_bound =", lower_bound)
    print("failure 数据个数 (>1.6):", len(failure_values))
    print("non-failure 数据个数 (<=1.6):", len(non_failure_values))
    
    # 对每个样本（每行）进行离散化
    sequences = []
    for idx, row in df.iterrows():
        seq = row.dropna().values
        disc_seq = discretize_sequence(seq, lower_bound, 1.6, n_non_failure_bins=6, failure_symbol=6)
        sequences.append(disc_seq)
    
    n_states = 7
    n_symbols = 7
    pi, A, B = initialize_hmm(n_states, n_symbols)
    print("初始 π:", pi)
    print("初始 A:\n", A)
    print("初始 B:\n", B)
    
    pi_hat, A_hat, B_hat, loglik_history = baum_welch_discrete(sequences, pi, A, B, n_iter=50, tol=1e-8)
    print("估计的 π:", pi_hat)
    print("估计的 A:\n", A_hat)
    print("估计的 B:\n", B_hat)
    
    plot_convergence(loglik_history)

if __name__ == "__main__":
    main()

