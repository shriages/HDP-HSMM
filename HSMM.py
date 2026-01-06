import pandas as pd
import os
from pyhsmm import models
from pyhsmm.basic.distributions import Gaussian, NegativeBinomialDuration
from pybasicbayes.util.text import progprint_xrange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.stdout.flush()
os.chdir('d:/My/清华大学/学习/4.2大四下/2毕业设计/MAPCSS')
"""
engine_id: 发动机编号
cycle: 时间步长，每个编号的发动机有自己的循环时间，表示当前时间步。
setting_1，setting_2，setting_3：操作设置（engine operating conditions）。
s_1 - s_21：传感器测量值（sensor readings），表示发动机运行状态。
"""
column_name =  ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
       's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
       's15', 's16', 's17', 's18', 's19', 's20', 's21' ]
train_FD001 = pd.read_table("./train_FD001.txt",header=None,delim_whitespace=True) #  delim_whitespace以空格分隔的文本文件
train_FD002 = pd.read_table("./train_FD002.txt",header=None,delim_whitespace=True)
train_FD003 = pd.read_table("./train_FD003.txt",header=None,delim_whitespace=True)
train_FD004 = pd.read_table("./train_FD004.txt",header=None,delim_whitespace=True)
train_FD001.columns = column_name
train_FD002.columns = column_name
train_FD003.columns = column_name
train_FD004.columns = column_name

# for data in ['train_FD00' + str(i) for  i in range(1,5)]:
#     # have a look at the info of each data file
#     eval(data).info()
    
def compute_rul_of_one_id(train_FD00X_of_one_id):
    '''
    输入train_FD001的一个engine_id的数据，输出这些数据对应的RUL（剩余寿命），type为list
    '''
    max_cycle = max(train_FD00X_of_one_id['cycle'])  # 故障时的cycle
    rul_of_one_id = max_cycle - train_FD00X_of_one_id['cycle']
    return rul_of_one_id.tolist()

def compute_rul_of_one_file(train_FD00X):
    '''
    输入train_FD001，输出一个list
    '''
    rul = []
    # 循环train中，''engine_id''这一列的每一种id值
    for id in set(train_FD00X['engine_id']):
        rul.extend(compute_rul_of_one_id(train_FD00X[train_FD00X['engine_id'] == id]))
    return rul

def compute_scaled_rul_of_one_id(train_FD00X_of_one_id):
    '''
    输入train_FD001的一个engine_id的数据，输出这些数据对应的RUL（剩余寿命），type为list
    '''
    max_cycle = max(train_FD00X_of_one_id['cycle'])  # 故障时的cycle
    rul_of_one_id = max_cycle - train_FD00X_of_one_id['cycle']
    scaled_rul_of_one_id = rul_of_one_id / max_cycle
    return scaled_rul_of_one_id.tolist()

def compute_scaled_rul_of_one_file(train_FD00X):
    '''
    输入train_FD001，输出一个list
    '''
    rul = []
    # 循环train中，''engine_id''这一列的每一种id值
    for id in set(train_FD00X['engine_id']):
        rul.extend(compute_scaled_rul_of_one_id(train_FD00X[train_FD00X['engine_id'] == id]))
    return rul

RUL = {}
data_root = 'd:/My/清华大学/学习/4.2大四下/2毕业设计/终期/4数据验证/HDPHSMM' 
save_root = 'figures/'

# 为4个data增加RUL列
# for data_file in ['train_FD00' + str(i) for  i in range(1,5)]:
for data_file in ['train_FD00' + str(i) for  i in range(1,2)]:
    
    fd_name = data_file  # e.g., train_FD001
    save_dir = os.path.join(save_root, fd_name)
    os.makedirs(save_dir, exist_ok=True)
    
    i = data_file[-1]
    # have a look at the info of each data file
    eval(data_file)['RUL'] = compute_scaled_rul_of_one_file(eval(data_file))
    RUL[f'FD00{i}'] = eval(data_file)[['engine_id', 'cycle', 'RUL']].rename(columns = {'engine_id': 'Path'})
    # output_filename = f"{data_file}_with_RUL.csv"
    # eval(data_file).to_csv(output_filename, index=False)
    df = eval(data_file)
    rul_seqs = [group['RUL'].values.astype(float) for _, group in df.groupby('engine_id') if len(group) > 5]
    
    # # 原始RUL 散点图
    # plt.figure()
    # plt.plot(rul_seqs[0], 'kx')
    # plt.title(f'{fd_name} First Engine RUL Data')
    # plt.xlabel('Time Step')
    # plt.ylabel('RUL')
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, 'raw_rul_data.png'))
    # plt.close()
    
    # pivoted df
    # df = RUL[f'FD00{i}']
    # pivot_df = df.pivot(index='Path', columns='cycle', values='RUL')
    # pivot_df.to_csv(f"train_FD00{i}_pivoted.csv")
    
    # HDP HMM model construction
    obs_dim = 1
    Nmax = 20  # 最大可能的隐藏状态数目

    # 观测分布的超参数设置
    obs_hypparams = {
        'mu_0': np.zeros(1),
        'sigma_0': np.eye(1),
        'kappa_0': 0.25,
        'nu_0': 5
    }
    obs_distns = [Gaussian(**obs_hypparams) for _ in range(Nmax)]
    dur_hypparams = {
        'alpha_0': 2.0,
        'beta_0': 2.0,
        'k_0': 0.5,
        'theta_0': 1.0,
        'r': 5,
        'p':0.5
    }

    dur_distns = [NegativeBinomialDuration(**dur_hypparams) for _ in range(Nmax)]
    for d in dur_distns:
        d.r = np.float64(d.r)
        # print("r:", d.r, "type:", type(d.r))

    # 构造 HDP-HSMM 模型（替换原本的 WeakLimitHDPHMM）
    posteriormodel = models.WeakLimitHDPHSMM(
        alpha=6., gamma=6., init_state_concentration=6.,
        obs_distns=obs_distns,
        dur_distns=dur_distns
    )

    # # 加入所有 engine_id 的观测序列
    # for seq in rul_seqs:
    #     seq = seq.reshape(-1, 1)  # 因为观测值是一维的
    #     if len(seq) >= 2:
    #         posteriormodel.add_data(seq)
    #     posteriormodel.add_data(seq)

    for seq in rul_seqs:
        seq = np.asarray(seq, dtype=float).squeeze()
        if seq.ndim == 1 and len(seq) >= 2:
            posteriormodel.add_data(seq.reshape(-1, 1))

        
    for idx in progprint_xrange(150):
        posteriormodel.resample_model()

    for i, s in enumerate(posteriormodel.states_list):
        try:
            print(f"🔍 RUL shape: {rul_seqs[i].shape}, StateSeq length: {len(s.stateseq)}")
            plt.figure(figsize=(10, 4))
            plt.plot(rul_seqs[i], label='RUL')
            plt.plot(s.stateseq, label='Hidden States')
            plt.legend()
            plt.title(f'{fd_name} Engine {i+1}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.tight_layout()

            save_path = os.path.join(save_dir, f'engine_{i+1}_state_sequence.png')
            plt.savefig(save_path, dpi=300)
            print(f"✅ 图像保存成功：{save_path}")
            plt.close()
        except Exception as e:
            print(f"❌ Engine {i+1} 绘图失败：{str(e)}")


    save_root = 'd:/My/清华大学/学习/4.2大四下/2毕业设计/终期/HDPHSMM_结果图'
    save_dir = os.path.join(save_root, fd_name)
    os.makedirs(save_dir, exist_ok=True)
    print("图像将保存到：", save_dir)
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"✅ 图像将保存到目录：{save_dir}")
    except Exception as e:
        print(f"❌ 创建目录失败：{save_dir}")
        raise e
    

    for i, s in enumerate(posteriormodel.states_list):
        plt.figure()
        plt.plot(rul_seqs[i], label='RUL')
        plt.plot(s.stateseq, label='Hidden States')
        plt.legend()
        plt.title(f'Engine {i+1}')
        save_path = os.path.join(save_dir, f'engine_{i+1}.png')
        plt.savefig(save_path,dpi=300)
        plt.close()

    # Gibbs Sampling
    allscores = []
    for itr in progprint_xrange(150):
        posteriormodel.resample_model()

    plt.figure()
    for scores in allscores:
        plt.plot(scores)
    plt.title(f'{fd_name} model vlb scores vs iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Variational Lower Bound')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vlb_scores.png'))
    plt.close()

    # 最佳模型的状态预测
        
    for i, s in enumerate(posteriormodel.states_list):
        plt.figure(figsize=(10, 4))
        plt.plot(rul_seqs[i], label='RUL')
        plt.plot(s.stateseq, label='Hidden States')
        plt.legend()
        plt.title(f'{fd_name} Engine {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'engine_{i+1}_state_sequence.png'))
        plt.close()
    
    # posteriormodel.plot()
    plt.title('Best model structure')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_model_plot.png'))
    plt.close()
