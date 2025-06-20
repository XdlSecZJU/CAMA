import pandas as pd
import numpy as np

# 读取 p_full 数据
p_full_path = "./E3_codellama.csv"
df_full = pd.read_csv(p_full_path)

# 读取 p_reduced 数据
p_reduced_path = "./E3_starchat_remove8.csv"
df_reduced = pd.read_csv(p_reduced_path)

# 确保两个 DataFrame 的 sha256 列一致
df_full = df_full.set_index('sha256')
df_reduced = df_reduced.set_index('sha256')

# 确保概率列表是浮点数列表
df_full['概率列表'] = df_full['概率列表'].apply(eval)
df_reduced['概率列表'] = df_reduced['概率列表'].apply(eval)

# 计算 MFS
mfs_scores = []
for sha256 in df_full.index:
    if sha256 in df_reduced.index:
        # 获取 p_full 的预测类别概率
        p_full_pred_label_prob = df_full.loc[sha256, '预测类别概率']
        # 获取预测类别
        pred_label = df_full.loc[sha256, '预测类别']
        # 获取 p_reduced 在 p_full 的预测类别上的概率
        p_reduced_pred_label_prob = df_reduced.loc[sha256, '概率列表'][pred_label]
        
        # 确保 p_reduced_pred_label_prob 是浮点数
        if not isinstance(p_reduced_pred_label_prob, (int, float)):
            raise ValueError(f"p_reduced_pred_label_prob for {sha256} is not a number: {p_reduced_pred_label_prob}")
        
        """
        print(sha256)
        print("\n")
        print(p_full_pred_label_prob)
        print("\n")
        print(pred_label)
        print("\n")
        print(p_reduced_pred_label_prob)
        print("\n")
        """

        # 计算 MFS
        mfs = (p_full_pred_label_prob - p_reduced_pred_label_prob) / p_full_pred_label_prob
        mfs_scores.append(mfs)
        print(sha256)
        print("\n")
        print(mfs)
        print("\n\n")
    else:
        print(f"Warning: {sha256} not found in reduced dataset.")

# 计算 MFS 平均值
if mfs_scores:
    avg_mfs = np.mean(mfs_scores)
    std_mfs = np.std(mfs_scores)
    print(f"平均 MFS: {avg_mfs:.4f}")
    print(f"MFS 标准差: {std_mfs:.4f}")
else:
    print("没有有效的 MFS 分数。")