import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

epsilon = 1e-10  # 避免 log(0) 错误
def normalize_distribution(p):
    """归一化分布，使其总和为 1"""
    p = np.array(p, dtype=np.float64)
    return p / (np.sum(p) + epsilon)

def kl_divergence(p, q):
    """计算 Kullback-Leibler 散度，归一化分布并使用 epsilon 避免 log(0)"""
    p = normalize_distribution(p) + epsilon
    q = normalize_distribution(q) + epsilon
    return np.sum(p * np.log(p / q))

def js_divergence(p, q):
    """计算 Jensen-Shannon 散度"""
    p, q = np.array(p, dtype=np.float64), np.array(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def compute_mcs(raw_scores, new_scores):
    """计算测量一致性分数 (MCS)"""
    raw_scores = normalize_distribution(raw_scores)  # 归一化
    new_scores = normalize_distribution(new_scores)  # 归一化
    jsd = js_divergence(raw_scores, new_scores)
    return 1 - jsd / np.log(2)  # 归一化到 (0,1)

# 读取 sha256 列表
apk_info_path = "./apk_info_filter.csv"
apk_df = pd.read_csv(apk_info_path, usecols=["sha256"])
sha256_list = apk_df["sha256"].tolist()

# 处理每个 json 文件
output_folder = "./codellama_regrade_output"
results = []

for sha256 in tqdm(sha256_list, desc="Processing APKs"):
    json_path = os.path.join(output_folder, f"{sha256}.json")
    
    if not os.path.exists(json_path):
        continue  # 跳过不存在的文件
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 提取有效的 Malicious Score
    raw_scores, new_scores = [], []
    for entry in data:
        raw = entry["Malicious Score"]
        new = entry["New Malicious Score"]
        
        if new == "Unknown":
            continue  # 过滤掉 Unknown 值
        
        raw_scores.append(float(raw))
        new_scores.append(float(new))

    if len(raw_scores) > 1:  # 至少需要两个值才能计算 JSD
        mcs = compute_mcs(raw_scores, new_scores)
    else:
        mcs = None  # 无法计算时设为 None

    results.append({"sha256": sha256, "MCS": mcs})

# 保存结果
output_csv = "./MCS_codellama.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"处理完成，结果已保存到 {output_csv}")
