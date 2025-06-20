import os
import json
import pandas as pd
import random

def get_top_k_functions(sha256, k):
    # 读取sha256.csv文件
    csv_file = f"./{sha256}.csv"
    if not os.path.exists(csv_file):
        return None
    df = pd.read_csv(csv_file)
    # 按分数降序排序
    df = df.sort_values(by="Malicious Score", ascending=False)
    # 如果有同分的情况，随机选取
    top_k_indices = []
    current_score = None
    current_indices = []
    for index, row in df.iterrows():
        if len(top_k_indices) >= k:
            break
        if row["Malicious Score"] != current_score:
            if len(current_indices) > 0:
                top_k_indices.extend(random.sample(current_indices, min(len(current_indices), k - len(top_k_indices))))
            current_score = row["Malicious Score"]
            current_indices = [row["Index"]]
        else:
            current_indices.append(row["Index"])
    if len(top_k_indices) < k and len(current_indices) > 0:
        top_k_indices.extend(random.sample(current_indices, min(len(current_indices), k - len(top_k_indices))))
    
    # 确保所有索引都是Python原生的int类型
    top_k_indices = [int(idx) for idx in top_k_indices]
    
    print(top_k_indices)
    print("\n")
    return top_k_indices

def main():
    # 读取apk_info_filter.csv文件
    apk_info_file = "./apk_info_filter.csv"
    df = pd.read_csv(apk_info_file)
    # 指定k值
    k = 30
    # 创建结果存储的json文件
    result_file = "./top_30_indices_codellama_replace.json"
    results = {}
    # 遍历sha256值
    for i in range(len(df)):
        sha256 = df.iloc[i]["sha256"]
        print(sha256)
        print("\n")
        top_k_indices = get_top_k_functions(sha256, k)
        if top_k_indices is not None:
            results[sha256] = top_k_indices
    # 将结果存储到json文件中
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()