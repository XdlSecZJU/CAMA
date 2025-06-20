import os
import pandas as pd
import Levenshtein
from tqdm import tqdm

def calculate_similarity(n_raw, n_reg):
    if pd.isna(n_raw) or pd.isna(n_reg):  # 检查是否为空
        return 0 
    n_raw, n_reg = str(n_raw), str(n_reg)  # 确保输入是字符串
    edit_distance = Levenshtein.distance(n_raw, n_reg)
    max_length = max(len(n_raw), len(n_reg))
    return 1 - edit_distance / max_length if max_length > 0 else 0

def process_apk_info(apk_info_path, name_consistency_dir, output_csv):
    # 读取 apk_info_filter.csv
    apk_info_df = pd.read_csv(apk_info_path)
    sha256_list = apk_info_df.iloc[:, 0].tolist()  # 获取第一列的 SHA256
    
    results = []
    
    # 使用 tqdm 包裹循环，添加进度条
    for sha256 in tqdm(sha256_list, desc="Processing APKs", unit="APK"):
        file_path = os.path.join(name_consistency_dir, f"{sha256}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if "Suggested Function Name" in df.columns and "New Suggested Function Name" in df.columns:
                similarities = [
                    calculate_similarity(row["Suggested Function Name"], row["New Suggested Function Name"])
                    for _, row in df.iterrows()
                ]
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            else:
                avg_similarity = 0
        else:
            avg_similarity = 0
        
        results.append([sha256, avg_similarity])
    
    # 保存结果到 CSV
    result_df = pd.DataFrame(results, columns=["SHA256", "NCS"])
    result_df.to_csv(output_csv, index=False)

# 示例调用
process_apk_info("./apk_info_filter.csv", "./name_consistency/codet5_filter", "E1_result_codet5_filter.csv")