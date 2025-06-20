import os
import pandas as pd
from tqdm import tqdm  # 引入进度条库

# 文件夹路径
apk_info_path = "./apk_info_filter.csv"
codet5_score_dir = "./codet5_output_score"
output_csv_path = "./score_10_count.csv"  # 输出 csv 文件路径

# 读取 apk_info_filter.csv 获取 SHA256 列
apk_info_df = pd.read_csv(apk_info_path)
sha256_list = apk_info_df.iloc[:, 0].tolist()  # 读取第一列（SHA256值）

# 初始化一个列表，用于存储统计结果
score_10_count_list = []

# 遍历 SHA256 列表，添加进度条
for sha256 in tqdm(sha256_list, desc="Processing SHA256 files"):
    score_file = os.path.join(codet5_score_dir, f"{sha256}.csv")
    
    # 如果评分文件存在
    if os.path.exists(score_file):
        try:
            # 读取评分文件
            score_df = pd.read_csv(score_file)
            if 'Score' not in score_df.columns:
                print(f"警告: {score_file} 缺少 Score 列，跳过")
                continue
            # 统计 score=10 的个数
            score_10_count = score_df[score_df['Score'] == 10].shape[0]
            # 将统计结果添加到列表中
            score_10_count_list.append({"SHA256": sha256, "Score_10_Count": score_10_count})
        except Exception as e:
            print(f"处理 {sha256} 时发生错误: {e}")
    else:
        print(f"文件不存在，跳过: {sha256}")

# 将统计结果保存到 csv 文件中
score_10_count_df = pd.DataFrame(score_10_count_list)
score_10_count_df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"统计结果已保存到 {output_csv_path}")