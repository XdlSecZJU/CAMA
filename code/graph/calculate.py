import pandas as pd

# 替换为你的 CSV 文件路径
file_path = "./E2_codellama_top10_output_replace_2gram.csv"

# 使用 open() 函数打开文件，并指定 errors 参数
with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    # 将文件对象传递给 pandas.read_csv()
    df = pd.read_csv(file)

# 检查指定的列是否存在
columns_to_check = ['BLEU', 'METEOR', 'ROUGE-L', 'Similarity Score', 'MCS']
for column_name in columns_to_check:
    if column_name not in df.columns:
        print(f"列 '{column_name}' 不存在于文件中。")
        continue

    # 获取指定列的数据
    column_data = df[column_name]

    # 计算均值
    mean_value = column_data.mean()
    print(f"列 '{column_name}' 的均值：{mean_value}")

    # 计算方差
    variance_value = column_data.var()
    print(f"列 '{column_name}' 的方差：{variance_value}")