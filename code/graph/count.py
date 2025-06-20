import os
import json
import csv
from tqdm import tqdm
import pandas as pd


# 读取apk_info_filter.csv文件中的sha256列
def read_sha256_from_csv(file_path):
    sha256_list = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 跳过表头
        for row in csvreader:
            sha256_list.append(row[0])  # 假设sha256是第一列
    return sha256_list


# 读取json文件并解析
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        return json.load(jsonfile)


# 替换Function Code中的内容
def replace_function_code(function_code, suggested_name):
    start = function_code.find('(')
    end = function_code.find(' ', start)
    if start != -1 and end != -1:
        return function_code[:start] + suggested_name + function_code[end:]
    return function_code


# 主程序
if __name__ == "__main__":
    # 读取sha256值
    sha256_list = read_sha256_from_csv('./apk_info_filter.csv')

    # 初始化分数列表
    before_scores = []
    after_scores = []

    # 遍历sha256值
    for sha256 in tqdm(sha256_list, desc="Processing SHA256"):
        before_json_path = os.path.join('./codellama_output', f"{sha256}.json")
        after_json_path = os.path.join('./codellama_output_replace', f"{sha256}.json")

        if os.path.exists(before_json_path) and os.path.exists(after_json_path):
            before_data = read_json(before_json_path)
            after_data = read_json(after_json_path)

            for before_item in before_data:
                function_code = before_item.get('Function Code', '')
                suggested_name = before_item.get('Suggested Function Name', '')
                modified_code = replace_function_code(function_code, suggested_name)

                for after_item in after_data:
                    if after_item.get('Function Code') == modified_code:
                        before_scores.append(before_item.get('Malicious Score', ''))
                        after_scores.append(after_item.get('Malicious Score', ''))
                        break

            # 统计0-10的分数分布
        score_bins = list(range(11))
        before_counts = pd.Series(before_scores).value_counts().reindex(score_bins, fill_value=0)
        after_counts = pd.Series(after_scores).value_counts().reindex(score_bins, fill_value=0)

        # 打印统计结果
        print("Before Scores Distribution:")
        print(before_counts)
        print("\nAfter Scores Distribution:")
        print(after_counts)