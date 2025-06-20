import csv
import json
import os

# 设置k的值，范围是0-30
k = 8  # 示例，你可以根据需要修改这个值

# 读取apk_info_filter.csv文件中的sha256列
csv_file_path = './apk_info_filter.csv'
sha256_values = []
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        sha256_values.append(row['sha256'])

# 加载top_30_indices_starchat.json文件
json_file_path = './top_30_indices_codellama_replace.json'
with open(json_file_path, mode='r', encoding='utf-8') as jsonfile:
    top_30_indices = json.load(jsonfile)

# 设置新的保存路径
output_folder = './codellama_output_replace_remove8'
os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在，则创建它

# 遍历每个sha256值
for sha256 in sha256_values:
    # 如果该sha256值在top_30_indices_starchat.json中存在
    if sha256 in top_30_indices:
        # 获取前k个index值
        indices_to_remove = top_30_indices[sha256][:k]
        
        # 构造对应的sha256.json文件路径
        sha256_json_file_path = os.path.join('./codellama_output_replace', f'{sha256}.json')
        
        # 如果该文件存在
        if os.path.exists(sha256_json_file_path):
            # 加载该json文件
            with open(sha256_json_file_path, mode='r', encoding='utf-8') as jsonfile:
                json_data = json.load(jsonfile)
            
            # 删除指定的index值对应的json项
            indices_to_remove.sort(reverse=True)  # 从大到小排序，避免索引变化影响删除操作
            for index in indices_to_remove:
                if 0 <= index < len(json_data):  # 确保索引有效
                    del json_data[index]
            
            # 构造新的保存路径
            new_sha256_json_file_path = os.path.join(output_folder, f'{sha256}.json')
            
            # 保存修改后的json文件
            with open(new_sha256_json_file_path, mode='w', encoding='utf-8') as jsonfile:
                json.dump(json_data, jsonfile, indent=4)
            print(f'已处理文件：{new_sha256_json_file_path}')
        else:
            print(f'文件不存在：{sha256_json_file_path}')
    else:
        print(f'sha256值未在top_30_indices_starchat.json中找到：{sha256}')