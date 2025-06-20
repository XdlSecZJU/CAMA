import os
import shutil
import csv

# 定义文件夹路径
apks_folder = '/data/home/Yiling/Hongyu/lamd_malware/apks'  # 存放所有apk文件的文件夹
filter_apks_folder = '/data/home/Yiling/Hongyu/lamd_malware/filter_apks'  # 存放筛选后的apk文件的文件夹

# 创建目标文件夹，如果不存在的话
if not os.path.exists(filter_apks_folder):
    os.makedirs(filter_apks_folder)

# 打开csv文件并读取sha256值
with open('/data/home/Yiling/Hongyu/data/apk_info_filter.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sha256 = row['sha256']
        # 构造源文件路径和目标文件路径
        source_file = os.path.join(apks_folder, f'{sha256}.apk')
        target_file = os.path.join(filter_apks_folder, f'{sha256}.apk')
        # 检查文件是否存在，如果存在则复制到目标文件夹
        if os.path.exists(source_file):
            shutil.copy(source_file, target_file)
            print(f'Copied {sha256}.apk to {filter_apks_folder}')
        else:
            print(f'{sha256}.apk not found in {apks_folder}')