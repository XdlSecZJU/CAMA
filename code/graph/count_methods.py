import os
import pandas as pd
import androguard.core.bytecodes.apk as apk
import androguard.core.bytecodes.dvm as dvm
from androguard.core.analysis.analysis import Analysis
from tqdm import tqdm

# 路径配置
CSV_PATH = "./malware_category_label.csv"
APK_DIR = "./apks"
OUTPUT_CSV = "./method_counts.csv"

# 读取 CSV 并获取 SHA256 列表
def get_sha256_list(csv_path):
    df = pd.read_csv(csv_path)
    return df['sha256'].tolist()

# 遍历目录查找 APK
def find_matching_apks(sha256_list, apk_dir):
    return {sha256: os.path.join(apk_dir, f"{sha256}.apk") 
            for sha256 in sha256_list 
            if os.path.exists(os.path.join(apk_dir, f"{sha256}.apk"))}

# 反编译并统计 APK 中的方法总数
def count_methods_in_apk(apk_path):
    method_count = 0
    try:
        a = apk.APK(apk_path)
        d = dvm.DalvikVMFormat(a.get_dex())
        dx = Analysis(d)
        methods = list(d.get_methods())
        method_count = len(methods)
    except Exception as e:
        print(f"Failed to process APK {apk_path}: {e}")
    return method_count

# 统计所有 APK 的方法总数并保存到 CSV 文件
def count_and_save_method_counts(sha256_list, apk_dir, output_csv):
    apk_files = find_matching_apks(sha256_list, apk_dir)
    total_apks = len(apk_files)
    
    # 创建一个列表来存储结果
    results = []

    with tqdm(total=total_apks, desc="Counting Methods in APKs", position=0) as apk_bar:
        for sha256, apk_path in apk_files.items():
            method_count = count_methods_in_apk(apk_path)
            results.append({"apk_name": sha256, "method_count": method_count})
            apk_bar.update(1)
            apk_bar.set_postfix({"Processed": f"{apk_bar.n}/{total_apks}"})

    # 将结果保存到 CSV 文件
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Method counts saved to {output_csv}")

# 主函数
def main():
    sha256_list = get_sha256_list(CSV_PATH)
    count_and_save_method_counts(sha256_list, APK_DIR, OUTPUT_CSV)

if __name__ == "__main__":
    main()