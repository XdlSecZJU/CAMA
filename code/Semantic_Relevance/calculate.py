import csv
import json
import os
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk.tokenize import word_tokenize
import nltk
import re

# 初始化进度条
def init_progress_bar(total):
    return tqdm(total=total, desc="Processing SHA256", unit="sha256")

# 读取CSV文件中的SHA256值
def read_sha256_from_csv(csv_file):
    sha256_list = []
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sha256_list.append(row['sha256'])
    return sha256_list

# 从JSON文件中获取SHA256对应的描述内容
def get_description_from_json(json_file, sha256):
    try:
        with open(json_file, mode='r', encoding='utf-8') as file:
            data = json.load(file)
        return data.get(sha256, "")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for SHA256 {sha256}: {e}")
        return ""

# 从日志文件中获取Application Purpose内容
def get_application_purpose_from_log(log_folder, sha256):
    log_file = os.path.join(log_folder, f"{sha256}.log")
    if not os.path.exists(log_file):
        return ""
    with open(log_file, mode='r', encoding='utf-8') as file:
        log_content = file.read()
    start_marker = "**Application Purpose:**"
    start = log_content.find(start_marker) + len(start_marker)
    if start != -1:
        # 从start_marker后开始，找到第一个句点的位置
        end = log_content.find('.', start)
        if end != -1:
            return log_content[start:end].strip()
        else:
            return log_content[start:].strip()
    return ""

# 替换每个非ASCII字符为占位符
def replace_non_ascii(text, placeholder="[NON-ASCII]"):
    # 使用正则表达式替换每个非ASCII字符
    return re.sub(r'[^\x00-\x7F]', placeholder, text)

# 计算BLEU、METEOR和ROUGE-L值
def calculate_scores(reference, hypothesis):
    # 替换每个非ASCII字符为占位符
    reference = replace_non_ascii(reference)
    hypothesis = replace_non_ascii(hypothesis)
    
    # BLEU
    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
    
    # METEOR
    meteor = meteor_score([reference_tokens], hypothesis_tokens)
    
    # ROUGE-L
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypothesis, reference)
    rouge_l_score = rouge_scores[0]['rouge-l']['f']
    
    return bleu_score, meteor, rouge_l_score

# 主函数
def main():
    csv_file = './apk_info_filter.csv'
    json_file = './E2_codet5_top10_output.json'
    log_folder = './malware_analysis_logs'
    output_csv = './E2_codet5_top10.csv'
    
    sha256_list = read_sha256_from_csv(csv_file)
    progress_bar = init_progress_bar(len(sha256_list))
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['sha256', 'description', 'application_purpose', 'BLEU', 'METEOR', 'ROUGE-L'])
        
        for sha256 in sha256_list:
            description = get_description_from_json(json_file, sha256)
            application_purpose = get_application_purpose_from_log(log_folder, sha256)

            if description and application_purpose:
                bleu, meteor, rouge_l = calculate_scores(description, application_purpose)
                writer.writerow([sha256, description, application_purpose, bleu, meteor, rouge_l])
            
            progress_bar.update(1)
    
    progress_bar.close()
    print("Processing completed.")

if __name__ == "__main__":
    main()