import os
import json
import pandas as pd
import re
from vllm import LLM, SamplingParams
import tqdm

# 设置路径
CSV_PATH = "./apk_info_filter.csv"
INPUT_DIR = "./codellama_output"
OUTPUT_DIR = "./codellama_regrade_output"
MODEL_PATH = "./CodeLlama-7b-Instruct-hf"
BATCH_SIZE = 128  # 设置批量大小

# 加载 vLLM 模型
llm = LLM(model=MODEL_PATH, tensor_parallel_size=4, max_model_len=1024, max_num_seqs=128, dtype="half")
sampling_params = SamplingParams(temperature=0.3, top_k=10, top_p=0.5, max_tokens=256)

# 读取 CSV 并获取 SHA256 列
def get_sha256_list(csv_path):
    df = pd.read_csv(csv_path)  # 读取 CSV
    sha256_list = df.iloc[:, 0].tolist()  # 取第一列
    return sha256_list

# 修改 Function Summary
def modify_function_summary(function_summary, suggested_function_name):
    modified_summary = function_summary.replace("This function", suggested_function_name)
    return modified_summary

# 让模型生成 new malicious score（批处理）
def model_inference_batch(modified_summaries, batch_size=BATCH_SIZE):
    if not modified_summaries:
        return []

    prompts = [
        (
            "### Instruction\n"
            "Rate the function summary's maliciousness on a scale from 0 to 10, where:\n"
            "    - 0 - Benign : No suspicious activity.\n"
            "    - 1-3 - Potentially Safe but Risky : Performs sensitive actions but could be legitimate.\n"
            "    - 4-6 - Suspicious : Uses permissions or techniques common in malware.\n"
            "    - 7-10 - Highly Malicious : Strong indicators of malware behavior.\n\n"
            "### Summary\n"
            + summary +
            "\n\n### Malicious Score\n"
        ) for summary in modified_summaries
    ]

    outputs = llm.generate(prompts, sampling_params)

    new_malicious_scores = []
    for output in outputs:
        # 假设模型输出的恶意分数是输出文本中的第一个数字
        match = re.search(r'\d+', output.outputs[0].text)
        if match:
            score = int(match.group())
            if 0 <= score <= 10:
                new_malicious_scores.append(score)
            else:
                new_malicious_scores.append("Unknown")
        else:
            new_malicious_scores.append("Unknown")

    return new_malicious_scores

# 处理所有 APK（添加 start_idx 和 end_idx 控制范围）
def process_apks(start_idx=0, end_idx=None, batch_size=BATCH_SIZE):
    sha256_list = get_sha256_list(CSV_PATH)

    # 如果 end_idx 为空或者超出范围，则取列表长度
    if end_idx is None or end_idx > len(sha256_list):
        end_idx = len(sha256_list)

    # 取指定范围的 sha256
    sha256_list = sha256_list[start_idx:end_idx]

    with tqdm.tqdm(total=len(sha256_list), desc="Processing APK Files") as pbar_apk:
        for sha256 in sha256_list:
            input_path = os.path.join(INPUT_DIR, f"{sha256}.json")
            output_path = os.path.join(OUTPUT_DIR, f"{sha256}.json")

            # 如果已处理，跳过
            if os.path.exists(output_path):
                print(f"Skipping {sha256}: already processed.")
                pbar_apk.update(1)
                continue

            # 读取 JSON 文件
            if not os.path.exists(input_path):
                print(f"Warning: {sha256}.json not found, skipping...")
                pbar_apk.update(1)
                continue

            with open(input_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error: {sha256}.json is not valid JSON, skipping...")
                    pbar_apk.update(1)
                    continue

            processed_results = []
            modified_summaries = []
            entries = []

            with tqdm.tqdm(total=len(data), desc=f"Processing {sha256}", leave=False) as pbar_json:
                for entry in data:
                    function_name = entry.get("Function Name", "")
                    function_code = entry.get("Function Code", "")
                    function_summary = entry.get("Function Summary", "")
                    suggested_function_name = entry.get("Suggested Function Name", "")
                    malicious_score = entry.get("Malicious Score", "")

                    # 修改 Function Summary
                    modified_summary = modify_function_summary(function_summary, suggested_function_name)

                    modified_summaries.append(modified_summary)
                    entries.append({
                        "Function Name": function_name,
                        "Function Code": function_code,
                        "Function Summary": function_summary,
                        "Suggested Function Name": suggested_function_name,
                        "Malicious Score": malicious_score
                    })

                    # 如果达到 batch_size，进行批量推理
                    if len(modified_summaries) == batch_size:
                        new_malicious_scores = model_inference_batch(modified_summaries, batch_size=batch_size)
                        for i in range(len(modified_summaries)):
                            entries[i]["New Malicious Score"] = new_malicious_scores[i]
                        processed_results.extend(entries)
                        modified_summaries = []
                        entries = []

                    pbar_json.update(1)

                # 处理剩余数据（不足 batch_size）
                if modified_summaries:
                    new_malicious_scores = model_inference_batch(modified_summaries, batch_size=len(modified_summaries))
                    for i in range(len(modified_summaries)):
                        entries[i]["New Malicious Score"] = new_malicious_scores[i]
                    processed_results.extend(entries)

            # 存入新的 JSON 文件
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_results, f, indent=4, ensure_ascii=False)

            pbar_apk.update(1)

    print(f"Processed APKs from index {start_idx} to {end_idx}.")

if __name__ == "__main__":
    process_apks(0, 120)