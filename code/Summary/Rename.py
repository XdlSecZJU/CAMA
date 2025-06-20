import os
import json
import pandas as pd
import re
from vllm import LLM, SamplingParams
import tqdm

# 设置路径
CSV_PATH = "./apk_info_filter.csv"
INPUT_DIR = "./codellama_output"
OUTPUT_DIR = "./codellama_rename_output"
MODEL_PATH = "./CodeLlama-7b-Instruct-hf"
BATCH_SIZE = 128  # 设置批量大小

# 加载 vLLM 模型
llm = LLM(model=MODEL_PATH, tensor_parallel_size=2, max_model_len=2048, max_num_seqs=256, dtype="half")
sampling_params = SamplingParams(temperature=0.3, top_k=10, top_p=0.5, max_tokens=256)

# 读取 CSV 并获取 SHA256 列
def get_sha256_list(csv_path):
    df = pd.read_csv(csv_path)  # 读取 CSV
    return df.iloc[:, 0].tolist()  # 取第一列

# 让模型生成新函数名（批处理）
def model_inference_batch(summaries):
    if not summaries:
        return []

    prompts = [
        (
            "Suggest a more descriptive function name based on the function summary below:"
            "\n\n    " + summary +
            "\n\nFunction Name:"
        ) for summary in summaries
    ]

    outputs = llm.generate(prompts, sampling_params)

    new_function_names = []
    for output in outputs:
        response_text = output.outputs[0].text.strip()

        # **提取第一行非空内容**
        function_name = ""
        for line in response_text.splitlines():
            clean_line = line.strip()
            if clean_line and clean_line != "?" and clean_line != "`":  # 找到第一行非空内容
                function_name = clean_line
                break

        # **去掉 `()`, `,`**
        function_name = re.sub(r'[()`]', '', function_name)

        # **如果 `Extracted Name` 为空，或长度超过50，则返回 `"Unknown"`**
        if not function_name or len(function_name) > 64:
            function_name = "Unknown"

        new_function_names.append(function_name)

    return new_function_names

# 处理所有 APK（添加 start_idx 和 end_idx 控制范围）
def process_apks(start_idx=0, end_idx=None, batch_size=BATCH_SIZE):
    sha256_list = get_sha256_list(CSV_PATH)

    if end_idx is None or end_idx > len(sha256_list):
        end_idx = len(sha256_list)

    sha256_list = sha256_list[start_idx:end_idx]

    with tqdm.tqdm(total=len(sha256_list), desc="Processing APK Files") as pbar_apk:
        for sha256 in sha256_list:
            input_path = os.path.join(INPUT_DIR, f"{sha256}.json")
            output_path = os.path.join(OUTPUT_DIR, f"{sha256}.json")

            if os.path.exists(output_path):
                print(f"Skipping {sha256}: already processed.")
                pbar_apk.update(1)
                continue

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
            summaries = []
            entries = []

            with tqdm.tqdm(total=len(data), desc=f"Processing {sha256}", leave=False) as pbar_json:
                for entry in data:
                    function_name = entry.get("Function Name", "")
                    function_code = entry.get("Function Code", "")
                    function_summary = entry.get("Function Summary", "")
                    suggested_function_name = entry.get("Suggested Function Name", "")

                    summaries.append(function_summary)
                    entries.append({
                        "Function Name": function_name,
                        "Function Code": function_code,
                        "Function Summary": function_summary,
                        "Suggested Function Name": suggested_function_name
                    })

                    if len(summaries) == batch_size:
                        new_function_names = model_inference_batch(summaries)
                        for i in range(len(summaries)):
                            entries[i]["New Suggested Function Name"] = new_function_names[i]
                        processed_results.extend(entries)
                        summaries = []
                        entries = []

                    pbar_json.update(1)

                if summaries:
                    new_function_names = model_inference_batch(summaries)
                    for i in range(len(summaries)):
                        entries[i]["New Suggested Function Name"] = new_function_names[i]
                    processed_results.extend(entries)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed_results, f, indent=4, ensure_ascii=False)

            pbar_apk.update(1)

    print(f"Processed APKs from index {start_idx} to {end_idx}.")

if __name__ == "__main__":
    process_apks(0, 40)