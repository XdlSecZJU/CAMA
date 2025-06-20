import os
import json
import pandas as pd
import androguard.core.bytecodes.apk as apk
import androguard.core.bytecodes.dvm as dvm
from androguard.decompiler.decompiler import DecompilerDAD
from androguard.core.analysis.analysis import Analysis
import tqdm
import re
from vllm import LLM, SamplingParams

# 路径配置
CSV_PATH = "./apk_info_filter.csv"
APK_DIR = "./filter_apks"
OUTPUT_DIR = "./codellama_output"
MODEL_PATH = "./CodeLlama-7b-Instruct-hf"
BATCH_SIZE = 512  # 设置批量大小

# 加载 vLLM 模型
llm = LLM(model=MODEL_PATH, tensor_parallel_size =4, max_model_len=8192, max_num_seqs=512, dtype="half", trust_remote_code=True, gpu_memory_utilization=0.9)
sampling_params = SamplingParams(temperature=0.3, top_k=10, top_p=0.5, max_tokens=256)

# 过滤输出
def filter_output(prompt, output):
    prompt_end_index = output.find(prompt)
    if prompt_end_index != -1:
        output_after_prompt = output[prompt_end_index + len(prompt):].strip()
    else:
        print("Prompt not found in the output.")
        return output
    output_after_prompt = "\n".join([line.strip() for line in output_after_prompt.splitlines() if line.strip()])
    return output_after_prompt.strip()

# 提取函数名
def extract_function_name(name_text):
    match = re.search(r'\S+(?=\()', name_text)
    return match.group(0) if match else name_text

# 解析模型输出
# 解析模型输出
def parse_model_output(output_text):
    summary_match = re.search(r'Function Summary:\s*(.*?)(?=\nSuggested function name:|\nMalicious Score?s?:|$)',
                              output_text, re.IGNORECASE | re.DOTALL)
    name_match = re.search(r'Suggested function name:\s*(.*?)(?=\n|Malicious Score?s?:|$)', output_text, re.IGNORECASE)

    function_summary = summary_match.group(1).strip() if summary_match else ""

    suggested_name = None
    if name_match:
        name_text = name_match.group(1).strip()
        quoted_name = re.search(r'["\'](.*?)["\']', name_text)
        if quoted_name:
            name_text = quoted_name.group(1)
        name_text = extract_function_name(name_text)
        if 1 <= len(name_text) <= 32:
            suggested_name = name_text

    # 兼容大小写 + 允许 "Malicious Score" 和 "Malicious Scores"
    malicious_score = None
    score_match = re.search(r'Malicious Score?s?\s*\(.*?\):\s*(\d+)', output_text, re.IGNORECASE)
    if score_match:
        malicious_score = int(score_match.group(1))
        if malicious_score < 0 or malicious_score > 10:
            malicious_score = None

    # 确保所有项都有值，否则返回 None
    if not function_summary.strip() or not suggested_name or malicious_score is None:
        return None

    return function_summary, suggested_name, malicious_score

# 判断是否跳过方法
def should_skip_method(decompiled_code):
    return len(decompiled_code.strip().split('\n')) <= 4

# 解析 APK，使用 vLLM 进行批量推理
def analyze_apk(apk_path, sha256):
    results = []
    try:
        a = apk.APK(apk_path)
        d = dvm.DalvikVMFormat(a.get_dex())
        dx = Analysis(d)
        decompiler = DecompilerDAD(d, dx)

        functions = []
        prompts = []
        for method in tqdm.tqdm(d.get_methods(), desc=f"Analyzing {sha256}"):
            try:
                decompiled_code = decompiler.get_source_method(method)
                if decompiled_code is None or should_skip_method(decompiled_code):
                    continue

                prompt = (
                    "[INST]\nYou are a cybersecurity expert specializing in reverse engineering and malware analysis. Your task is to analyze a decompiled Android function and generate a structured function summary based on the following aspects :\n"
                    "   1. Function Summary : <Provide a brief, high-level description of what this function does. Summarize its purpose, key operations, and intent.>\n"
                    "   2. Suggested Function Name : <Suggest a clearer, more descriptive function name that accurately represents its behavior.>\n"
                    "   3. Malicious Score(0-10) : <Rate the function's maliciousness on a scale from 0 to 10, where:\n"
                    "       - 0 - Benign : No suspicious activity.\n"
                    "       - 1-3 - Potentially Safe but Risky : Performs sensitive actions but could be legitimate.\n"
                    "       - 4-6 - Suspicious : Uses permissions or techniques common in malware.\n"
                    "       - 7-10 - Highly Malicious : Strong indicators of malware behavior.>\n[/INST]\n"
                    "[FUNC]" + decompiled_code + "[/FUNC]\n"
                )

                functions.append((method.get_name(), decompiled_code))
                prompts.append(prompt)

                # 批量推理
                if len(prompts) >= BATCH_SIZE:
                    outputs = llm.generate(prompts, sampling_params)
                    print("\n")
                    for (func_name, func_code), output in zip(functions, outputs):
                        parsed_result = parse_model_output(output.outputs[0].text)
                        if parsed_result:
                            func_summary, suggested_name, malicious_score = parsed_result
                            results.append({
                                "Function Name": func_name,
                                "Function Code": func_code,
                                "Function Summary": func_summary,
                                "Suggested Function Name": suggested_name,
                                "Malicious Score": malicious_score
                            })
                    functions.clear()
                    prompts.clear()

            except Exception as e:
                print(f"Error analyzing method {method.get_name()}: {e}")

        # 处理剩余的函数
        if prompts:
            outputs = llm.generate(prompts, sampling_params)
            for (func_name, func_code), output in zip(functions, outputs):
                parsed_result = parse_model_output(output.outputs[0].text)
                generated_text = output.outputs[0].text  # 提取生成的文本
                print(generated_text)  # 打印生成的文本
                print("\n")
                if parsed_result:
                    func_summary, suggested_name, malicious_score = parsed_result
                    results.append({
                        "Function Name": func_name,
                        "Function Code": func_code,
                        "Function Summary": func_summary,
                        "Suggested Function Name": suggested_name,
                        "Malicious Score": malicious_score
                    })

    except Exception as e:
        print(f"Failed to process APK {apk_path}: {e}")
        return []

    return results

# 读取 CSV 获取 SHA256 列表
def get_sha256_list(csv_path, start_idx=None, end_idx=None):
    df = pd.read_csv(csv_path)
    if start_idx is not None and end_idx is not None:
        df = df.iloc[start_idx:end_idx]
    return df['sha256'].tolist()

# 遍历目录查找 APK
def find_matching_apks(sha256_list, apk_dir):
    return {sha256: os.path.join(apk_dir, f"{sha256}.apk") for sha256 in sha256_list if os.path.exists(os.path.join(apk_dir, f"{sha256}.apk"))}

# 主函数
def main(start_idx=None, end_idx=None):
    sha256_list = get_sha256_list(CSV_PATH, start_idx, end_idx)
    apk_files = find_matching_apks(sha256_list, APK_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with tqdm.tqdm(total=len(apk_files), desc="Processing APKs") as apk_bar:
        for sha256, apk_path in apk_files.items():
            output_path = os.path.join(OUTPUT_DIR, f"{sha256}.json")
            if os.path.exists(output_path):
                apk_bar.update(1)
                continue
            results = analyze_apk(apk_path, sha256)
            if results:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)
            apk_bar.update(1)

if __name__ == "__main__":
    main(0, 120)