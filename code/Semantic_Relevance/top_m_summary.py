import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    return tokenizer, model

# 过滤输出
def filter_output(prompt, output):
    # 找到 prompt 在 output 中的位置
    prompt_end_index = output.find(prompt)
    if prompt_end_index != -1:
        # 截取 prompt 之后的内容
        output_after_prompt = output[prompt_end_index + len(prompt):].strip()
    else:
        print("Prompt not found in the output.")
        return output

    # 去除多余的换行符和空格
    output_after_prompt = "\n".join([line.strip() for line in output_after_prompt.splitlines() if line.strip()])

    # 在第一个句号处截断
    first_period_index = output_after_prompt.find(".")
    if first_period_index != -1:
        output_after_prompt = output_after_prompt[:first_period_index + 1]

    # 添加前缀
    final_output = "The application appears to " + output_after_prompt.strip()

    return final_output.strip()

# 与模型交互的函数
def model_inference(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
    inputs = {k: v.to(device) for k, v in inputs.items()}
    attention_mask = inputs["attention_mask"]  # 获取 attention_mask

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,  # 传递 attention_mask
            max_length=32768,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.8,
            no_repeat_ngram_size=3,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id, 
            pad_token_id=tokenizer.eos_token_id
        )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 过滤输出
    output = filter_output(prompt, decoded_output)
    return output

def get_function_summaries(sha256, indices):
    # 读取sha256.json文件
    json_file = f"./codellama_output/{sha256}.json"
    if not os.path.exists(json_file):
        return None
    with open(json_file, "r") as f:
        functions = json.load(f)
    # 根据Index项检索Function Summary
    summaries = []
    for index in indices:
        if index < len(functions):
            summaries.append(functions[index]["Function Summary"])
    return summaries

def generate_prompt(summaries):
    # 生成codellama的输入prompt，包含few-shot示例
    prompt = (
        "Generate new Application Purpose based on the function summary of the application below, each summary should be a complete, detailed sentence that describes the functionality of the application:\n\n"
        + "\n".join(summaries)  # 将summaries列表中的所有字符串元素合并成一个单一的字符串
        + "\nApplication Purpose:"
        + "\nThe application appears to"
    )
    return prompt

def main():
    # 模型路径
    model_path = "./CodeLlama-7b-Instruct-hf"
    tokenizer, model = load_model(model_path)
    
    # 读取存储的top_30_indices.json文件
    indices_file = "./top_30_indices.json"
    with open(indices_file, "r") as f:
        top_30_indices = json.load(f)
    
    # 指定m值
    m = 30
    # 创建结果存储的json文件
    result_file = "E2_codellama_top30_output_0.json"
    results = {}
    
    # 使用 tqdm 创建进度条
    for sha256 in tqdm(top_30_indices.keys(), desc="Processing APKs"):
        indices = top_30_indices[sha256]
        selected_indices = indices[:m]
        summaries = get_function_summaries(sha256, selected_indices)
        if summaries is not None:
            prompt = generate_prompt(summaries)
            # 调用模型获取Application Purpose
            application_purpose = model_inference(tokenizer, model, prompt)
            results[sha256] = application_purpose
    # 将结果存储到json文件中
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()