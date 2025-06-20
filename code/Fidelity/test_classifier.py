import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # 导入 tqdm 用于进度条

# 类别映射
custom_class_mapping = {
    'Backdoor': 0, 'Riskware': 1, 'PUA': 2, 
    'Adware': 3, 'Scareware': 4, 'Trojan': 5
}

# 1. 读取 APK 数据
csv_path = "./apk_info_filter.csv"
df = pd.read_csv(csv_path)

# 2. 读取 JSON 并提取文本特征和数值特征
apk_text_features, apk_num_features, apk_labels, apk_sha256 = [], [], [], []
json_folder = "./codellama_output_replace_remove8"  # JSON 目录

for sha256, category in tqdm(zip(df["sha256"], df["category"]), total=len(df), desc="Processing JSON"):
    json_path = os.path.join(json_folder, f"{sha256}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            functions = json.load(f)
        
        # 提取文本信息 (Suggested Function Name + Function Summary)
        text_feature = " ".join([
            f"{func['Suggested Function Name']} {func['Function Summary']}" 
            for func in functions
        ])
        apk_text_features.append(text_feature)
        
        # 提取数值特征 (Malicious Score 平均值)
        mal_scores = [func['Malicious Score'] for func in functions if 'Malicious Score' in func]
        avg_mal_score = np.mean(mal_scores) if mal_scores else 0  # 计算平均恶意得分
        apk_num_features.append(avg_mal_score)

        # 记录分类标签
        apk_labels.append(custom_class_mapping.get(category, -1))
        apk_sha256.append(sha256)  # 记录 sha256 值

# 过滤无效数据
valid_indices = [i for i in range(len(apk_labels)) if apk_labels[i] != -1]
apk_text_features = [apk_text_features[i] for i in valid_indices]
apk_num_features = np.array([apk_num_features[i] for i in valid_indices]).reshape(-1, 1)
apk_labels = np.array([apk_labels[i] for i in valid_indices])
apk_sha256 = [apk_sha256[i] for i in valid_indices]  # 过滤无效数据后的 sha256 列表

# 3. 文本特征提取（TF-IDF）
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
text_features_tfidf = vectorizer.fit_transform(apk_text_features).toarray()

# 4. 数值特征标准化
scaler = StandardScaler()
num_features_scaled = scaler.fit_transform(apk_num_features)

# 5. 拼接文本特征 + 数值特征
X = np.hstack((text_features_tfidf, num_features_scaled))

# 6. 加载模型
model_path = "./lgbm_classifier_codellama_replace.pkl"
lgbm = joblib.load(model_path)

# 7. 预测
y_pred = lgbm.predict(X)
y_pred_proba = lgbm.predict_proba(X)  # 获取预测概率

# 8. 输出结果
# 8. 输出结果
results = []
for sha256, true_label, pred_label, proba in zip(apk_sha256, apk_labels, y_pred, y_pred_proba):
    true_label_prob = proba[true_label]  # 获取真实类别上的概率
    pred_label_prob = proba[pred_label]  # 获取预测类别上的概率
    results.append({
        'sha256': sha256,
        '真实类别': true_label,
        '预测类别': pred_label,
        '概率列表': proba.tolist(),  # 将概率数组转换为列表
        '真实类别概率': true_label_prob,  # 添加真实类别上的概率
        '预测类别概率': pred_label_prob  # 添加预测类别上的概率
    })

# 将结果保存到 CSV 文件
results_df = pd.DataFrame(results)
results_df.to_csv("./E3_codellama_replace_remove8.csv", index=False, encoding="utf-8")
print("结果已保存到")