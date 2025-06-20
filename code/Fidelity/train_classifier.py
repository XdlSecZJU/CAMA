import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # 处理类别不均衡
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # 用于保存模型

# 类别映射
custom_class_mapping = {
    'Backdoor': 0, 'Riskware': 1, 'PUA': 2, 
    'Adware': 3, 'Scareware': 4, 'Trojan': 5
}

# 1. 读取 APK 数据
csv_path = "./apk_info_filter.csv"
df = pd.read_csv(csv_path)

# 2. 读取 JSON 并提取文本特征和数值特征
apk_text_features, apk_num_features, apk_labels = [], [], []
json_folder = "./starchat_output_replace"  # JSON 目录

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

# 过滤无效数据
valid_indices = [i for i in range(len(apk_labels)) if apk_labels[i] != -1]
apk_text_features = [apk_text_features[i] for i in valid_indices]
apk_num_features = np.array([apk_num_features[i] for i in valid_indices]).reshape(-1, 1)
apk_labels = np.array([apk_labels[i] for i in valid_indices])

# 3. 文本特征提取（TF-IDF）
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
text_features_tfidf = vectorizer.fit_transform(apk_text_features).toarray()

# 4. 数值特征标准化
scaler = StandardScaler()
num_features_scaled = scaler.fit_transform(apk_num_features)

# 5. 拼接文本特征 + 数值特征
X = np.hstack((text_features_tfidf, num_features_scaled))

# 6. 处理类别不均衡
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, apk_labels)

# 7. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=None, random_state=42)

# 8. 训练 LightGBM
lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, class_weight="balanced")
lgbm.fit(X_train, y_train)

# 9. 评估 LightGBM
y_pred_lgbm = lgbm.predict(X_test)
acc_lgbm = accuracy_score(y_test, y_pred_lgbm)

print(f"LightGBM 准确率: {acc_lgbm:.4f}")
print("LightGBM 分类报告:\n", classification_report(y_test, y_pred_lgbm))

# 10. 保存模型
model_path = "./lgbm_classifier_starchat_replace.pkl"
joblib.dump(lgbm, model_path)
print(f"模型已保存到 {model_path}")