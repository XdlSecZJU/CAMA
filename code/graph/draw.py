import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 主程序
if __name__ == "__main__":
    # 定义自定义的区间
    bins = [0, 1, 4, 7, 10]  # 分别对应0, (0,3], (3,6], (6,10]
    labels = ['0', '1-3', '4-6', '7-10']

    # Before Scores Distribution
    before_counts = pd.Series({
        0: 1008958,
        1: 67987,
        2: 8864,
        3: 5213,
        4: 51956,
        5: 2653,
        6: 7624,
        7: 105415,
        8: 5977,
        9: 1158,
        10: 2440
    })

    # After Scores Distribution
    after_counts = pd.Series({
        0: 1044868,
        1: 722,
        2: 0,
        3: 11,
        4: 32895,
        5: 2,
        6: 758,
        7: 187711,
        8: 1003,
        9: 0,
        10: 275
    })

    # 将分数分箱
    before_binned = pd.cut(before_counts.index, bins=bins, labels=labels, include_lowest=True)
    after_binned = pd.cut(after_counts.index, bins=bins, labels=labels, include_lowest=True)

    # 计算每个区间的计数
    before_grouped = before_counts.groupby(before_binned).sum()
    after_grouped = after_counts.groupby(after_binned).sum()

    # 绘制条形图
    x = np.arange(len(labels))
    width = 0.48

    plt.figure(figsize=(8, 5))
    bar1 = plt.bar(x - width / 2, before_grouped, width, label='Before Rename', color='blue', alpha=0.6)
    bar2 = plt.bar(x + width / 2, after_grouped, width, label='After Rename', color='orange', alpha=0.6)

    # 在柱状图上方显示值
    for rect in bar1 + bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize=13)

    plt.xlim(-0.58, (len(labels) - 0.5) * 1.01)
    plt.ylim(0, max(max(before_grouped), max(after_grouped)) * 1.1)
    plt.xticks(x, labels, fontsize=22)
    plt.yticks(fontsize=22)  # 调整纵坐标刻度标签的字体大小
    plt.xlabel("Maliciousness Score", fontsize=24)
    plt.ylabel("Count", fontsize=24)
    # plt.title("StarChat Maliciousness Score Distribution", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()