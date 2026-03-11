

import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def analyze_silu_log(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    in_values = []
    
    # 正则表达式匹配 "In:" 后面的浮点数
    # 适配格式: [SiLU _Debug_] In:    0.3505 | ...
    pattern = re.compile(r"In:\s+([-+]?\d*\.\d+|\d+)")

    print(f"正在读取日志文件: {file_path} ...")
    
    with open(file_path, 'r') as f:
        for line in f:
            if "[SiLU Debug]" in line:
                match = pattern.search(line)
                if match:
                    in_values.append(float(match.group(1)))

    if not in_values:
        print("未在日志中找到匹配的数据，请检查日志格式。")
        return

    print(f"成功提取 {len(in_values)} 条数据。")

    # 计算基本统计信息
    in_array = np.array(in_values)
    print(f"统计信息:")
    print(f"  最小值 (Min): {in_array.min():.4f}")
    print(f"  最大值 (Max): {in_array.max():.4f}")
    print(f"  平均值 (Mean): {in_array.mean():.4f}")
    print(f"  中位数 (Median): {np.median(in_array):.4f}")
    print(f"  标准差 (Std): {in_array.std():.4f}")

    # 绘图
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # 绘制直方图和密度曲线
    sns.histplot(in_array, kde=True, color='skyblue', bins=50, edgecolor='black', alpha=0.7)

    # 标注均值线
    plt.axvline(in_array.mean(), color='red', linestyle='--', label=f'Mean: {in_array.mean():.2f}')
    
    plt.title('Distribution of SiLU Input Values (In)', fontsize=15)
    plt.xlabel('Input Value (In)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    
    # 保存图片
    output_img = "silu_in_distribution_norm.png"
    plt.savefig(output_img)
    print(f"分布图已保存至: {output_img}")
    
    plt.show()

if __name__ == "__main__":
    log_path = "/deltadisk/congxiao/code/github/Quamba/debug_silu_norm.log"
    analyze_silu_log(log_path)