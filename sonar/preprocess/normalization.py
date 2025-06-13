import numpy as np
import pandas as pd
import os

def normalize_to_range(arr, min_val, max_val, lower_percentile=2, upper_percentile=98):
	"""
	Normalize array to [min_val, max_val] with outlier robustness using percentiles
	"""
	arr = np.array(arr)
	p_min = np.percentile(arr, lower_percentile)
	p_max = np.percentile(arr, upper_percentile)
	if p_max == p_min:
		return np.full_like(arr, (min_val + max_val) / 2)

	# Clip to reduce outlier impact
	arr_clipped = np.clip(arr, p_min, p_max)

	# Normalize based on clipped bounds
	norm_arr = min_val + (arr_clipped - p_min) / (p_max - p_min) * (max_val - min_val)
	return norm_arr

def hbo_normalize(file_path):
     z_score_normalization(file_path)
    #  min_max_normalization(file_path)
     
def min_max_normalization(file_path):
    # 获取文件所在目录和文件名（不带扩展名）
    folder_path = os.path.dirname(file_path)
    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    normalized_hbo_path = os.path.join(folder_path, f"{file_name_no_ext}_normalized.csv")

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 对每一列进行归一化（缩放到[-1, 1]），但跳过 "time" 列
    for col in df.columns:
        if col != "time":  # 排除 "time" 列
            df[col] = 2 * ((df[col] - df[col].min()) / (df[col].max() - df[col].min())) - 1

    # 保存归一化后的数据到新CSV文件
    df.to_csv(normalized_hbo_path, index=False)
    print(f"归一化完成，结果已保存至: {normalized_hbo_path}")

def z_score_normalization(file_path):
	# 获取文件所在目录和文件名（不带扩展名）
	folder_path = os.path.dirname(file_path)
	file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
	normalized_hbo_path = os.path.join(folder_path, f"{file_name_no_ext}_zscored.csv")

	# 读取CSV文件
	df = pd.read_csv(file_path)

	# 对每一列进行z-score标准化（排除"time"列）
	for col in df.columns:
		if col != "time":  # Skip the "time" column
			mean = df[col].mean()
			std = df[col].std()
			df[col] = (df[col] - mean) / std

	# 保存标准化后的数据到新CSV文件
	df.to_csv(normalized_hbo_path, index=False)
	print(f"标准化完成，结果已保存至: {normalized_hbo_path}")

if __name__ == "__main__":
	# 使用示例
	file_path = "res/trainingcamp/HC1-hbo.csv"
	hbo_normalize(file_path)
