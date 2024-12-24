import pandas as pd
import random

# 读取原始CSV文件
original_csv = 'SUN.csv'  # 请替换为实际的原始CSV文件名
df = pd.read_csv(original_csv)

# 确保数据按类别排序
df = df.sort_values(by='Category')

# 获取所有不重复的类别
categories = df['Category'].unique()

# 分割训练集和测试集逻辑

# 训练集：包含所有前300个类别的所有图片
train_categories = categories[:300]
train_df = df[df['Category'].isin(train_categories)]
train_csv = 'train_data_1024.csv'
train_df.to_csv(train_csv, index=False)
print(f"训练集数据（包含所有属于前300类别的图片）已保存到 {train_csv}")

# 测试集：每个类别最多50张图片
test_df = pd.DataFrame(columns=df.columns)
for category in categories:
    # 限制测试集中每个类别的图片数量
    category_df = df[df['Category'] == category]
    if len(category_df) > 50:
        # 如果该类别图片数量超过50张，则随机选择50张
        sampled_indices = random.sample(list(category_df.index), 50)
        sampled_df = category_df.loc[sampled_indices]
    else:
        # 如果图片数量不足50张，使用所有图片
        sampled_df = category_df
    test_df = pd.concat([test_df, sampled_df], ignore_index=True)

test_csv = 'test_data_1024.csv'
test_df.to_csv(test_csv, index=False)
print(f"测试集数据（每个类别最多50张图片）已保存到 {test_csv}")