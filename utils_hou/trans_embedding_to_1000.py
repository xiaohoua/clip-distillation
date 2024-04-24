import os
import numpy as np
import time
from sklearn.random_projection import GaussianRandomProjection

#升维会报warning
import warnings
warnings.filterwarnings("ignore")
# 源目录与目标目录
src_dir = 'data/imagenet/image_embedding'
dst_dir = 'data/imagenet/image_embedding_1000'

# 确保目标目录存在
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# 创建随机投影器，设置目标维度为1000
rp = GaussianRandomProjection(n_components=1000)

# 遍历源目录中的 .npy 文件
file_count = 0
print_frequency = 100
total_files = len([f for f in os.listdir(src_dir) if f.endswith('.npy')])
progress_bar_length = 70

start_time = time.time()  # 记录开始处理时间

for filename in os.listdir(src_dir):
    if filename.endswith('.npy'):
        src_filepath = os.path.join(src_dir, filename)
        dst_filepath = os.path.join(dst_dir, filename)

        # 检查目标目录下是否存在已处理的对应文件，存在则跳过本次处理
        if os.path.isfile(dst_filepath):
            print(f"> File {filename} already exists in destination directory. Skipping.")
            continue

        file_count += 1

        # 加载嵌入向量
        embeddings = np.load(src_filepath)

        # 处理不同情况：
        #   - 如果单个向量，将其扩展为一维数组
        #   - 如果多个向量组成的二维数组，直接使用
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # 升维至1000维
        embeddings_1000d = rp.fit_transform(embeddings)

        # 保存升维后的嵌入向量
        np.save(dst_filepath, embeddings_1000d)

        # 每处理完print_frequency个文件打印一次进度信息
        if file_count % print_frequency == 0 or file_count == total_files:
            elapsed_time = time.time() - start_time  # 计算已过去的时间
            progress = int(file_count / total_files * progress_bar_length)
            remaining = progress_bar_length - progress
            progress_bar = '>' * progress + '-' * remaining
            estimated_remaining_time = (elapsed_time / file_count) * (total_files - file_count) if file_count > 0 else 0
            print(f"Processing files... [{progress_bar}] {file_count}/{total_files} ({file_count / total_files * 100:.1f}% completed) "
                  f"Elapsed: {elapsed_time:.1f} s, Remaining: {estimated_remaining_time:.1f} s")

print("\nAll files processed successfully.")