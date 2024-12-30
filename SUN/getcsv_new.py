import os
import csv

root_folder = 'SUN397'  # SUN397 文件夹的路径
csv_file = 'SUN.csv'  # 保存结果的CSV文件名
embedding_base_folder = '/clip-distillation/clip-distillation/SUN/data/SUN397/ViT-L-14__laion400m_e32/image_embedding'  # 嵌入文件夹路径

# 创建类别到数字的映射字典
category_to_number = {}
number = 0

# 打开CSV文件并准备写入数据
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Path', 'Category', 'Embedding Path'])  # 写入表头

    # 遍历根目录下的所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # 过滤掉非图片文件
        filenames = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        
        for filename in filenames:
            # 构建图片完整路径和其对应的类别路径
            image_path = os.path.join(dirpath, filename)  # 图像路径是相对于 root_folder 的
            category_path = dirpath[len(root_folder)+1:].replace(os.sep, '/')  # 类别路径转换为以"/"分隔
            
            # 将类别路径转换为数字
            if category_path not in category_to_number:
                category_to_number[category_path] = number
                number += 1
            
            # 构建嵌入文件路径
            image_id = os.path.splitext(filename)[0]
            embedding_path = os.path.join(embedding_base_folder, f"{image_id}.npy")
            
            # 写入图片路径、类别数字和嵌入文件路径
            writer.writerow([image_path, category_to_number[category_path], embedding_path])