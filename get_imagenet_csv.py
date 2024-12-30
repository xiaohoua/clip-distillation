from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

dataset = ImageFolder('/ImageNet/train', transform=transform)
embedding_dir = '/clip-distillation/clip-distillation/data/imagenet/ViT-L-14__laion400m_e32/image_embedding'
# 获取图像路径和 logits 对应的列表
image_paths = [item[0] for item in dataset.imgs]
logits_list = [item[1] for item in dataset.samples]

print(image_paths[0])
print(logits_list[0])
import csv

max_classes = 800  # 最大类别数
max_images_per_class = 500  # 每个类别的最大图像数

# 将图像路径和 logits 写入CSV文件
with open('imagepath_category_embedding.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'logits', 'embedding'])

    class_count = {}
    
    for i in range(len(dataset)):
        if i % 10000 == 0:
            print(i)
        image_path, logits = dataset.imgs[i][0], dataset.samples[i][1]
        class_index = logits

        if class_index >= max_classes:
            continue  # 跳过超过最大类别数的类别

        if class_index not in class_count:
            class_count[class_index] = 0

        if class_count[class_index] >= max_images_per_class:
            continue  # 跳过超过最大图像数的类别中的图像

        # 不检查嵌入文件是否存在，直接写入CSV
        embedding_filename = os.path.join(embedding_dir, os.path.splitext(os.path.basename(image_path))[0] + '.npy')

        writer.writerow([image_path, logits, embedding_filename])
        class_count[class_index] += 1

        if len(class_count) == max_classes:
            break  # 达到最大类别数后停止遍历

print("数据已保存到 imagepath_category_embedding.csv 文件中")