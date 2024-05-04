from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import SUN397
import csv
from torch.utils.data import (
    Dataset,
    DataLoader
)
import os
def get_image_paths_and_logits(csv_file):
    image_paths = []
    logits = []

    # 读取CSV文件
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头

        # 遍历CSV文件中的每一行
        for row in reader:
            image_path = row[0]  # 图片路径位于第一列
            logit = row[1]

            image_paths.append(image_path)
            logits.append(logits)

    return image_paths, logits

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

# dataset = ImageFolder('../ImageNet/train', transform=transform)
# embedding_dir = 'data/imagenet/ViT-g-14-laion2B-s34B-b88K/image_embedding'
# dataset = SUN397('SUN/SUN397', transform=transform)
class ImageDataset(Dataset):
    def __init__(self, image_paths, categories, transform=None):
        self.image_paths = image_paths
        self.categories = categories
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        category = self.categories[index]
        return image, category
image_paths, logits = get_image_paths_and_logits("sun397_imagespath_category22.csv")
dataset = ImageDataset(image_paths, logits, transform)
embedding_dir = 'data/SUN397/ViT-H-14-378-quickgelu/image_embedding'
# 获取图像路径和 logits 对应的列表
image_paths = [item[0] for item in dataset.image_paths]
logits_list = [item[1] for item in dataset.categories]

print(image_paths[0])
print(logits_list[0])
import csv

max_classes = 300 # 最大类别数
max_images_per_class = 5000  # 每个类别的最大图像数

# 将图像路径和 logits 写入CSV文件
with open('SUN_imagepath_category_embedding.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'logits', 'embedding'])

    class_count = {}
    class_images_count = {}

    for i in range(len(dataset)):
        if i%10000 == 0:
            print(i)
        image_path, logits = dataset.imgs[i][0], dataset.samples[i][1]
        class_index = logits

        if class_index >= max_classes:
            continue  # 跳过超过最大类别数的类别

        if class_index not in class_count:
            class_count[class_index] = 0

        if class_count[class_index] >= max_images_per_class:
            continue  # 跳过超过最大图像数的类别中的图像

        embedding_filename = os.path.join(embedding_dir, os.path.splitext(os.path.basename(image_path))[0] + '.npy')
        if not os.path.exists(embedding_filename):
            print(embedding_filename + "not exist")
            continue

        writer.writerow([image_path, logits, embedding_filename])
        class_count[class_index] += 1
        class_images_count[class_index] = class_count[class_index]

        if len(class_count) == max_classes:
            break  # 达到最大类别数后停止遍历

print("数据已保存到 SUN_imagepath_category_embedding.csv 文件中")