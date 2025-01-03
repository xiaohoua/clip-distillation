from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

dataset = ImageFolder('../ImageNet/train', transform=transform)

# 获取图像路径和 logits 对应的列表
image_paths = [item[0] for item in dataset.imgs]
logits_list = [item[1] for item in dataset.samples]

print(image_paths[0])
print(logits_list[0])
import csv

max_classes = 800  # 最大类别数
max_images_per_class = 500  # 每个类别的最大图像数

# 将图像路径和 logits 写入CSV文件
with open('imagepath_category.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_path', 'logits'])

    class_count = {}
    class_images_count = {}

    for i in range(len(dataset)):
        image_path, logits = dataset.imgs[i][0], dataset.samples[i][1]
        class_index = logits

        if class_index >= max_classes:
            continue  # 跳过超过最大类别数的类别

        if class_index not in class_count:
            class_count[class_index] = 0

        if class_count[class_index] >= max_images_per_class:
            print("===============jump========")
            continue  # 跳过超过最大图像数的类别中的图像

        writer.writerow([image_path, logits])
        class_count[class_index] += 1
        class_images_count[class_index] = class_count[class_index]

        if len(class_count) == max_classes:
            break  # 达到最大类别数后停止遍历

print("数据已保存到 imagepath_category.csv 文件中")