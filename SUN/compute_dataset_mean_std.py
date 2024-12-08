import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from time import time
import csv

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        image_paths = []
        categories = []
        embedding_path = []
        for row in reader:
            image_paths.append(row[0])
            categories.append(int(row[1]))
            embedding_path.append(row[2])
        return image_paths, categories, embedding_path

class ImageEmbeddingDataset(Dataset):
    def __init__(self, image_paths, categories, embedding_paths, transform=None):
        self.image_paths = image_paths
        self.embedding_paths = embedding_paths
        self.categories = categories
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        embedding = np.load(self.embedding_paths[index])
        category = self.categories[index]
        return image, embedding, category


N_CHANNELS = 3

image_paths, categories, embedding_paths = read_csv("SUN/SUN_imagepath_category_embedding.csv")

dataset = ImageEmbeddingDataset(
        image_paths=image_paths,
        categories = categories,
        embedding_paths=embedding_paths,
        transform=transforms.ToTensor()
    )
full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())

before = time()
mean = torch.zeros(N_CHANNELS)
std = torch.zeros(N_CHANNELS)
print('==> Computing mean and std..')
for inputs, _labels, embedding in tqdm(full_loader):
    for i in range(N_CHANNELS):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()
mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)

print("time elapsed: ", time()-before)
#tensor([0.4758, 0.4603, 0.4248]) tensor([0.2358, 0.2343, 0.2469])