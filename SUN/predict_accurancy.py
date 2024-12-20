import torchvision
from torch.nn.functional import one_hot
import torch
import timm
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data.dataset import Dataset
import csv
import pandas as pd
from PIL import Image
# import torchvision.datasets.SUN397
import open_clip
#解决huggingface连接不稳定问题 命令行HF_ENDPOINT=https://hf-mirror.com python xxx.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

SUN_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
SUN_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

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
        return image, category, embedding


class SUNDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        # 使用 SUN397 来加载图像
        self.dataset = torchvision.datasets.SUN397(root=root_dir, transform=transform, download=False)
        self.transform = transform
        
        # 获取所有图像路径和标签
        self.image_files = self.dataset._image_files
        self.labels = self.dataset._labels
        
        # 读取CSV文件并创建映射表
        self.csv_data = pd.read_csv(csv_file)
        # 假设 CSV 文件中的 'Image Path' 是相对于 root_dir 的路径
        # 并且 'Embedding Path' 是相对于 '/clip-distillation/SUN/data/SUN397/' 的路径
        embeddings_base_path = os.path.join(root_dir, 'data', 'SUN397')
        
        # 创建映射表时，确保路径是绝对路径
        self.image_to_embedding = {
            os.path.basename(row['Image Path']): os.path.join(embeddings_base_path, row['Embedding Path'])
            for idx, row in self.csv_data.iterrows()
        }
        
        # 确保所有图像都有对应的embedding路径
        missing_embeddings = [
            str(img_path) for img_path, label in zip(self.image_files, self.labels) 
            if os.path.basename(img_path) not in self.image_to_embedding
        ]
        if missing_embeddings:
            raise ValueError(f"Missing embeddings for images: {missing_embeddings}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        base_name = os.path.basename(image_path)
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # 获取对应的embedding路径
        embedding_path = self.image_to_embedding.get(base_name)
        if embedding_path is None:
            raise ValueError(f"No embedding found for image: {image_path}")
        
        # 加载embedding
        embedding = np.load(embedding_path)
        
        return image, label, embedding

if __name__ == "__main__":

    parser = ArgumentParser()
    # parser.add_argument("model_type", type=str, default="distillation",help="timm")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=378)#SUN397在蒸馏的时候是224
    parser.add_argument("--is_student", action="store_true")
    parser.add_argument("--model_name", type=str, default="ViT-H-14-378-quickgelu")
    parser.add_argument("--pretrained", type=str, default="../data/models/clip_model/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin",help="如果是教师模型的话需要设置这个参数")#蒸馏需要
    parser.add_argument("--checkpoint_path", type=str, default="SUN/model/distillation_models/ViT-H-14-378-quickgelu/resnet18/checkpoint.pth")
    parser.add_argument("--output_dim", type=int, default=1024)
    parser.add_argument("--num_class", type=int, default=397)#normal_train 397 #SUN397在蒸馏的时候是1024这里有疑问？？？
    parser.add_argument("--text_embedding_path", type=str, default="SUN/data/SUN397/ViT-H-14-378-quickgelu/text_embeddings.npy")
    parser.add_argument("--image_embedding_path", type=str, default="SUN/data/SUN397/ViT-H-14-378-quickgelu/image_embedding",help="如果是教师模型需要")
    parser.add_argument("--test_csv_path", type=str, default="SUN/test_data.csv")
    parser.add_argument("--device", type=str, default="cuda:6")
    args = parser.parse_args()

    device = args.device

    transform = Compose([
            Resize(size=args.image_size, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            # Resize(args.image_size),
            CenterCrop(args.image_size),
            ToTensor(),
            Normalize(SUN_DEFAULT_MEAN, SUN_DEFAULT_STD)
        ])
    image_paths, categories, embedding_paths = read_csv(args.test_csv_path)
    #方法一
    # dataset = ImageEmbeddingDataset(
    #     image_paths=image_paths,
    #     categories = categories,
    #     embedding_paths=embedding_paths,
    #     transform=transform
    # )
    #方法二
    # print(transform)
    # dataset = torchvision.datasets.SUN397('/clip-distillation/SUN',transform=transform)
    # # 获取图像路径和对应的标签
    # image_paths = [item[0] for item in dataset.samples]
    # labels = [item[1] for item in dataset.samples]

    # # 打印前5个图像路径及其标签
    # for path, label in zip(image_paths[:5], labels[:5]):
    #     print(f"Path: {path}, Label: {label}")
    # exit()
    dataset = SUNDataset('/clip-distillation/SUN', csv_file=args.test_csv_path, transform=transform)
    if args.is_student:
        model = timm.create_model(
            model_name=args.model_name,
            num_classes=args.output_dim
        )
        print("model is student")
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
    else:
        print("model is teacher")

        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, 
            pretrained=args.pretrained
            #pretrained='/clip_distillation/data/models/ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model.bin'
        )

    model = model.eval().to(device)
    # print(model)


    def embedding_to_probs(embedding, text_embedding, temp=100.):
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        logits = embedding @ text_embedding.T
        logits = F.softmax(temp * logits, dim=-1)
        return logits

    if args.is_student:
        text_embeddings = torch.from_numpy(
            np.load(args.text_embedding_path)
        ).to(device).float()
    else:
        text_embeddings = torch.from_numpy(
            np.load(args.text_embedding_path)
        ).to(device).float()

    data_loader = DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        shuffle=False,
        batch_size=args.batch_size
    )

    labels=[]
    predictions=[]
    
    with torch.no_grad():

        for image,label,embedding in tqdm(iter(data_loader)):
            
            if args.is_student:
                output_embedding = model(image.to(device))
            else:
                # output_embedding = model.encode_image(image.to(device))
                output_embedding = embedding.to(device)

            probs = embedding_to_probs(
                output_embedding,
                text_embeddings
            )
            probs = probs.detach().cpu().numpy()
            for i in range(probs.shape[0]):
                prob = probs[i]
                prob =prob.flatten()
                prob_indices = np.argsort(prob)[::-1]
                predictions.append(prob_indices[0])

            for item in label.numpy().tolist():
                labels.append(item)


    labels=torch.tensor(labels)
    predictions=torch.tensor(predictions)

    from torchmetrics.classification import MulticlassAccuracy,MulticlassF1Score

    acc = MulticlassAccuracy(num_classes=args.num_class,average='macro')
    f1 = MulticlassF1Score(num_classes=args.num_class,average='macro')


    print('Accuracy:\t',acc(labels,predictions).item())

    print('F1-score:\t',f1(labels,predictions).item())
