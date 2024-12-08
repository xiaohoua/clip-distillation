import torchvision
from torch.nn.functional import one_hot
import torch
import timm
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data.dataset import Dataset
import csv
from PIL import Image
import open_clip
#解决huggingface连接不稳定问题 命令行HF_ENDPOINT=https://hf-mirror.com python xxx.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

SUN_DEFAULT_MEAN = (0.4758, 0.4603, 0.4248)
SUN_DEFAULT_STD = (0.2358, 0.2343, 0.2469)

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
if __name__ == "__main__":

    parser = ArgumentParser()
    # parser.add_argument("model_type", type=str, default="distillation",help="timm")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)#SUN397在蒸馏的时候是224
    parser.add_argument("--is_student", action="store_true")
    parser.add_argument("--model_name", type=str, default="ViT-H-14-378-quickgelu")
    parser.add_argument("--pretrained", type=str, default="../data/models/clip_model/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin",help="如果是教师模型的话需要设置这个参数")#蒸馏需要
    parser.add_argument("--checkpoint_path", type=str, default="SUN/model/distillation_models/ViT-H-14-378-quickgelu/resnet18/checkpoint.pth")
    parser.add_argument("--num_class", type=int, default=397)#normal_train 397 #SUN397在蒸馏的时候是1024这里有疑问？？？
    parser.add_argument("--text_embedding_path", type=str, default="SUN/data/SUN397/ViT-H-14-378-quickgelu/text_embeddings.npy")
    parser.add_argument("--image_embedding_path", type=str, default="SUN/data/SUN397/ViT-H-14-378-quickgelu/image_embedding",help="如果是教师模型需要")
    parser.add_argument("--test_csv_path", type=str, default="SUN/test_data.csv")
    parser.add_argument("--device", type=str, default="cuda:7")
    args = parser.parse_args()

    device = args.device

    transform = Compose([
            Resize(args.image_size),
            CenterCrop(args.image_size),
            ToTensor(),
            Normalize(SUN_DEFAULT_MEAN, SUN_DEFAULT_STD)
        ])
    image_paths, categories, embedding_paths = read_csv(args.test_csv_path)

    dataset = ImageEmbeddingDataset(
        image_paths=image_paths,
        categories = categories,
        embedding_paths=embedding_paths,
        transform=transform
    )
    # print(dataset[0])
    if args.is_student:
        model = timm.create_model(
            model_name=args.model_name,
            num_classes=args.num_class
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
        )

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
                output_embedding = embedding

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
