import torch
import timm
from timm.models import vision_transformer
import numpy as np
import argparse
import glob
import os
from torch.utils.data import (
    Dataset,
    DataLoader
)
from PIL import Image
import tqdm
import torch.nn.functional as F
import json
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
    CenterCrop
)
from torch.utils.tensorboard import SummaryWriter
import csv
#解决huggingface连接不稳定问题 命令行HF_ENDPOINT=https://hf-mirror.com python xxx.py
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        image_paths = []
        categories = []
        for row in reader:
            image_paths.append(row[0])
            categories.append(int(row[1]))
        return image_paths, categories

class ImageDataset(Dataset):
    def __init__(self, image_paths, categories, transform=None):
        self.image_paths = image_paths
        self.categories = categories
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        category = self.categories[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, category

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("model_type", type=str, default="timm")
    parser.add_argument("images_folder", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--class_num", type=int, default=512, help="Dimension of output embedding.  Must match the embeddings generated.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--pretrained_clip", default="/clip_distillation/data/models/ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model.bin")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--momentum", type=float, default=0.)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sdg"])
    parser.add_argument("--criterion", type=str, default="mse", choices=["mse", "l1", "huber"])
    parser.add_argument("--use_asp", action="store_true")
    parser.add_argument("--init_checkpoint", type=str, default=None)
    # parser.add_argument("--use_qat", action="store_true")
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #model
    if(args.model_type == "timm"):
        model = timm.create_model(
            model_name=args.model_name,
            pretrained=args.pretrained,
            num_classes=args.class_num
        )
    elif args.model_type == "clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, 
        pretrained=args.pretrained_clip
        )
    elif args.model_type == "ViT":
        model = vision_transformer.__dict__[args.model_name](
                        pretrained=args.pretrained,
                        num_classes=args.class_num
                        )

    model = model.to(args.device)
    #criterion
    if args.criterion == "mse":
        criterion = F.mse_loss
    elif args.criterion == "l1":
        criterion = F.l1_loss
    elif args.criterion == "huber":
        criterion = F.huber_loss
    else:
        raise RuntimeError(f"Unsupported criterion {args.criterion}")
    #optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    #dataset
    csv_file = "imagepath_category.csv"  
    image_paths, categories = read_csv(csv_file)

    transform = Compose([
        Resize(args.image_size),
        CenterCrop(args.image_size),
        ToTensor(),
        Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    dataset = ImageDataset(
        image_paths=image_paths,
        categories=categories,
        transform=transform
    )
    #dataloader
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        batch_size=args.batch_size
    )
    #tensorboard
    writer_path = os.path.join(args.output_dir, "log")
    writer = SummaryWriter(writer_path)
    #权重稀疏化
    if args.use_asp:
        from apex.contrib.sparsity import ASP
        ASP.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d], allow_recompute_mask=False, allow_permutation=False)
        ASP.init_optimizer_for_pruning(optimizer)
        ASP.compute_sparse_masks()
        print(f"Pruned model for 2:4 sparse weights using ASP")
    #断点重新训练模型
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1 # pick up on previous epoch
    elif args.init_checkpoint is not None and os.path.exists(args.init_checkpoint):
        checkpoint = torch.load(args.init_checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = 0  # don't use start checkpoints epoch
    else:
        start_epoch = 0
    
    batch_size=args.batch_size
    
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.
        for image, categories in tqdm.tqdm(iter(data_loader)):
            image = image.to(args.device)

            optimizer.zero_grad()
            output = model(image)

            num_samples_in_batch = categories.shape[0]
            valid_indices = torch.arange(num_samples_in_batch)
            category = torch.zeros(num_samples_in_batch, args.class_num)  # 创建全零张量
            category[valid_indices, categories[:num_samples_in_batch]] = 1

            category = category.to(args.device)  
            loss = criterion(output, category)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)

        writer.add_scalar(
            "loss",
            scalar_value=epoch_loss,
            global_step=epoch
        )
        
        print(f"EPOCH: {epoch} - LOSS: {epoch_loss}")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )

            
