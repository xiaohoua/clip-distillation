# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import timm
import open_clip
import PIL.Image
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
import csv
from torch.utils.tensorboard import SummaryWriter
from vision_transformer_model import VisionTransformer
import pdb
from torchvision.transforms import InterpolationMode
import pandas as pd
import torchvision
#解决huggingface连接不稳定问题 命令行HF_ENDPOINT=https://hf-mirror.com python xxx.py
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

SUN_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)#同openAI
SUN_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


class SUNDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        # 使用 SUN397 来加载图像
        self.dataset = torchvision.datasets.SUN397(root=root_dir, transform=transform, download=False)
        self.transform = transform
        
        # 读取CSV文件并创建映射表
        self.csv_data = pd.read_csv(csv_file)
        embeddings_base_path = os.path.join(root_dir, 'data', 'SUN397')
        
        # 创建映射表时，确保路径是绝对路径
        self.image_to_embedding = {
            os.path.basename(row['Image Path']): os.path.join(embeddings_base_path, row['Embedding Path'])
            for idx, row in self.csv_data.iterrows()
        }

        # 过滤图像路径和标签，只保留有embedding的
        filtered_image_files = []
        filtered_labels = []
        for img_path, label in zip(self.dataset._image_files, self.dataset._labels):
            if os.path.basename(img_path) in self.image_to_embedding:
                filtered_image_files.append(img_path)
                filtered_labels.append(label)

        self.image_files = filtered_image_files
        self.labels = filtered_labels

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
        embedding_path = self.image_to_embedding[base_name]
        
        # 加载embedding
        embedding = np.load(embedding_path)
        
        return image, label, embedding

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, default="timm",help="timm,Write_by_hand")
    parser.add_argument("model_name", type=str)
    # parser.add_argument("images_folder", type=str)
    # parser.add_argument("embeddings_folder", type=str)
    parser.add_argument("csv_path", type=str)
    parser.add_argument("output_dir", type=str, help = "模型的checkpoints保存的位置")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dim", type=int, default=1024, help="Dimension of output embedding.  Must match the embeddings generated.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--momentum", type=float, default=0.)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sdg"])
    parser.add_argument("--criterion", type=str, default="mse", choices=["mse", "l1", "huber"])
    parser.add_argument("--use_asp", action="store_true")
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--use_qat", action="store_true")
    #蒸馏手写ViT

    parser.add_argument("--vit_size", type=int, default=224)#vit_size=224 vit_patch_sz就要等于4，或者vit_size=112 vit_patch_sz=8
    parser.add_argument("--vit_patch_sz", type=int, default=4)
    parser.add_argument("--vit_layers", type=int, default=3)
    parser.add_argument("--vit_heads", type=int, default=8)
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed existing checkpoint file at {checkpoint_path}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, "args.json")
    print(f"Running with args {args_dict}")
    print(f"Writing args to {args_path}...")
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=2)
        

    if args.criterion == "mse":
        criterion = F.mse_loss
    elif args.criterion == "l1":
        criterion = F.l1_loss
    elif args.criterion == "huber":
        criterion = F.huber_loss
    else:
        raise RuntimeError(f"Unsupported criterion {args.criterion}")

    if args.use_qat:
        from pytorch_quantization import quant_modules
        # use QAT monkey-patching
        print("Initializing quantization aware training (QAT)")
        quant_modules.initialize()
    #model
    if(args.model_type == "timm"):
        model = timm.create_model(
            model_name=args.model_name,
            pretrained=args.pretrained,
            num_classes=args.output_dim,
            pretrained_cfg_overlay=dict(file=args.pretrained_path),
            features_only = False,
            # out_indices = [3,4]
        )
    elif args.model_type == "Write_by_hand":
        model = VisionTransformer(
            input_resolution=args.vit_size,
            patch_size=args.vit_patch_sz,
            width=args.output_dim,  # Embed dim
            layers=args.vit_layers,
            heads=args.vit_heads,
        )
    
    
    model = model.to(args.device)

    # Setup optimizer
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
    print("=======================optimizer==============")
    transform = Compose([
            Resize(size=args.image_size, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            # Resize(args.image_size),
            CenterCrop(args.image_size),
            ToTensor(),
            Normalize(SUN_DEFAULT_MEAN, SUN_DEFAULT_STD)
        ])


    dataset = SUNDataset('/clip-distillation/clip-distillation/SUN', csv_file=args.csv_path, transform=transform)
    print("=======================dataset==============")
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        batch_size=args.batch_size
    )

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(checkpoint_path)
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

    writer_path = os.path.join(args.output_dir, "log")
    writer = SummaryWriter(writer_path)

    model = model.train()

    print("=======================model.train==============")
    print(start_epoch)
    if args.use_asp:
        from apex.contrib.sparsity import ASP
        ASP.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d], allow_recompute_mask=False, allow_permutation=False)
        ASP.init_optimizer_for_pruning(optimizer)
        ASP.compute_sparse_masks()
        print(f"Pruned model for 2:4 sparse weights using ASP")

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.
        for image, _, embedding in tqdm.tqdm(iter(data_loader)):
            image = image.to(args.device)
            embedding = embedding.to(args.device)
            
            optimizer.zero_grad()
            output_embedding = model(image)

            loss = criterion(output_embedding, embedding)

            loss.backward()
            # break
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