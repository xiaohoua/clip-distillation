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
from vision_transformer_model import VisionTransformer
#解决huggingface连接不稳定问题 命令行HF_ENDPOINT=https://hf-mirror.com python xxx.py
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

SUN_DEFAULT_MEAN = (0.4758, 0.4603, 0.4248)
SUN_DEFAULT_STD = (0.2358, 0.2343, 0.2469)

def get_image_id_from_path(image_path):
    return os.path.basename(image_path).split('.')[0]

def get_embedding_path(embedding_folder, image_id):
    return os.path.join(embedding_folder, image_id + ".npy")


# def find_images(images_folder: str):
        # image_paths = []
        # train_folder = os.path.join(images_folder, 'train')

        # # 获取train文件夹下的类别文件夹
        # class_folders = sorted(glob.glob(os.path.join(train_folder, "*")))
        # selected_class_folders = class_folders[:800]

        # for class_folder in selected_class_folders:
        #     # 获取类别文件夹下的图像文件路径
        #     class_images = sorted(glob.glob(os.path.join(class_folder, "*.JPEG")))[:500]
        #     image_paths.extend(class_images)

        # return image_paths
def find_images(images_folder: str):
    image_paths = []
    # 遍历根目录下的所有子文件夹
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            # 过滤掉非图片文件
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                # 构建图片完整路径
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    return image_paths


class ImageEmbeddingDataset(Dataset):
    def __init__(self, image_paths, embedding_paths, transform=None):
        self.image_paths = image_paths
        self.embedding_paths = embedding_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        embedding = np.load(self.embedding_paths[index])
        return image, embedding



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, default="timm",help="timm,Write_by_hand")
    parser.add_argument("model_name", type=str)
    parser.add_argument("images_folder", type=str)
    parser.add_argument("embeddings_folder", type=str)
    parser.add_argument("output_dir", type=str, help = "模型的checkpoints保存的位置")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dim", type=int, default=397, help="Dimension of output embedding.  Must match the embeddings generated.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--pretrained", action="store_true")
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
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, "args.json")
    print(f"Running with args {args_dict}")
    print(f"Writing args to {args_path}...")
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=2)
        
    all_image_paths = find_images(args.images_folder)
    print(f"Found {len(all_image_paths)} in the folder {args.images_folder}")

    image_paths = []
    embedding_paths = []

    for image_path in all_image_paths:
        image_id = get_image_id_from_path(image_path)
        embedding_path = get_embedding_path(args.embeddings_folder, image_id)
        if os.path.exists(embedding_path):
            image_paths.append(image_path)
            embedding_paths.append(embedding_path)
    
    print(f"Found embeddings for {len(embedding_paths)} out of {len(all_image_paths)} images.")

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
            num_classes=args.output_dim
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
        Resize(args.image_size),
        CenterCrop(args.image_size),
        ToTensor(),
        Normalize(SUN_DEFAULT_MEAN, SUN_DEFAULT_STD)
    ])

    dataset = ImageEmbeddingDataset(
        image_paths=image_paths,
        embedding_paths=embedding_paths,
        transform=transform
    )
    # print(image_paths[0])
    # print(embedding_paths[0])

    # print(image_paths[22:35])
    # print(embedding_paths[22:35])
    
    print("=======================dataset==============")
    print(dataset[0])
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        batch_size=args.batch_size
    )

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
    print("=======================checkpoint_path==============")
    writer_path = os.path.join(args.output_dir, "log")
    writer = SummaryWriter(writer_path)

    model = model.train()

    print("=======================model.train==============")
    print("start_epoch")
    print(start_epoch)
    if args.use_asp:
        from apex.contrib.sparsity import ASP
        ASP.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d], allow_recompute_mask=False, allow_permutation=False)
        ASP.init_optimizer_for_pruning(optimizer)
        ASP.compute_sparse_masks()
        print(f"Pruned model for 2:4 sparse weights using ASP")

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.

        for image, embedding in tqdm.tqdm(iter(data_loader)):
            image = image.to(args.device)
            embedding = embedding.to(args.device)
            
            optimizer.zero_grad()
            output_embedding = model(image)
            # print(output_embedding.shape)
            
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