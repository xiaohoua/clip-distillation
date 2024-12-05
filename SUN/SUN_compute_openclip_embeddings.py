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

import open_clip
import glob
import os
import PIL.Image
import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from open_clip.pretrained import _PRETRAINED
import csv
#解决huggingface连接不稳定问题 命令行HF_ENDPOINT=https://hf-mirror.com python xxx.py
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
def get_embedding_path(embedding_folder, image_id):
        return os.path.join(embedding_folder, image_id + ".npy")

def get_image_and_embedding_paths(csv_file, embedding_folder):
    image_paths = []
    embedding_paths = []

    # 读取CSV文件
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头

        # 遍历CSV文件中的每一行
        for row in reader:
            image_path = row[0]  # 图片路径位于第一列
            image_id = os.path.basename(image_path).split('.')[0]  # 提取图像ID
            embedding_path = get_embedding_path(embedding_folder, image_id)  # 生成嵌入文件路径

            image_paths.append(image_path)
            embedding_paths.append(embedding_path)

    return image_paths, embedding_paths
if __name__ == "__main__":

    parser = ArgumentParser()
    # parser.add_argument("input_folder", type=str)
    parser.add_argument("--output_folder", type=str, default="data/SUN397/ViT-H-14-378-quickgelu/image_embedding")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="ViT-H-14-378-quickgelu")
    parser.add_argument("--pretrained", type=str, default="data/models/clip_model/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin")
    parser.add_argument("--device", type=str, default="cuda:3")
    args = parser.parse_args()
    
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

   
    def get_image_id_from_path(image_path):
        return os.path.basename(image_path).split('.')[0]
    
    image_paths, embedding_paths = get_image_and_embedding_paths("sun397_imagespath_category.csv", "data/SUN397/ViT-H-14-378-quickgelu/image_embedding")
    old_image_paths = image_paths
    image_paths = [
        image_path for image_path in image_paths
        if not os.path.exists(
            get_embedding_path(
                args.output_folder, 
                get_image_id_from_path(image_path)
            )
        )
    ]
    print(get_image_id_from_path(image_paths[0]))
    print(old_image_paths[0])
    print(image_paths[0])
    print(embedding_paths[0])
    num_skip = len(old_image_paths) - len(image_paths)
    if num_skip == len(old_image_paths):
        print(f"All embeddings already computed. Nothing left to do.")
        exit()
    elif num_skip > 0:
        print(f"Skipping computation of {num_skip} embeddings because they already exist.")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, 
        pretrained=args.pretrained
    )
    model.to(device)
    class ImageDataset(Dataset):
        def __init__(self, image_paths, preproc):
            self.image_paths = image_paths
            self.preproc = preproc

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            image = PIL.Image.open(self.image_paths[index])
            image = self.preproc(image)
            return index, image

    dataset = ImageDataset(image_paths, preprocess)
    print(dataset[146])
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )

    print(f"Computing embeddings for {len(image_paths)} images...")
    with torch.no_grad():
        for indices, images in tqdm.tqdm(iter(data_loader)):
            images = images.to(device)
            count = len(indices)
            embeddings = model.encode_image(images)
            for idx in range(count):
                image_path_idx = int(indices[idx])
                image_path = dataset.image_paths[image_path_idx]
                embedding_path = get_embedding_path(
                    args.output_folder,
                    get_image_id_from_path(image_path)
                )
                embedding = embeddings[idx].detach().cpu().numpy()
                np.save(embedding_path, embedding)


