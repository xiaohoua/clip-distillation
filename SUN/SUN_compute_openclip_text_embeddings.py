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
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from open_clip.pretrained import _PRETRAINED
#解决huggingface连接不稳定问题 命令行HF_ENDPOINT=https://hf-mirror.com python xxx.py
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class CustomCLIP(nn.Module):
    def __init__(self, model_name, pretrained, output_dim=397):
        super(CustomCLIP, self).__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.linear = nn.Linear(self.model.visual.output_dim, output_dim)

    def forward(self, images):
        # 获取模型的原始输出
        with torch.no_grad():
            outputs = self.model.encode_text(images)
        # 通过线性层改变输出维度
        embeddings = self.linear(outputs)
        return embeddings

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("text_prompts_file", type=str, default="SUN_text_prompts.txt")
    parser.add_argument("output_path", type=str, default="data/SUN397/ViT-H-14-378-quickgelu/text_embeddings.npy")
    parser.add_argument("--model_name", type=str, default="ViT-H-14-378-quickgelu")
    parser.add_argument("--pretrained", type=str, default="../data/models/clip_model/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin")
    parser.add_argument("--output_dim", type=int, default=397)#SUN397 有397个类
    args = parser.parse_args()

    with open(args.text_prompts_file, 'r') as f:
        text_prompts = f.readlines()
        text_prompts = [tp.strip() for tp in text_prompts]

    print(f"Found the following {len(text_prompts)} text prompts in {args.text_prompts_file}")
    print(text_prompts)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, 
        pretrained=args.pretrained
    )
    # model = CustomCLIP(args.model_name, args.pretrained, args.output_dim)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    with torch.no_grad():
        text = tokenizer(text_prompts)
        # text_embeddings = model(text)
        text_embeddings = model.encode_text(text)
        text_embeddings = text_embeddings.detach().cpu().numpy()

        print(f"Saving text embeddings to {args.output_path}")
        np.save(args.output_path, text_embeddings)
        print(text_embeddings.shape)
    