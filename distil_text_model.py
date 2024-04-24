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
#解决huggingface连接不稳定问题 命令行HF_ENDPOINT=https://hf-mirror.com python xxx.py
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def get_text_id_from_path(text_path):
    return os.path.basename(text_path).split('.')[0]


def get_text_embedding_path(text_embedding_folder, text_id):
    return os.path.join(text_embedding_folder, text_id + ".npy")