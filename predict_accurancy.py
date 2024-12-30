import torchvision
import torch
import timm
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
import open_clip
#解决huggingface连接不稳定问题 命令行HF_ENDPOINT=https://hf-mirror.com python xxx.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--is_student", action="store_true")
    parser.add_argument("--model_name", type=str, default="ViT-g-14")
    parser.add_argument("--pretrained", type=str, default="data/models/clip_model/ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model.bin")
    parser.add_argument("--checkpoint_path", type=str, default="data/models/distillation_models/ViT-g-14-laion2B-s34B-b88K/resnet18/checkpoint.pth")
    parser.add_argument("--num_classes", type=int, default=512)#这个叫num_classes不合适，因为对于imagenet数据集1000类要设置成1024
    parser.add_argument("--text_embedding_path", type=str, default="data/imagenet/ViT-g-14-laion2B-s34B-b88K/.npy")
    parser.add_argument("--device", type=str, default="cuda:6")
    args = parser.parse_args()

    device = args.device

    transform = Compose([
            Resize(args.image_size),
            CenterCrop(args.image_size),
            ToTensor(),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    imagenet_val = torchvision.datasets.ImageFolder('/ImageNet/val',transform=transform)
    print(imagenet_val)

    if args.is_student:
        model = timm.create_model(
            model_name=args.model_name,
            num_classes=args.num_classes
        )
        print("model is student")
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
    else:
        print("model is teacher")
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, 
            pretrained=args.pretrained
            # pretrained='/clip_distillation/data/models/ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model.bin'
        )

    model = model.eval().to(device)
    print(model)


    def embedding_to_probs(embedding, text_embedding, temp=100.):
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        logits = embedding @ text_embedding.T
        logits = F.softmax(temp * logits, dim=-1)
        return logits

    text_embeddings = torch.from_numpy(
        # np.load('data/imagenet/text_embeddings.npy')
        np.load(args.text_embedding_path)
    ).to(device).float()

    data_loader = DataLoader(
        dataset=imagenet_val,
        num_workers=16,
        shuffle=False,
        batch_size=args.batch_size
    )

    labels=[]
    predictions=[]

    with torch.no_grad():

        for image,label in tqdm(iter(data_loader)):

            if args.is_student:
                output_embedding = model(image.to(device))
            else:
                output_embedding = model.encode_image(image.to(device))              
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

    acc = MulticlassAccuracy(num_classes=1000,average='macro')
    f1 = MulticlassF1Score(num_classes=1000,average='macro')


    print('Accuracy:\t',acc(labels,predictions).item())

    print('F1-score:\t',f1(labels,predictions).item())
