import torchvision
import torch
import timm
import torchvision
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
import open_clip

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--is_student", action="store_true")
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--checkpoint_path", type=str, default="data/models/resnet18/checkpoint.pth")
    parser.add_argument("--num_classes", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
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
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, 
            pretrained=args.pretrained
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
        np.load('imagenet/text_embeddings.npy')
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
