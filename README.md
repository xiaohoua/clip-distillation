# CLIP Knowledge Distillation
基于英伟达的知识蒸馏框架(https://github.com/NVIDIA-AI-IOT/clip-distillation)实现一套新的用于广义零样本的知识蒸馏方法。

原有框架可以对相关数据集进行蒸馏，这里我先使用更知名的大型数据集ImageNet进行实验，后续补充零样本学习领域的数据集的相关实验。


## Instructions

1. [Step 1 - Search and download relevant unlabeled images to use for distillation](#step-1)
2. [Step 2 - Pre-compute OpenCLIP embeddings](#step-2)
3. [Step 3 - Train the student CNN model to mimic the OpenCLIP model](#step-3)
4. [Step 4 - Run inference using the distilled model](#step-4)
5. [Step 5 (advanced) - Train a student model with structured sparsity](#step-5)
6. [Step 6 (advanced) - Train a student with Quantization aware training and INT8 precision](#step-6)
7. [Next Steps](#next-steps)

<a name="step-1"></a>

## Step 1 - Search and download images with CLIP filtering

### Search for relevant image URLs in the LAION database using CLIP filtering

首先生成图片数据集的相关prompts(对于使用的ImageNet来说就是图片路径于图片类别相关的名称)。以供后续使用。

使用``utils_hou/build_imagenet_mapping.py``生成prompts，并将生成的prompts存放在 ``data/imagenet/text_prompts.txt`` 


## Step 2 - Compute OpenCLIP embeddings

在蒸馏过程中，我们图像数据将用作我们的教师和学生模型的输入。但是，在蒸馏期间执行教师模型可能很慢。

为了加快这个过程，我们将预先计算我们的教师模型的输出，这样我们就不需要在训练期间执行教师模型。也就是提前将图片数据喂给教师模型，提前得到教师模型的输出（embeddings）。然后再将embeddings喂给学生模型，这样就能大大提高学生模型的学习速度。

To do this, call the ``compute_openclip_embeddings.py`` script as follows,

```bash
python3 compute_openclip_embeddings.py \
    /data/dataset/ImageNet/extract \
    data/imagenet/ViT-g-14-laion2B-s34B-b88K/image_embedding \
    --batch_size 16 \
    --num_workers 8 \
    --model_name ViT-B-32 \
    --pretrained laion2b_s34b_b79k
```

这将把输出嵌入写入文件夹``data/imagenet/ViT-g-14-laion2B-s34B-b88K/image_embedding``，文件名与图像文件名匹配，除了文件扩展名。

因为大模型普遍都有CLIP架构，也就是同时具有图片编码器和文本编码器，上述方法可以获取图像嵌入，我们也可以用类似的方法获取文本嵌入，具体代码见``compute_openclip_text_embeddings.py``

<a name="step-3"></a>

## Step 3 - Train the student CNN model to mimic the OpenCLIP model

目前实现了两种蒸馏方法，一种是以图像嵌入为目标，尽可能让学生模型和教师模型生成的图像嵌入类似，从而让学生模型的性能逼近教师模型，具体代码见
 ``distil_model_embeddings.py`` 

```bash
python3 distil_model_embeddings.py \
    --model_name resnet18 \
    --images_folder /data/dataset/ImageNet/extract \
    --embeddings_folder data/imagenet/ViT-g-14-laion2B-s34B-b88K/image_embedding \
    --text_embedding_path data/imagenet/ViT-g-14-laion2B-s34B-b88K/.npy \
    --output_dir data/models/distillation_models/ViT-g-14-laion2B-s34B-b88K/resnet18 \
```

第二种是把图像嵌入和图像类别同时作为蒸馏目标，具体代码见``distil_model_embeddings_label.py``

```bash
python3 distil_model_embeddings_label.py \
    --model_name resnet18 \
    --images_folder /data/dataset/ImageNet/extract \
    --embeddings_folder data/imagenet/ViT-g-14-laion2B-s34B-b88K/image_embedding \
    --text_embedding_path data/imagenet/ViT-g-14-laion2B-s34B-b88K/.npy \
    --output_dir data/models/distillation_models/ViT-g-14-laion2B-s34B-b88K/resnet18 \
    --output_dim 512 \
    --pretrained
```
输出模型的 checkpoints 会被保存在 ``data/models/distillation_models/ViT-g-14-laion2B-s34B-b88K/resnet18``.


<a name="step-4"></a>

## Step 4 - Run inference using the distilled model

### Compute text embeddings

During distillation, we trained our student model to match the *features* of our open-clip model.  However, we're interested in creating a classification model.
如果我们知识蒸馏的过程中是以图像嵌入和图像类别作为共同的目标，那学生模型就能直接用来做图像分类。如果只用了图像嵌入的话，就需要结合文本编码器做图像分类。
具体代码见``predict_accurancy.py``


### Predict single image with PyTorch

在一开始获得了text_prompts，可以利用它和pytorch模型进行图像分类

```bash
python3 predict_pytorch.py \
    resnet18 \
    data/models/resnet18/checkpoint.pth \
    data/text_embeddings.npy \
    assets/cat.jpg \
    --text_prompts data/text_prompts.txt
```



## 普通训练模型

为了对比知识蒸馏的作用，可以通过普通训练的方式得到一个模型进行对比，具体代码见``normal_train_for_zeroshot.py``




## 其他

```bash
python3 export_onnx.py \
    resnet18 \
    data/models/resnet18_sparse/checkpoint.pth \
    data/onnx/resnet18_sparse.onnx \
    --use_asp
```



