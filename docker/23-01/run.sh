#!/bin/bash


docker run \
    -it \
    -d \
    --rm \
    --ipc host \
    --gpus all \
    --shm-size 14G \
    -v $(pwd):/clip-distillation \
    -v /data/dataset/ImageNet/extract/:/ImageNet:ro \
    clip_distillation:23-01