FROM nvcr.io/nvidia/pytorch:23.01-py3
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install timm open_clip_torch -i https://pypi.tuna.tsinghua.edu.cn/simple/

# upgrade pillow to fix "UnidentifiedImageError"
RUN pip install pillow --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple/