import os

directory = "data/imagenet/image_embedding"
npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
print("Number of .npy files:", len(npy_files))
# import os

# def count_images_in_directory(directory):
#     total_count = 0
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(('JPEG','.jpg', '.jpeg', '.png', '.bmp', '.gif')):  # 常见的图片文件扩展名
#                 total_count += 1
#     return total_count

# image_folder_path = "../ImageNet/train"  # 替换为您的ImageNet文件夹的实际路径
# total_images = count_images_in_directory(image_folder_path)
# print(f"Total number of images in the ImageNet folder: {total_images}")