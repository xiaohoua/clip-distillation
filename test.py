import os
import glob

def count_images_in_folders(root_folder):
    """递归统计指定根目录下所有图片的数量"""
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.gif')  # 定义图片文件的扩展名
    total_count = 0  # 初始化图片总数
    
    # 遍历根目录下的所有文件和子文件夹
    for foldername, subfolders, filenames in os.walk(root_folder):
        # 对于每个子文件夹，统计其中的图片
        for extension in image_extensions:
            # 使用glob匹配当前文件夹下的所有图片文件
            images = glob.glob(os.path.join(foldername, extension))
            total_count += len(images)
            # 可选：打印每个子文件夹的图片数量
            # print(f"{foldername}: {len(images)} images")
    
    return total_count

# 设置根目录
root_directory = 'SUN/SUN397'  # 请将此处的 'your_root_directory' 替换为你的实际目录路径

# 调用函数并打印结果
total_images = count_images_in_folders(root_directory)
print(f"Total images in '{root_directory}' and its subfolders: {total_images}")