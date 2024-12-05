
# 打开原始TXT文件读取数据
with open('SUN/SUN397/ClassName.txt', 'r') as file:
    lines = file.readlines()

# 准备写入到新文件的数据
modified_lines = []

# 处理每一行
for line in lines:
    # 去除开头的字符（如'a', 'b', 'c'等）和斜杠，然后根据下划线分割
    content = line.strip()[2:]  # 这里去除了开头的 '/' 和随后的一个字符
    parts = content.split('/')
    # 对分割后的每个部分进一步处理，如果含有下划线则再分割
    final_parts = []
    for part in parts:
        if '_' in part:
            final_parts.extend(part.split('_'))
        else:
            final_parts.append(part)
    # 将处理后的部分用空格连接并添加到结果列表
    modified_lines.append(' '.join(final_parts) + '\n')

# 写入到新的TXT文件
with open('SUN_text_prompts.txt', 'w') as new_file:
    new_file.writelines(modified_lines)

print("数据已成功写入到 'modified_file.txt'")