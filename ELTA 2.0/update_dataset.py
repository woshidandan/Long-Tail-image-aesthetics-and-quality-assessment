import pandas as pd
import os

'''
将基于已有图像生成的新图像的信息添加到原始的标注文件
'''

# 原始标注文件
csv_file = '/home/llm/elta/code/dataset_backup/TAD66K/train.csv'

# 生成图像的总目录
generated_images_root = 'diffusion/result/TAD66K/generated/'

# 读取原始csv
df = pd.read_csv(csv_file)

# 建立一个image到(score, category)的查找表
image_to_info = dict(zip(df['image_id'], zip(df['score'], df['category'])))

# 收集新增的数据行
new_rows = []

# 遍历 generated_images 目录下的所有类别子目录
for category_dir in os.listdir(generated_images_root):
    category_path = os.path.join(generated_images_root, category_dir)
    
    if not os.path.isdir(category_path):
        continue  # 忽略不是目录的东西
    
    # 遍历子目录下的所有生成图像
    for gen_image_name in os.listdir(category_path):
        if not gen_image_name.endswith('.jpg'):
            continue

        gen_image_full_path = os.path.join(category_path, gen_image_name)

        # 去掉后缀，得到原图名
        original_image_name = gen_image_name.split('_generated_')[0]

        # 查找原图信息
        if original_image_name in image_to_info:
            score, category = image_to_info[original_image_name]
            new_rows.append({
                'image_id': gen_image_name, 
                'score': score,
                'category': category
            })
        else:
            print(f"警告：找不到原图 {original_image_name}，跳过")

# 将新增的行转成DataFrame
new_df = pd.DataFrame(new_rows)

# 合并原始数据和新增数据
final_df = pd.concat([df, new_df], ignore_index=True)

# 保存到新的csv
final_df.to_csv('/home/llm/elta/code/dataset_backup/TAD66K/update_train.csv', index=False)

print(f"\n处理完成，新增 {len(new_rows)} 条数据，已更新train.csv")


