import os
import glob
from tqdm import tqdm

def get_all_classes(root_path):
    print(f"Scanning directory: {root_path} ...")
    
    # 1. 寻找所有 .txt 文件
    # 你的结构是 annotation/bbox/token/*.txt
    # 使用 recursive=True 递归查找所有子文件夹下的 txt
    search_pattern = os.path.join(root_path, '**', '*.txt')
    txt_files = glob.glob(search_pattern, recursive=True)
    
    if len(txt_files) == 0:
        print("Error: No .txt files found! Please check the path.")
        return

    print(f"Found {len(txt_files)} files. Reading classes...")
    
    unique_classes = set()
    
    # 2. 遍历文件读取类别
    for file_path in tqdm(txt_files):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        # KITTI 格式的第一列是类别 (Type)
                        obj_type = parts[0]
                        unique_classes.add(obj_type)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 3. 打印结果
    sorted_classes = sorted(list(unique_classes))
    
    print("\n" + "="*40)
    print(f"统计完成！共发现 {len(sorted_classes)} 个唯一类别")
    print("="*40)
    
    # 打印列表形式，方便你直接复制到代码里
    print("classes = [")
    for idx, cls in enumerate(sorted_classes):
        print(f"    '{cls}',")
    print("]")
    print("="*40)

if __name__ == "__main__":
    # 你指定的路径
    BBOX_ROOT = '$PATH_TO_DATASET$/Data_indoor/annotation/bbox'
    
    if os.path.exists(BBOX_ROOT):
        get_all_classes(BBOX_ROOT)
    else:
        print(f"Path does not exist: {BBOX_ROOT}")