import json
import os
import random
import argparse

def main():
    # ================= 配置区域 =================
    # 数据集根目录 (请修改为你实际的路径)
    ROOT_PATH = '$PATH_TO_DATASET$/Data_indoor' 
    
    # 划分比例
    TRAIN_RATIO = 0.8
    
    # 随机种子 (保证每次运行生成的划分结果一致)
    SEED = 2026
    # ===========================================

    print(f"Dataset Root: {ROOT_PATH}")
    clips_path = os.path.join(ROOT_PATH, 'clips.json')
    frames_path = os.path.join(ROOT_PATH, 'frames.json')

    # 1. 检查文件是否存在
    if not os.path.exists(clips_path) or not os.path.exists(frames_path):
        print(f"Error: 找不到 clips.json 或 frames.json，请检查路径: {ROOT_PATH}")
        return

    # 2. 加载数据
    print("Loading JSON files...")
    with open(clips_path, 'r') as f:
        all_clips = json.load(f)
    with open(frames_path, 'r') as f:
        all_frames = json.load(f)

    print(f"Loaded {len(all_clips)} clips and {len(all_frames)} frames.")

    # 3. 按 Clip 进行打乱和划分
    random.seed(SEED)
    random.shuffle(all_clips)

    split_idx = int(len(all_clips) * TRAIN_RATIO)
    
    train_clips = all_clips[:split_idx]
    val_clips = all_clips[split_idx:]

    # 提取 Clip 的 token 用于索引
    # 根据你提供的结构：clips.json 中的唯一标识是 "token"
    train_clip_tokens = set([c['token'] for c in train_clips])
    val_clip_tokens = set([c['token'] for c in val_clips])

    print(f"Split Result (Clips): Train={len(train_clips)}, Val={len(val_clips)}")

    # 4. 根据 Clip 的归属分配 Frame
    train_frames = []
    val_frames = []

    print("Distributing frames...")
    for frame in all_frames:
        # 根据你提供的结构：frames.json 中关联 Clip 的字段是 "scene_token"
        s_token = frame.get('scene_token')

        if s_token in train_clip_tokens:
            train_frames.append(frame)
        elif s_token in val_clip_tokens:
            val_frames.append(frame)
        else:
            # 这种情况通常不应该发生，除非 frames 里有 clips 里没有的 token
            print(f"Warning: Frame {frame.get('token')} matches no clip (scene_token: {s_token})")

    print(f"Split Result (Frames): Train={len(train_frames)}, Val={len(val_frames)}")

    # 5. 保存结果
    train_json_path = os.path.join(ROOT_PATH, 'train_frames.json')
    val_json_path = os.path.join(ROOT_PATH, 'val_frames.json')

    print(f"Saving to {train_json_path} ...")
    with open(train_json_path, 'w') as f:
        json.dump(train_frames, f, indent=4)

    print(f"Saving to {val_json_path} ...")
    with open(val_json_path, 'w') as f:
        json.dump(val_frames, f, indent=4)

    print("Done! Dataset split complete.")

if __name__ == "__main__":
    main()