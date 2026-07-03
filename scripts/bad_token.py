import json
import shutil
from pathlib import Path

# 你给的是 scene_token（不是 frame token）
BAD_SCENE_TOKENS = {
    "68ea2c36935b8e4ab9be9e79",
    "68ea2c36935b8e4ab9be9e8f",
    "68ea2c36935b8e4ab9be9e91",
    "68ea2c36935b8e4ab9be9e92",
    "68edc5b6f13153901bf8dddc",
    "68ea2c36935b8e4ab9be9e76",
    "68ea2c36935b8e4ab9be9e7a",
    "68ea2c36935b8e4ab9be9e90",
    "68ea2c36935b8e4ab9be9e97",
}

FRAMES_JSON = "$PATH_TO_DATASET$/Data_indoor/val_frames.json"  # 改成你的训练/测试集 frames.json 路径

def main():
    p = Path(FRAMES_JSON)
    assert p.exists(), f"Not found: {p}"

    # 备份
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)

    with p.open("r", encoding="utf-8") as f:
        frames = json.load(f)

    # 统计：frames.json 里出现过的 scene_token
    present_scene_tokens = {fr.get("scene_token") for fr in frames if isinstance(fr, dict)}
    not_found = sorted([t for t in BAD_SCENE_TOKENS if t not in present_scene_tokens])

    # 过滤：删除 scene_token 命中的整条 frame（即整个 scene 的所有帧都会被删掉）
    new_frames = [
        fr for fr in frames
        if not (isinstance(fr, dict) and fr.get("scene_token") in BAD_SCENE_TOKENS)
    ]

    removed = len(frames) - len(new_frames)

    # 写回（不改其它任何字段，只是删掉对应条目）
    with p.open("w", encoding="utf-8") as f:
        json.dump(new_frames, f, ensure_ascii=False, indent=2)

    print(f"[OK] frames before: {len(frames)}")
    print(f"[OK] removed frames: {removed}")
    print(f"[OK] frames after: {len(new_frames)}")
    if not_found:
        print("[WARN] scene_tokens not found in this frames.json:")
        for t in not_found:
            print("  -", t)
    print(f"[OK] backup saved at: {bak}")

if __name__ == "__main__":
    main()
