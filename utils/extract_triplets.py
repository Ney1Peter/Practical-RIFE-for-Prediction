import cv2
import os
import argparse
from tqdm import tqdm


# ============================================================
# 读取视频基础信息
# ============================================================
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("无法打开视频: " + video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": w,
        "height": h,
        "duration_sec": duration
    }


# ============================================================
# 将视频切帧
# ============================================================
def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    frame_paths = []

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="切帧中")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 输出文件名
        fname = f"frame_{frame_id:05d}.png"
        fpath = os.path.join(output_dir, fname)

        cv2.imwrite(fpath, frame)

        frame_paths.append(fpath)
        frame_id += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    return frame_paths


# ============================================================
# 生成 3-frame triplets
# ============================================================
def generate_triplets(frame_paths, output_txt):
    with open(output_txt, "w") as f:
        for i in range(len(frame_paths) - 2):
            f.write(f"{frame_paths[i]} {frame_paths[i+1]} {frame_paths[i+2]}\n")

    print(f" 已生成三帧一组列表: {output_txt}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频生成三帧训练样本")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="视频文件名（放在 data/ 下）"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # 固定你的路径
    # ------------------------------------------------------------
    DATA_ROOT = "/home/zheng/Practical-RIFE-for-Prediction/data/train"

    # 视频真实路径
    video_path = os.path.join(DATA_ROOT, args.video)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    # 输出帧目录
    video_name = os.path.splitext(args.video)[0]
    frame_dir = os.path.join(DATA_ROOT, video_name)

    # triplets 文件
    triplet_txt = os.path.join(DATA_ROOT, f"{video_name}_triplets.txt")

    # ------------------------------------------------------------
    # 打印视频信息
    # ------------------------------------------------------------
    print("\n================== 视频信息 ==================")
    info = get_video_info(video_path)
    print(f"分辨率:       {info['width']} x {info['height']}")
    print(f"FPS:          {info['fps']}")
    print(f"总帧数:       {info['frame_count']}")
    print(f"时长(秒):     {info['duration_sec']:.2f}")
    print(f"时长(分钟):   {info['duration_sec']/60:.2f}")
    print("================================================\n")

    # ------------------------------------------------------------
    # 切帧
    # ------------------------------------------------------------
    print(f"开始切帧：{video_path}")
    print(f"输出目录：{frame_dir}\n")
    frame_paths = extract_frames(video_path, frame_dir)
    print(f" 总共提取帧数: {len(frame_paths)}\n")

    # ------------------------------------------------------------
    # 生成 triplets.txt
    # ------------------------------------------------------------
    print("生成三帧训练列表 ...")
    generate_triplets(frame_paths, triplet_txt)

    print("\n 完成！数据已经可以用于训练。")
