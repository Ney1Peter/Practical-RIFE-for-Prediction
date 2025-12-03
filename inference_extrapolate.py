import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import skvideo.io

warnings.filterwarnings("ignore")


#########################################
# 音频拷贝函数（与原版一致）
#########################################
def transferAudio(sourceVideo, targetVideo):
    import shutil
    tempAudioFileName = "./temp/audio.mkv"

    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")

    # 仅拷贝音频
    os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a copy -vn {tempAudioFileName}')

    # 先把无音频视频改名
    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)

    # 合成视频 + 音频
    os.system(f'ffmpeg -y -i "{targetNoAudio}" -i {tempAudioFileName} -c copy "{targetVideo}"')

    # 检查大小
    if os.path.getsize(targetVideo) == 0:
        print("音频合成失败，使用 AAC 重试")
        tempAudioFileName = "./temp/audio.m4a"
        os.system(f'ffmpeg -y -i "{sourceVideo}" -c:a aac -b:a 160k -vn {tempAudioFileName}')
        os.system(f'ffmpeg -y -i "{targetNoAudio}" -i {tempAudioFileName} -c copy "{targetVideo}"')

        if os.path.getsize(targetVideo) == 0:
            print("音频合成完全失败")
            os.rename(targetNoAudio, targetVideo)
        else:
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    shutil.rmtree("temp")


#########################################
# 参数
#########################################
parser = argparse.ArgumentParser(description='RIFE 外插推理（未来预测）')
parser.add_argument('--video', type=str, required=True)
parser.add_argument('--output', type=str, default="output_extrapolated.mp4")
parser.add_argument('--model', type=str, default="train_log")
parser.add_argument('--fps', type=int, default=None)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--t', type=float, default=1.3, help="外插时间 (默认=2.0，越大越往未来)")

args = parser.parse_args()


#########################################
# 加载 RIFE-HDv3（支持 timestep）
#########################################
from train_log.RIFE_HDv3 import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

model = Model()
model.load_model(args.model, -1)
model.eval()
model.device()

print("Loaded RIFE HDv3 (supports timestep extrapolation)")


#########################################
# 视频信息读取
#########################################
cap = cv2.VideoCapture(args.video)
fps_in = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

fps_out = fps_in if args.fps is None else args.fps

# skvideo 读取的视频是 **RGB**
videogen = skvideo.io.vreader(args.video)
firstframe = next(videogen)   # RGB

h, w, _ = firstframe.shape


#########################################
# OpenCV VideoWriter 必须使用 BGR
#########################################
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_out = cv2.VideoWriter(args.output, fourcc, fps_out, (w, h))


#########################################
# 工具函数
#########################################
def pad_image(img):
    return F.pad(img, padding)

def frame_to_tensor(f_rgb):
    """ skvideo 是 RGB，因此保持 RGB → Tensor """
    return torch.from_numpy(np.transpose(f_rgb, (2,0,1))).float().to(device).unsqueeze(0) / 255.


#########################################
# RIFE padding 计算
#########################################
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)


#########################################
# 主循环（每三帧 → 外插下一帧）
#########################################
pbar = tqdm(total=total_frames)
frame_buffer = [firstframe]  # RGB

for frame_rgb in videogen:
    frame_buffer.append(frame_rgb)
    pbar.update(1)

    if len(frame_buffer) == 3:
        f1, f2, orig3 = frame_buffer

        # 转为 tensor
        I1 = pad_image(frame_to_tensor(f1))
        I2 = pad_image(frame_to_tensor(f2))

        # ⚠️ 外插核心：使用 t=2.0（未来两帧）
        pred = model.inference(I1, I2, timestep=args.t, scale=args.scale)

        # 转为 RGB numpy
        pred_np = (pred[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
        pred_np = pred_np[:h, :w]

        # ⚠️ OpenCV 是 BGR，必须转换
        pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
        f1_bgr = cv2.cvtColor(f1, cv2.COLOR_RGB2BGR)
        f2_bgr = cv2.cvtColor(f2, cv2.COLOR_RGB2BGR)

        # 写入视频（始终 BGR）
        vid_out.write(f1_bgr)
        vid_out.write(f2_bgr)
        vid_out.write(pred_bgr)

        frame_buffer = []

pbar.close()


#########################################
# 合成音频
#########################################
vid_out.release()
transferAudio(args.video, args.output)

print("\n处理完成:", args.output)
