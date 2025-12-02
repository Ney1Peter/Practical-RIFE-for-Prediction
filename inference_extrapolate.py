import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")


# 音频拷贝
def transferAudio(sourceVideo, targetVideo):
    import shutil
    tempAudioFileName = "./temp/audio.mkv"

    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")

    os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)

    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(
        targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0:
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(
            sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(
            targetNoAudio, tempAudioFileName, targetVideo))

        if os.path.getsize(targetVideo) == 0:
            os.rename(targetNoAudio, targetVideo)
            print("音频合成失败，输出视频无音频。")
        else:
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    shutil.rmtree("temp")


# 参数解析
parser = argparse.ArgumentParser(description='RIFE 外插推理（1,2→3）')
parser.add_argument('--video', type=str, required=True, help="输入视频路径")
parser.add_argument('--output', type=str, default="output_extrapolated.mp4", help="外插后的输出视频")
parser.add_argument('--model', type=str, default="train_log", help="模型权重路径（train_log/flownet.pkl）")
parser.add_argument('--fps', type=int, default=None, help="输出视频帧率（默认保持与源视频一致）")
parser.add_argument('--fp16', action='store_true', help="使用半精度推理")
parser.add_argument('--UHD', action='store_true', help="是否处理超高清（自动缩放）")
parser.add_argument('--scale', type=float, default=1.0, help="缩放因子，4K 视频建议使用 0.5")
parser.add_argument('--ext', type=str, default="mp4")

args = parser.parse_args()

# 加载模型 支持 timestep 外插
from train_log.RIFE_HDv3 import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

model = Model()
model.load_model(args.model, -1)
model.eval()
model.device()


# 视频读取信息
cap = cv2.VideoCapture(args.video)
fps_in = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# 输出帧率：默认保持不变
fps_out = fps_in if args.fps is None else args.fps

videogen = skvideo.io.vreader(args.video)
firstframe = next(videogen)

h, w, _ = firstframe.shape

# 创建视频输出
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
vid_out = cv2.VideoWriter(args.output, fourcc, fps_out, (w, h))


# 工具函数
def pad_image(img):
    """按照 RIFE 的要求对输入图像做边界 padding"""
    return F.pad(img, padding)

def frame_to_tensor(f):
    """把 numpy BGR 图像转换为模型需要的 1x3xHxW tensor，并归一化"""
    return torch.from_numpy(np.transpose(f, (2,0,1))).float().to(device).unsqueeze(0) / 255.


# 根据 RIFE 规则计算 padding，使宽高都对齐到 128 倍数
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)   # (left, right, top, bottom)


# 主循环（每三帧一组：f1,f2→预测 f3）
pbar = tqdm(total=total_frames)
frame_buffer = [firstframe]   # 缓存三帧

for frame in videogen:
    frame_buffer.append(frame)
    pbar.update(1)

    # 缓存满三帧再处理： f1, f2 → pred3
    if len(frame_buffer) == 3:
        f1, f2, orig3 = frame_buffer

        # 转为 tensor 并 padding
        I1 = pad_image(frame_to_tensor(f1))
        I2 = pad_image(frame_to_tensor(f2))

        # 外插核心：使用 t=1.0
        # RIFE 会假设 t=0.0 → 第一帧
        #            t=1.0 → 第二帧之后的下一帧（外推）
        pred3 = model.inference(I1, I2, timestep=1.0, scale=args.scale)

        # 转 numpy 进行写入
        pred3 = (pred3[0] * 255).byte().cpu().numpy().transpose(1,2,0)[:h, :w]

        # 输出顺序：写入 f1, f2, 预测出的 f3
        vid_out.write(f1)
        vid_out.write(f2)
        vid_out.write(pred3)

        # 清空缓存，进入下一组三帧
        frame_buffer = []

pbar.close()

# 合成音频
vid_out.release()
transferAudio(args.video, args.output)

print("处理完成:", args.output)
