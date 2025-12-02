import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

############################################################
# 基础设置
############################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


############################################################
# 参数
############################################################
parser = argparse.ArgumentParser(description='RIFE extrapolation (img1,img2 → predicted img3)')
parser.add_argument('--img', nargs=2, required=True, help="输入两张图片路径")
parser.add_argument('--t', default=1.0, type=float, help="外插时间 t，默认=1.0")
parser.add_argument('--model', type=str, default='train_log', help='模型文件目录')
parser.add_argument('--output', type=str, default='pred.png', help="预测结果输出路径")

args = parser.parse_args()


############################################################
# 加载模型（自动匹配 v1/v2/v3/v4）
############################################################
try:
    try:
        from model.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.model, -1)
        print("Loaded v2.x HD model")
    except:
        from train_log.RIFE_HDv3 import Model
        model = Model()
        model.load_model(args.model, -1)
        print("Loaded v3.x/4.x HD model")
except:
    from model.RIFE_HD import Model
    model = Model()
    model.load_model(args.model, -1)
    print("Loaded v1.x HD model")

# old model (v3.x) compatibility
if not hasattr(model, 'version'):
    model.version = 0

model.eval()
model.device()


############################################################
# 读取图像
############################################################
img0 = cv2.imread(args.img[0], cv2.IMREAD_COLOR)
img1 = cv2.imread(args.img[1], cv2.IMREAD_COLOR)

# 统一为较小分辨率（可选）
#img0 = cv2.resize(img0, (448, 256))
#img1 = cv2.resize(img1, (448, 256))

# 转为 Tensor 格式 (1,3,H,W)
img0_t = torch.from_numpy(img0.transpose(2, 0, 1)).float().to(device) / 255.
img1_t = torch.from_numpy(img1.transpose(2, 0, 1)).float().to(device) / 255.
img0_t = img0_t.unsqueeze(0)
img1_t = img1_t.unsqueeze(0)

# padding 到 64 倍数
n, c, h, w = img0_t.shape
ph = ((h - 1) // 64 + 1) * 64
pw = ((w - 1) // 64 + 1) * 64
padding = (0, pw - w, 0, ph - h)

img0_t = F.pad(img0_t, padding)
img1_t = F.pad(img1_t, padding)


############################################################
# 外插逻辑：关键点！！！
# 工程版 RIFE-HDv3/v4 使用 timestep=t
# t = 1.0 表示向前预测一帧（外插）
############################################################
if model.version >= 3.9:
    print(f"Model supports timestep, predicting with t={args.t}")
    pred = model.inference(img0_t, img1_t, timestep=args.t)
else:
    ########################################################
    # 旧版模型不支持 timestep，因此无法外插
    # 只能做 t=0.5 的插帧
    ########################################################
    print("⚠ 当前模型不支持 timestep（t>1 外插），只能插帧 t=0.5")
    pred = model.inference(img0_t, img1_t)   # t=0.5
    print("输出的是插帧结果（中间帧），不是预测第三帧！")


############################################################
# 保存输出图片
############################################################
pred_img = (pred[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
pred_img = pred_img[:h, :w]

cv2.imwrite(args.output, pred_img)
print("Saved:", args.output)
