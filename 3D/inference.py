import os
import glob
import numpy as np
import torch
import torch.nn as nn
import shutil

##############################################################################
# 1) 參數與路徑設定
##############################################################################
test_patches_root = "/media/disk2/HSW/MachineLearning/Final_project/3D/convert/test_patches"
# 結構例如：
# test_patches/
#   001/
#     001_patch000.npz
#     001_patch001.npz
#     ...
#   002/
#     002_patch000.npz
#     ...
#   ...
#
# 其中每個子資料夾對應原始 (300,300,1259) 影像，但經過 padding => (320,320,1280) =>
# 再切成 (64,64,64) patches (無重疊)。

ID = 1  # 您的模型 ID
EPOCH = 27   # 您的模型訓練到第幾個 epoch
OUTPUT_ROOT = F"/media/disk2/HSW/MachineLearning/Final_project/3D/train-test_output/inference/3dunet_{ID}-{EPOCH}"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

MODEL_PATH = "/media/disk2/HSW/MachineLearning/Final_project/3D/train-test_output/models/best_3dunet_1.pth"  # 您的訓練好 PyTorch 權重檔

# 原始影像大小
ORIG_SHAPE = (300, 300, 1259)
# Padding 後大小 (需和您切 patches 時一致)
PAD_SHAPE = (320, 320, 1280)
PATCH_SIZE = (64,64,64)
STRIDE = 64  # 若當時是無重疊，則 stride=64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

##############################################################################
# 2) 定義 3D U-Net (需與訓練階段結構相同)
##############################################################################
def conv_block_3d(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        # Encoder
        self.enc1 = conv_block_3d(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block_3d(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = conv_block_3d(64, 128)

        # Decoder
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block_3d(64+64, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block_3d(32+32, 32)

        self.out_conv = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)        # (N,32,D,H,W)
        p1 = self.pool1(c1)      # (N,32,D/2,H/2,W/2)
        c2 = self.enc2(p1)       # (N,64,D/2,H/2,W/2)
        p2 = self.pool2(c2)      # (N,64,D/4,H/4,W/4)

        # Bottleneck
        b  = self.bottleneck(p2) # (N,128,D/4,H/4,W/4)

        # Decoder
        u2 = self.up2(b)         # (N,64,D/2,H/2,W/2)
        cat2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(cat2)     # (N,64,D/2,H/2,W/2)

        u1 = self.up1(d2)        # (N,32,D,H,W)
        cat1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(cat1)     # (N,32,D,H,W)

        logits = self.out_conv(d1) # (N,1,D,H,W)
        return logits


##############################################################################
# 3) 建立並載入模型
##############################################################################
model = UNet3D().to(DEVICE)
# 載入 state_dict
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded model from {MODEL_PATH}")

##############################################################################
# 4) 拼回與去除 padding 的函式 (無重疊版本)
##############################################################################
def reconstruct_volume(patches, vol_shape, stride=64):
    """
    將 (N,64,64,64) 的 patches, 依照無重疊 (stride=64) 排列
    拼回 shape=vol_shape (e.g. (320,320,1280))
    patches 順序必須與切塊時一致 (z,y,x) 三重迴圈
    """
    D, H, W = vol_shape
    recon = np.zeros((D, H, W), dtype=np.float32)

    idx = 0
    num_z = D // stride  # e.g. 320/64=5
    num_y = H // stride  # e.g. 320/64=5
    num_x = W // stride  # e.g. 1280/64=20

    for z in range(num_z):
        for y in range(num_y):
            for x in range(num_x):
                patch = patches[idx]
                idx += 1
                z0, y0, x0 = z*stride, y*stride, x*stride
                recon[z0:z0+64, y0:y0+64, x0:x0+64] = patch
    return recon

def unpad_volume(volume, orig_shape):
    """
    依原始大小 (300,300,1259), 
    從 padding 後的 (320,320,1280) volume 裁減回去
    """
    D, H, W = orig_shape
    return volume[:D, :H, :W]


##############################################################################
# 5) 推論流程
##############################################################################
test_subfolders = sorted(glob.glob(os.path.join(test_patches_root, "*")))

with torch.no_grad():
    for subf in test_subfolders:
        if not os.path.isdir(subf):
            continue
        
        folder_name = os.path.basename(subf)
        print(f"[推論] 處理測試資料夾: {folder_name}")

        patch_files = sorted(glob.glob(os.path.join(subf, "*.npz")))
        if len(patch_files) == 0:
            print(f"  -> [警告] 無 .npz 檔案，跳過")
            continue

        # 讀取全部 patch
        all_patches = []
        for pf in patch_files:
            data = np.load(pf)
            patch = data['image']  # shape=(64,64,64)
            all_patches.append(patch)

        all_patches = np.array(all_patches, dtype=np.float32)  # (N,64,64,64)
        N = len(all_patches)
        print(f"  -> 共有 {N} 個 patches, shape={all_patches.shape}")

        # 為了避免一次載入過多可用 batch 處理：
        batch_size = 2
        pred_results = []

        # 分批推論
        for i in range(0, N, batch_size):
            batch_data = all_patches[i:i+batch_size]  # shape=(b,64,64,64)
            # 增加 channel 維度 => (b,1,64,64,64)
            batch_data = batch_data[:, np.newaxis, ...]  
            batch_tensor = torch.from_numpy(batch_data).float().to(DEVICE)

            # forward
            logits = model(batch_tensor)         # shape=(b,1,64,64,64)
            probs  = torch.sigmoid(logits)       # => [0,1]
            probs_np = probs.squeeze(1).cpu().numpy()  # shape=(b,64,64,64)
            pred_results.append(probs_np)
        
        pred_results = np.concatenate(pred_results, axis=0)  # (N,64,64,64)

        # 拼回 (320,320,1280)
        recon_padded = reconstruct_volume(pred_results, PAD_SHAPE, stride=STRIDE)
        # 去 padding => (300,300,1259)
        recon_unpad = unpad_volume(recon_padded, ORIG_SHAPE)

        # 您可考慮做二值化:
        recon_bin = (recon_unpad > 0.5).astype(np.uint8)
        # 或保留浮點機率
        out_subdir = os.path.join(OUTPUT_ROOT, folder_name)
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, f"{folder_name}_predict.npy")

        np.save(out_path, recon_bin)
        print(f"  -> 完成推論, 結果存於: {out_path}\n")
