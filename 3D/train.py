import os
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1. 參數與路徑設定
# -----------------------------
PATCHES_ROOT = "/media/disk2/HSW/MachineLearning/Final_project/3D/convert/train_patches"
INPUT_SHAPE = (64, 64, 64)  # 不包含 channel，因為 PyTorch tensor shape: (N, C, D, H, W)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID = 1

# -----------------------------
# 2. 讀取 .npz 檔案路徑並切分 Train/Val
# -----------------------------
all_npz_files = glob.glob(os.path.join(PATCHES_ROOT, "*", "*.npz"))
np.random.shuffle(all_npz_files)
split_ratio = 0.8
split_index = int(len(all_npz_files) * split_ratio)

train_files = all_npz_files[:split_index]
val_files   = all_npz_files[split_index:]

print(f"總共有 {len(all_npz_files)} 個 .npz 檔案")
print(f"Train set: {len(train_files)}  Val set: {len(val_files)}")


# -----------------------------
# 3. 建立 Dataset
# -----------------------------
class PatchDataset(Dataset):
    """
    讀取先前切好的 3D patches (.npz)
    每個 .npz 內含:
      - 'image': shape = (64,64,64)
      - 'label': shape = (64,64,64)
    需轉為 (C,D,H,W)，其中 C=1
    """
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        data = np.load(fpath)
        img = data['image']  # shape: (64,64,64)
        lbl = data['label']  # shape: (64,64,64)

        # 加 channel 維度 => (1,64,64,64)
        img = np.expand_dims(img, axis=0)  
        lbl = np.expand_dims(lbl, axis=0)  

        # 轉成 torch tensor, dtype = float32
        img = torch.from_numpy(img.astype(np.float32))
        lbl = torch.from_numpy(lbl.astype(np.float32))

        return img, lbl


# -----------------------------
# 4. 建立 DataLoader
# -----------------------------
train_dataset = PatchDataset(train_files)
val_dataset   = PatchDataset(val_files)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# -----------------------------
# 5. 定義 Dice Loss
# -----------------------------
def dice_coef(pred, target, eps=1e-6):
    """
    pred, target shape: (N, 1, D, H, W)
    pred 為模型的輸出 (sigmoid後)， target 為 0 or 1
    """
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = torch.sum(target * pred)
    union = torch.sum(target) + torch.sum(pred)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice

def dice_loss(pred, target):
    return 1.0 - dice_coef(pred, target)


# -----------------------------
# 6. 定義 3D U-Net
# -----------------------------
def conv_block_3d(in_ch, out_ch):
    """兩層 3D conv + BN + ReLU"""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNet3D(nn.Module):
    """
    簡化版 3D U-Net:
    Encoder(32->64->128) + Bottleneck(128) + Decoder(64->32)
    """
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
        cat2 = torch.cat([u2, c2], dim=1)  # (N,64+64, D/2,H/2,W/2)
        d2 = self.dec2(cat2)     # (N,64, D/2,H/2,W/2)

        u1 = self.up1(d2)        # (N,32,D,H,W)
        cat1 = torch.cat([u1, c1], dim=1)  # (N,32+32,D,H,W)
        d1 = self.dec1(cat1)     # (N,32, D,H,W)

        out = self.out_conv(d1)  # (N,1, D,H,W)
        # 最後要做 sigmoid
        return out


# -----------------------------
# 7. 初始化模型/優化器
# -----------------------------
model = UNet3D().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 可考慮結合 BCE + dice，也可只用 dice_loss
bce_loss_func = nn.BCEWithLogitsLoss()


# -----------------------------
# 8. 訓練/驗證函式
# -----------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)   # shape: (B,1,D,H,W)
        labels = labels.to(DEVICE)   # shape: (B,1,D,H,W)

        # forward
        logits = model(images)       # (B,1,D,H,W) raw logits
        # loss: e.g. BCE + dice
        bce_loss = bce_loss_func(logits, labels)
        probs = torch.sigmoid(logits)
        d_loss = dice_loss(probs, labels)
        
        loss = bce_loss + d_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 計算 dice (for metric)
        dice_val = dice_coef(probs, labels).item()
        epoch_dice += dice_val

    n = len(loader)
    return epoch_loss/n, epoch_dice/n


def validate(model, loader):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            bce_loss = bce_loss_func(logits, labels)
            probs = torch.sigmoid(logits)
            d_loss = dice_loss(probs, labels)
            loss = bce_loss + d_loss
            
            val_loss += loss.item()

            # dice
            dice_val = dice_coef(probs, labels).item()
            val_dice += dice_val

    n = len(loader)
    return val_loss/n, val_dice/n


# -----------------------------
# 9. 主訓練迴圈 (含 EarlyStopping)
# -----------------------------
best_val_loss = float("inf")
patience = 10
waiting = 0

for epoch in range(1, EPOCHS+1):
    train_loss, train_dice = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_dice = validate(model, val_loader)

    print(f"[Epoch {epoch}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f} | "
          f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

    # 檢查是否為最優
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        waiting = 0
        # 儲存最佳權重
        torch.save(model.state_dict(), f"/media/disk2/HSW/MachineLearning/Final_project/3D/train-test_output/models/best_3dunet_{ID}.pth")
        print("  --> Best model saved!")
    else:
        waiting += 1
        if waiting >= patience:
            print(f"Early stopping at epoch {epoch}!")
            break

# -----------------------------
# 10. 儲存最終模型
# -----------------------------
torch.save(model.state_dict(), f"/media/disk2/HSW/MachineLearning/Final_project/3D/train-test_output/models/final_3dunet_model_{ID}.pth")
print("Training complete. Final model saved.")
