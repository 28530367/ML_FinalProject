import os
import glob
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import transforms
from datetime import datetime

########################################
# 檢查 cuda
########################################
print(torch.cuda.is_available())
print(torch.cuda.device_count())

########################################
# 建立資料集類別
########################################
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        assert len(self.img_paths) == len(self.mask_paths), "影像與mask數量不一致"
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert('L')   # img轉為灰階
        mask = Image.open(mask_path).convert('L')     # mask為灰階

        if self.transform is not None:
            # 將影像進行 transform
            image = self.transform(image)
            # mask為0與1的二元分類，轉成tensor後應該shape為(1,H,W)
            mask = transforms.ToTensor()(mask)
            # 確保mask為0/1的二元類別
            mask = (mask > 0.5).float()  # 若原本為0/255，這行將其正規化為0/1
        
        return image, mask
    
########################################
# U-Net (只做兩次pooling)
########################################
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetTwoPools(nn.Module):
    def __init__(self, n_class=1):
        super(UNetTwoPools, self).__init__()

        self.dconv_down1 = DoubleConv(1, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dconv_up2 = DoubleConv(256 + 128, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        # 下採樣部分
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)        # 1st pooling
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)        # 2nd pooling
        conv3 = self.dconv_down3(x)
        
        # 上採樣部分
        x = self.upsample(conv3)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        return out
    
########################################
# 設定訓練
########################################
def train():
    # 模型id
    id = 10

    # 資料路徑
    train_img_dir = f'/media/disk2/HSW/MachineLearning/Final_project/2D/train/imgs'
    train_mask_dir = f'/media/disk2/HSW/MachineLearning/Final_project/2D/train/masks'

    # 紀錄
    log_str = ""

    # 開始時間
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_now)  # 輸出類似：2024-12-15 12:34:56
    log_str +=  "Start time : " + formatted_now + "\n"

    # 不縮放影像，僅轉 tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 建立資料集與 DataLoader
    train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetTwoPools(n_class=1).to(device)
    
    # 使用二元分割對應的損失函數
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    print("Setting epochs: ", epochs)
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        output_line = f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.5f}"
        print(output_line)      # 在終端機印出
        log_str += output_line + "\n"  # 將此行結果累加到log_str中

        # 每10個 epoch 儲存一次模型
        if epoch % 10 == 0:
            # 儲存訓練完成的模型
            torch.save(model.state_dict(), f"/media/disk2/HSW/MachineLearning/Final_project/2D/train-test_output/model/unet_segmentation_{id}.pth")
            print("模型已儲存。")

            # 結束時間
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
            print(formatted_now)  # 輸出類似：2024-12-15 12:34:56
            log_str += "End time : " + formatted_now + "\n"

            file_path = f"/media/disk2/HSW/MachineLearning/Final_project/2D/train-test_output/model/unet_segmentation_{id}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(log_str)
            print(f"訓練紀錄已寫入 {file_path}")

            id = id + 1

if __name__ == "__main__":
    train()