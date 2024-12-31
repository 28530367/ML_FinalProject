## model
### unet_segmentation_1.pth
- train: <br>
dark-side-train-data-part1.zip<br>
- batch_size: 16
- epochs: 20
- socre: NaN

### unet_segmentation_2.pth
- train: <br>
dark-side-train-data-part1.zip
- batch_size: 16
- epochs: 30
- socre: 0.492031

### unet_segmentation_3.pth
- train: <br>
dark-side-train-data-part1.zip<br>
dark-side-train-data-part2.zip<br>
dark-side-train-data-part3.zip<br>
dark-side-train-data-part4.zip<br>
dark-side-train-data-part5.zip<br>
- batch_size: 32
- epochs: 30
- socre: 0.599375

### unet_segmentation_4.pth
- train: <br>
dark-side-train-data-part1.zip<br>
dark-side-train-data-part2.zip<br>
dark-side-train-data-part3.zip<br>
dark-side-train-data-part4.zip<br>
dark-side-train-data-part5.zip<br>
- batch_size: 16
- epochs: 30
- socre: 0.593503

### unet_segmentation_5.pth
- train: <br>
dark-side-train-data-part1.zip<br>
dark-side-train-data-part2.zip<br>
dark-side-train-data-part3.zip<br>
dark-side-train-data-part4.zip<br>
dark-side-train-data-part5.zip<br>
dark-side-train-data-part6.zip<br>
dark-side-train-data-part7.zip<br>
dark-side-train-data-part8.zip<br>
dark-side-train-data-part9.zip<br>
dark-side-train-data-part10.zip<br>
- batch_size: 32
- epochs: 50
- socre: 0.638715

### unet_segmentation_6.pth
- train: <br>
dark-side-train-data-part1.zip<br>
dark-side-train-data-part2.zip<br>
dark-side-train-data-part3.zip<br>
dark-side-train-data-part4.zip<br>
dark-side-train-data-part5.zip<br>
- batch_size: 32
- epochs: 60
- socre: 0.567519
- remark: 輸入影像做對比度以及降噪處理

```py
# 讀取影像與mask
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰階讀取影像

# CLAHE 對比度強化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(image)

# 高斯濾波處理
smoothed_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

# 將處理後的影像轉為 PIL 格式
image = Image.fromarray(smoothed_image)
```

### unet_segmentation_7.pth
- train: <br>
dark-side-train-data-part1.zip<br>
dark-side-train-data-part2.zip<br>
dark-side-train-data-part3.zip<br>
dark-side-train-data-part4.zip<br>
dark-side-train-data-part5.zip<br>
- batch_size: 32
- epochs: 80
- socre: 0.599629
- remark: 修改 DoubleConv 的第二個卷積層，3x3 改為 5x5

### unet_segmentation_8.pth
- train: <br>
dark-side-train-data-part1.zip<br>
dark-side-train-data-part2.zip<br>
dark-side-train-data-part3.zip<br>
dark-side-train-data-part4.zip<br>
dark-side-train-data-part5.zip<br>
dark-side-train-data-part6.zip<br>
dark-side-train-data-part7.zip<br>
dark-side-train-data-part8.zip<br>
dark-side-train-data-part9.zip<br>
dark-side-train-data-part10.zip<br>
dark-side-train-data-part11.zip<br>
dark-side-train-data-part12.zip<br>
dark-side-train-data-part13.zip<br>
dark-side-train-data-part14.zip<br>
dark-side-train-data-part15.zip<br>
dark-side-train-data-part16.zip<br>
dark-side-train-data-part17.zip<br>
dark-side-train-data-part18.zip<br>
dark-side-train-data-part419.zip<br>
dark-side-train-data-part20.zip<br>
- batch_size: 32
- epochs: 10
- socre: 0.688188

### unet_segmentation_9.pth
- train: <br>
dark-side-train-data-part1.zip<br>
dark-side-train-data-part2.zip<br>
dark-side-train-data-part3.zip<br>
dark-side-train-data-part4.zip<br>
dark-side-train-data-part5.zip<br>
dark-side-train-data-part6.zip<br>
dark-side-train-data-part7.zip<br>
dark-side-train-data-part8.zip<br>
dark-side-train-data-part9.zip<br>
dark-side-train-data-part10.zip<br>
dark-side-train-data-part11.zip<br>
dark-side-train-data-part12.zip<br>
dark-side-train-data-part13.zip<br>
dark-side-train-data-part14.zip<br>
dark-side-train-data-part15.zip<br>
dark-side-train-data-part16.zip<br>
dark-side-train-data-part17.zip<br>
dark-side-train-data-part18.zip<br>
dark-side-train-data-part419.zip<br>
dark-side-train-data-part20.zip<br>
- batch_size: 32
- epochs: 20
- socre: 0689454

# Models
## model(1~6)
```py
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
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
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
```

## model(7~*)
```py
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
            nn.Conv2d(out_ch, out_ch, 5, padding=2), # 只改這邊
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
```