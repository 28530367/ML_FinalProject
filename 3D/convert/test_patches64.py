import os
import glob
import numpy as np

# 1) 定義 padding 與切塊函式
def pad_to_multiples_of_64(volume):
    """
    將 volume 在 (D, H, W) 三個維度補到可被 64 整除的大小，使用 zero-padding。
    回傳 (vol_padded, pad_info)，其中 pad_info 記錄每個維度實際補了多少。
    """
    D, H, W = volume.shape

    pad_D = ((0, 64 - (D % 64))) if (D % 64) != 0 else (0, 0)
    pad_H = ((0, 64 - (H % 64))) if (H % 64) != 0 else (0, 0)
    pad_W = ((0, 64 - (W % 64))) if (W % 64) != 0 else (0, 0)

    vol_padded = np.pad(
        volume,
        (pad_D, pad_H, pad_W),  # ((top, bottom), (left, right), (front, back))
        mode='constant',
        constant_values=0
    )
    return vol_padded, (pad_D, pad_H, pad_W)

def extract_64_cubes(volume, stride=64):
    """
    從已經在 (D, H, W) 三維皆可被64整除的 volume 中，
    以 stride (預設64，無重疊) 做 3D sliding window，
    回傳 shape 為 (N, 64, 64, 64) 的 numpy array。
    """
    patches = []
    D, H, W = volume.shape
    for z in range(0, D, stride):
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                patch = volume[z:z+64, y:y+64, x:x+64]
                patches.append(patch)
    return np.array(patches, dtype=volume.dtype)

# 2) 設定路徑
test_root = "/media/disk2/HSW/MachineLearning/Final_project/2D/convert/test_original"
# 假設要把切好的 patch 存到以下目錄
output_root = "/media/disk2/HSW/MachineLearning/Final_project/3D/convert/test_patches"
os.makedirs(output_root, exist_ok=True)

# 3) 取得所有子資料夾 (50個)
subfolders = sorted(glob.glob(os.path.join(test_root, "*")))  
# 注意：確保這些子資料夾內才是 .npy (300,300,1259)

# 4) 開始遍歷
for subf in subfolders:
    if not os.path.isdir(subf):
        continue  # 如果不是資料夾就跳過

    # 取資料夾名稱，例如 "test_original/001" -> folder_name = "001"
    folder_name = os.path.basename(subf)

    # 該資料夾裡面只有一個 .npy，shape應該是 (300,300,1259)
    npy_files = glob.glob(os.path.join(subf, "*.npy"))
    if len(npy_files) == 0:
        print(f"[警告] 在資料夾 {folder_name} 找不到 .npy 檔案，跳過。")
        continue

    test_path = npy_files[0]  # 只取第一個檔案

    # 1) 讀取 .npy 檔案
    volume = np.load(test_path)  # shape 通常是 (300,300,1259)

    # 2) pad 到 64的倍數
    vol_pad, pad_info = pad_to_multiples_of_64(volume)

    # 3) extract patches (64,64,64) * N
    patches_img = extract_64_cubes(vol_pad, stride=64)

    # 4) 將這些 patch 儲存到 output_root
    #    可以選擇將每個 patch 存成單一檔案，也可以把所有 patch 存成一個 .npz
    #    以下示範「每個 patch 存一個檔案」的做法，並把它們放到對應子資料夾中
    out_subdir = os.path.join(output_root, folder_name)
    os.makedirs(out_subdir, exist_ok=True)

    for i, pimg in enumerate(patches_img):
        # 檔名格式： {子資料夾名稱}_patch{i}.npz
        out_name = f"{folder_name}_patch{i:03d}.npz"
        out_path = os.path.join(out_subdir, out_name)

        # 儲存為壓縮 npz，裡面包含 'image'
        np.savez_compressed(out_path, image=pimg)
    
    print(f"[完成] 資料夾 {folder_name} => 產生 {len(patches_img)} 個 (64,64,64) patch。")
