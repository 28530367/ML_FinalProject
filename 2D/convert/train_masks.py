import numpy as np
import os
from PIL import Image

# 定義資料夾路徑
input_dir = "/media/disk2/HSW/MachineLearning/Final_project/2D/convert/train_original"  # 替換為您的主資料夾路徑
output_dir = "/media/disk2/HSW/MachineLearning/Final_project/2D/train/masks"  # 保存圖片的資料夾名稱
record_file = "/media/disk2/HSW/MachineLearning/Final_project/2D/convert/train_masks-processed.txt" # 記錄處理過檔案的路徑

# 創建保存圖片的主資料夾
os.makedirs(output_dir, exist_ok=True)

# 讀取已處理過的檔名記錄
processed_files = set()
if os.path.exists(record_file):
    with open(record_file, "r") as f:
        for line in f:
            processed_files.add(line.strip())

# 遍歷資料夾中的所有檔案
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.startswith("fault") and file_name.endswith(".npy"):
             # 如果這個檔案已處理過，就跳過
            if file_name in processed_files:
                print(f"跳過已處理檔案: {file_name}")
                continue

            # 構建完整的文件路徑
            file_path = os.path.join(root, file_name)
            
            # 加載數據
            data = np.load(file_path)
            num_slices = data.shape[-1]  # 切片數量
            
            # 遍歷每個切片並保存為圖片
            for i in range(num_slices):
                slice_data = data[:, :, i]
                
                # 歸一化到 [0, 255] 並轉換為 8 位無符號整數
                normalized_slice = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
                image = Image.fromarray(normalized_slice.astype(np.uint8))
                
                # 構建唯一文件名
                output_file_name = f"{os.path.splitext(file_name)[0]}_slice_{i+1:04d}.png"
                image.save(os.path.join(output_dir, output_file_name))
            
            print(f"已處理檔案: {file_name}，共保存 {num_slices} 張切片圖片到資料夾 '{output_dir}'")

            # 將已處理的檔名加入紀錄中
            processed_files.add(file_name)

# 將所有已處理過的檔名寫回紀錄檔
with open(record_file, "w") as f:
    for pf in processed_files:
        f.write(pf + "\n")

print("所有檔案處理完成！")
