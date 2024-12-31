import numpy as np
import os
from PIL import Image

# 定義資料夾路徑
input_dir = "/media/disk2/HSW/MachineLearning/Final_project/2D/convert/test_original"  # 替換為您的主資料夾路徑
output_dir = "/media/disk2/HSW/MachineLearning/Final_project/2D/test/imgs"  # 保存圖片的資料夾名稱

# 創建保存圖片的主資料夾
os.makedirs(output_dir, exist_ok=True)

# Count
k = 0

# 遍歷資料夾中的所有檔案
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.startswith("seismicCubes") and file_name.endswith(".npy"):
            k += 1

            # 構建完整的文件路徑
            file_path = os.path.join(root, file_name)
            
            # 獲取外層資料夾名稱
            parent_dir_name = os.path.basename(root)

            # 加載數據
            data = np.load(file_path)
            num_slices = data.shape[-1]  # 切片數量
            
            # 為當前文件創建一個子資料夾
            file_output_dir = os.path.join(output_dir, os.path.splitext(f"{parent_dir_name}")[0])
            os.makedirs(file_output_dir, exist_ok=True)
            
            # 遍歷每個切片並保存為圖片
            for i in range(num_slices):
                slice_data = data[:, :, i]
                
                # 歸一化到 [0, 255] 並轉換為 8 位無符號整數
                normalized_slice = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
                image = Image.fromarray(normalized_slice.astype(np.uint8))
                
                # 保存圖片
                image.save(os.path.join(file_output_dir, f"slice_{i+1:04d}.png"))
            
            print(f"已處理檔案: {file_name}，共保存 {num_slices} 張切片圖片到資料夾 '{file_output_dir}'")

print("所有檔案處理完成！")
