import os
import numpy as np
import cv2

# === 配置部分 ===
dataset_id = 23
input_folder = fr"D:\dataset\Frame_Event_Dataset\dataset_{dataset_id}\Frames\frame{dataset_id}"  # 输入 .raw 文件夹路径
output_folder = fr"D:\dataset\Frame_Event_Dataset\dataset_{dataset_id}\Frames"  # 输出 .png 保存路径
width = 1816         # 原始图像宽度（像素）
height = 1020         # 原始图像高度（像素）
crop_width = 1803     # 裁剪后的图像宽度（像素）
crop_height = 1014    # 裁剪后的图像高度（像素）
resize_width = 1280   # resize为与事件相同的分辨率
resize_height = 720
pixel_format = "Mono8"  # 像素格式

# === 灵活裁剪参数 ===
crop_top = 15      # 裁掉上方像素数s
crop_bottom = 0  # 裁掉下方像素数
crop_left = 6     # 裁掉左方像素数
crop_right = 7    # 裁掉右方像素数

# =====================

# === 创建输出文件夹 ===
os.makedirs(output_folder, exist_ok=True)

# === 遍历文件夹 ===
for i, file_name in enumerate(os.listdir(input_folder)):
    if not file_name.lower().endswith(".raw"):
        continue

    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(
        output_folder,
        os.path.splitext(file_name)[0].split("_")[0] + '_' + str(i) + ".png"
    )

    # === 读取 RAW 数据 ===
    raw_data = np.fromfile(input_path, dtype=np.uint8)
    expected_size = width * height

    if raw_data.size != expected_size:
        print(f"[警告] {file_name} 尺寸不匹配，跳过。({raw_data.size} != {expected_size})")
        continue

    # === 重塑为二维灰度图 ===
    img = raw_data.reshape((height, width))

    # === 新的灵活裁剪逻辑 ===
    new_top = crop_top
    new_bottom = height - crop_bottom
    new_left = crop_left
    new_right = width - crop_right

    if new_top < new_bottom and new_left < new_right:
        img_cropped = img[new_top:new_bottom, new_left:new_right]
    else:
        print(f"[警告] 裁剪参数错误，跳过裁剪。")
        img_cropped = img

    # === resize 到目标分辨率 ===
    if (resize_width != img_cropped.shape[1]) or (resize_height != img_cropped.shape[0]):
        img_resized = cv2.resize(img_cropped, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_cropped

    # === 保存为 PNG ===
    cv2.imwrite(output_path, img_resized)
    print(f"[完成] {file_name} → {output_path} (尺寸 {img_resized.shape[1]}x{img_resized.shape[0]})")

print("所有 RAW 文件已成功转换、裁剪并保存为 PNG！")
