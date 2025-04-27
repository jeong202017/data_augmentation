import os
import shutil
import random

# ✅ 이미지 + 라벨이 들어 있는 폴더들
input_dirs = [
    "0423_left_cam", "0423_left_cam_light_on",
    "0423_right_cam", "0423_right_cam_light_on"
]

# ✅ 비율 설정
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# ✅ 결과 저장 폴더 (YOLO 형식에 맞게)
output_base = "dataset"
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(output_base, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, "labels"), exist_ok=True)

# ✅ 이미지 리스트 수집 (.jpg, .png)
image_paths = []
valid_exts = (".jpg", ".jpeg", ".png")
for folder in input_dirs:
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.lower().endswith(valid_exts):
                full_path = os.path.join(folder, file)
                # 새 이름: 폴더명_원래파일명 (예: 0423_left_cam_frame_0001.jpg)
                new_filename = f"{folder}_{file}"
                image_paths.append((full_path, new_filename))

print(f"✅ 총 이미지 수: {len(image_paths)}")

# ✅ 셔플 및 분할
random.shuffle(image_paths)
n_total = len(image_paths)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)
n_test = n_total - n_train - n_val

split_data = {
    "train": image_paths[:n_train],
    "val": image_paths[n_train:n_train + n_val],
    "test": image_paths[n_train + n_val:]
}

# ✅ 복사 수행
for split, paths in split_data.items():
    for img_path, new_filename in paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(os.path.dirname(img_path), base + ".txt")

        # 이미지 복사 (새 파일명)
        dst_img = os.path.join(output_base, split, "images", new_filename)
        shutil.copyfile(img_path, dst_img)

        # 라벨 복사 (새 파일명, 확장자만 바꿔서)
        if os.path.exists(label_path):
            dst_lbl_name = os.path.splitext(new_filename)[0] + ".txt"
            dst_lbl = os.path.join(output_base, split, "labels", dst_lbl_name)
            shutil.copyfile(label_path, dst_lbl)
        else:
            print(f"⚠️ 라벨 누락: {label_path}")

print("✅ 모든 이미지와 라벨을 고유 이름으로 변경하여 YOLO 형식으로 분할 완료!")

