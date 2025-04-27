import os
import cv2
import albumentations as A
from tqdm import tqdm

# 원본 이미지 폴더
input_dir = 'dataset/train/images'
# 증강 이미지 저장할 폴더
output_dir = 'dataset/train/images_aug'

os.makedirs(output_dir, exist_ok=True)

# 증강 파이프라인 정의
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Rotate(limit=10, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.Resize(height=640, width=640)
])

# input_dir 안에 있는 모든 파일을 불러와서 증강
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in tqdm(image_files, desc="증강 중"):
    img_path = os.path.join(input_dir, img_file)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 증강 적용
    augmented = transform(image=image)
    augmented_image = augmented['image']

    # 다시 BGR로 변환해서 저장 (OpenCV는 BGR 저장)
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

    # 파일명 변경해서 저장
    save_path = os.path.join(output_dir, f"aug_{img_file}")
    cv2.imwrite(save_path, augmented_image)

print("✅ 모든 증강 완료!")
