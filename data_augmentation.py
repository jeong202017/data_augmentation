import os
import cv2
import albumentations as A
from tqdm import tqdm

# 경로 설정
image_dir = 'dataset/train/images'
label_dir = 'dataset/train/labels'
output_image_dir = 'dataset/train/images_aug'
output_label_dir = 'dataset/train/labels_aug'

# 저장 폴더 생성
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 증강 파이프라인 정의
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Rotate(limit=10, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.Resize(height=640, width=640)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 이미지 파일 리스트
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in tqdm(image_files, desc="증강 중"):
    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

    # 이미지 읽기
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 바운딩 박스 읽기
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)

    # 증강 적용
    if bboxes:
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']
    else:
        # 라벨이 없는 경우에도 증강
        transformed = transform(image=image)
        transformed_image = transformed['image']
        transformed_bboxes = []
        transformed_labels = []

    # 저장
    save_img_path = os.path.join(output_image_dir, f"aug_{img_file}")
    save_label_path = os.path.join(output_label_dir, f"aug_{img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')}")

    # 이미지 저장
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_img_path, transformed_image)

    # 라벨 저장
    with open(save_label_path, 'w') as f:
        for bbox, label in zip(transformed_bboxes, transformed_labels):
            x_center, y_center, width, height = bbox
            f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ 이미지 + 라벨 모두 증강 완료!")

