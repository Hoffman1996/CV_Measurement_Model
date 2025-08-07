import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

def visualize_yolo_labels(image_dir, label_dir, num_images=6):
    image_paths = list(Path(image_dir).glob("*.jpg"))
    image_paths = random.sample(image_paths, min(num_images, len(image_paths)))

    for img_path in image_paths:
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.strip().split())
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Class {int(cls)}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(img_path.name)
        plt.axis('off')
        plt.show()

# Change these paths if needed
visualize_yolo_labels(
    image_dir="datasets/yolo_dataset/images/train",
    label_dir="datasets/yolo_dataset/labels/train"
)
visualize_yolo_labels(
    image_dir="datasets/yolo_dataset/images/val",
    label_dir="datasets/yolo_dataset/labels/val"
)
