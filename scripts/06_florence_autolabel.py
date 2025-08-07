import os
from transformers import AutoProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

# === SETTINGS ===
MODEL_NAME = "microsoft/florence-2-base"
INPUT_FOLDER = "new_dataset_for_florence"
OUTPUT_FOLDER = "florence_labels"
LABEL_CLASS = "window"  # you can modify this prompt

# === LOAD MODEL & PROCESSOR ===
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME).to(device)

# === PREPARE OUTPUT ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === PROCESS EACH IMAGE ===
for fname in os.listdir(INPUT_FOLDER):
    if not fname.lower().endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(INPUT_FOLDER, fname)
    image = Image.open(img_path).convert("RGB")
    W, H = image.size

    inputs = processor(images=image, text=LABEL_CLASS, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=torch.tensor([[H, W]])).pop()

    label_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(fname)[0] + ".txt")
    with open(label_path, "w") as f:
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x_min, y_min, x_max, y_max = box.tolist()
            x_center = (x_min + x_max) / 2 / W
            y_center = (y_min + y_max) / 2 / H
            width = (x_max - x_min) / W
            height = (y_max - y_min) / H
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"✅ Labeled: {fname}")

print(f"\n🏁 Done. Labels saved in `{OUTPUT_FOLDER}`.")
