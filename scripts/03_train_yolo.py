from ultralytics import YOLO

# === CONFIGURATION ===
model_arch = 'yolov8n.pt'
data_yaml = 'datasets/yolo_dataset/data.yaml'
imgsz = 640 # Image size for training 640x640
epochs = 50
batch = 8
project = 'yolo_training_output' # Output directory for training results
name = 'yolov8n_window_detector' # Name of this training run

# === START TRAINING ===
model = YOLO(model_arch)
model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, project=project, name=name)

# Optional: print best metrics
metrics = model.val()
print(metrics)
