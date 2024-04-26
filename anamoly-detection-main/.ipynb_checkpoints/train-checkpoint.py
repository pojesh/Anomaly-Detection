from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
model.train(data='F:\\Fall Sem 2023-24\\DSA\\DSA Project\\yolovvvvvv\\final-db\\',
            epochs=10, imgsz=64) 
