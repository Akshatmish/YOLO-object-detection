**Drone Detection Using YOLOv8n (COCO Pretrained + Custom Training)
**
This project demonstrates real-time drone detection using YOLOv8n starting from a COCO-pretrained model (yolov8n.pt) and then training on a custom drone dataset to produce the final trained model best.pt.

**Features
**
Real-time drone detection using YOLOv8n

Starts from COCO-pretrained yolov8n.pt

Fine-tuned on a custom drone dataset → output model: best.pt

Supports image, video, and webcam inference

Exports models to ONNX, TensorRT, CoreML, etc.


**Training YOLOv8n on Custom Drone Dataset
Using COCO Pretrained YOLOv8n.pt**

yolo detect train \model=yolov8n.pt \data=data.yaml \epochs=50 \imgsz=640


**Output Model:**

runs/detect/train/weights/best.pt

**Inference (Using best.pt)
**
**Predict on an image:
**
yolo detect predict \
    model=runs/detect/train/weights/best.pt \
    source="test.jpg"

**Predict on a video:
**
yolo detect predict \
    model=runs/detect/train/weights/best.pt \
    source="drone_video.mp4"

    
**Real-time webcam detection:
**
yolo detect predict \
    model=runs/detect/train/weights/best.pt \
    source=0

