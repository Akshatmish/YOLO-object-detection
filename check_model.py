from ultralytics import YOLO

model = YOLO("C:/Users/Akshat/drone-detection/Unmanned-Aerial-Vehicle/yolov8n.pt")
print("\n🟦 Model Classes:")
print(model.names)
