"""
Universal YOLO Detector:
- Webcam detection
- Image folder detection
- Video folder detection
"""

import os
import argparse
import cv2
from ultralytics import YOLO


# DRONE ALERT STYLE

def draw_drone(frame, xyxy, conf):
    x1, y1, x2, y2 = map(int, xyxy)
    label = f"drone {conf:.2f}"

    red = (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), red, 3)

    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2)



# GENERAL OBJECT BOX

def draw_object(frame, xyxy, cls_name, conf):
    x1, y1, x2, y2 = map(int, xyxy)
    label = f"{cls_name} {conf:.2f}"

    green = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), green, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)


# PROCESS FRAME FOR ALL SOURCES

def detect_frame(model, frame, imgsz, conf, drone_only=False):
    results = model(frame, imgsz=imgsz, conf=conf, iou=0.6)

    for r in results:
        names = r.names
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            cls_id = int(b.cls[0])
            conf_score = float(b.conf[0])
            cls_name = names[cls_id].lower()

            is_drone = cls_name in ["drone", "quadcopter", "uav"]

            if drone_only:
                if is_drone:
                    draw_drone(frame, xyxy, conf_score)
                continue

            if is_drone:
                draw_drone(frame, xyxy, conf_score)
            else:
                draw_object(frame, xyxy, names[cls_id], conf_score)

    return frame


# WEBCAM MODE 

def run_webcam(model, imgsz, conf, drone_only):
    print("🎥 Starting webcam detection...")
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_frame(model, frame, imgsz, conf, drone_only)

        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# IMAGE FOLDER

def run_images(model, folder, imgsz, conf, drone_only):
    os.makedirs("runs/output_images", exist_ok=True)

    imgs = [f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in imgs:
        path = os.path.join(folder, img_name)
        img = cv2.imread(path)

        result = detect_frame(model, img, imgsz, conf, drone_only)

        cv2.imwrite(f"runs/output_images/DET_{img_name}", result)

    print(" Image detection complete!")


# VIDEO FOLDER

def run_videos(model, folder, imgsz, conf, drone_only):
    os.makedirs("runs/output_videos", exist_ok=True)

    vids = [f for f in os.listdir(folder)
            if f.lower().endswith((".mp4", ".avi", ".mkv"))]

    cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)

    for vid in vids:
        cap = cv2.VideoCapture(os.path.join(folder, vid))

        w = int(cap.get(3))
        h = int(cap.get(4))
        fps = int(cap.get(5))

        out = cv2.VideoWriter(
            f"runs/output_videos/DET_{vid}",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect_frame(model, frame, imgsz, conf, drone_only)

            out.write(frame)

            cv2.resizeWindow("Video Detection", w, h)
            cv2.imshow("Video Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()

    print("🎬 Video detection complete!")


# MAIN

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", required=True)
    parser.add_argument("--task", default="webcam",
                        choices=["webcam", "images", "videos"])
    parser.add_argument("--source", default="test-images")
    parser.add_argument("--drone_only", action="store_true")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)

    args = parser.parse_args()

    print("\n========= UNIVERSAL DETECTOR =========")
    print("Weights:", args.weights)
    print("Task:", args.task)
    print("Drone Only:", args.drone_only)
    print("======================================\n")

    model = YOLO(args.weights)

    if args.task == "webcam":
        run_webcam(model, args.imgsz, args.conf, args.drone_only)

    elif args.task == "images":
        run_images(model, args.source, args.imgsz, args.conf, args.drone_only)

    elif args.task == "videos":
        run_videos(model, args.source, args.imgsz, args.conf, args.drone_only)


if __name__ == "__main__":
    main()
