import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

model = YOLO("runs/detect/train/weights/best.pt")

st.title("🛸 Drone Detection using YOLOv8")
st.write("Upload an image or use your webcam for real-time drone detection.")

# Image Detection
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    results = model(image_np)
    annotated = results[0].plot()

    st.image(annotated, caption="Detected Output")

# Webcam Detection
st.subheader("Live Webcam Detection")
run_webcam = st.checkbox("Start Webcam")

if run_webcam:
    cam = cv2.VideoCapture(0)

    st_frame = st.empty()

    while run_webcam:
        ret, frame = cam.read()
        if not ret:
            st.write("Webcam not detected.")
            break

        results = model(frame)
        annotated = results[0].plot()

        st_frame.image(annotated, channels="BGR")
else:
    st.write("Turn on the checkbox to start webcam detection.")
