import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import tempfile
import os
import math

# Load the model from Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="samoooo/yolo-lane", filename="Copy of best.pt")
    model = YOLO(model_path)
    return model

model = load_model()

# Streamlit interface title
st.title('رصد مخالفات للمركبات التي لا تترك مسافة امنة')

# Arabic text for file uploader
uploaded_file = st.file_uploader("قم برفع فيديو لمراقبة المركبات", type=["mp4", "avi", "mov"])

# Parameters
safe_distance_threshold_meters = 2
calibration_factor = 0.0175
lane_threshold = 50

if uploaded_file is not None:
    # Temporary file to store the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Capture video from the uploaded file
    cap = cv2.VideoCapture(tfile.name)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to save the processed video
    output_path = 'processed_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    stframe = st.empty()

    # Processing video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        centroids = []
        boxes_list = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, classes):
                if conf > 0.3:
                    x1, y1, x2, y2 = map(int, box)
                    class_name = 'IN LANE' if int(cls) == 0 else 'OUT OF LANE'

                    centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                    centroids.append(centroid)
                    boxes_list.append((x1, y1, x2, y2, class_name))

        # Calculate the closest distance between cars
        min_distance = float('inf')
        closest_pair = None

        for i, centroid1 in enumerate(centroids):
            for j, centroid2 in enumerate(centroids):
                if i != j and abs(centroid1[0] - centroid2[0]) < lane_threshold:
                    pixel_distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
                    real_distance_meters = pixel_distance * calibration_factor

                    if real_distance_meters < min_distance and centroid1[1] < centroid2[1]:
                        min_distance = real_distance_meters
                        closest_pair = (i, j)

        # Draw the boxes and labels
        for i, (x1, y1, x2, y2, class_name) in enumerate(boxes_list):
            color = (0, 255, 0) if class_name == 'IN LANE' else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            if i < len(centroids):
                cv2.circle(frame, centroids[i], 10, (255, 0, 0), -1)

            if closest_pair and i in closest_pair:
                rear_car_index = closest_pair[1] if centroids[closest_pair[0]][1] < centroids[closest_pair[1]][1] else closest_pair[0]
                if i == rear_car_index and min_distance < safe_distance_threshold_meters:
                    color = (0, 0, 255)
                    label = "Violating Safe Distance"
                    cv2.line(frame, centroids[closest_pair[0]], centroids[closest_pair[1]], (0, 255, 255), 3)
                    midpoint = ((centroids[closest_pair[0]][0] + centroids[closest_pair[1]][0]) // 2,
                                 (centroids[closest_pair[0]][1] + centroids[closest_pair[1]][1]) // 2)
                    cv2.putText(frame, f"Dist: {min_distance:.2f}m", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # Show frame
        stframe.image(frame, channels="BGR")  # Use "BGR" if the colors are not correct

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()

    # Check if the video file exists
    if os.path.exists(output_path):
        st.success("تم حفظ الفيديو المعالج بنجاح!")


        # Display the processed video
        st.subheader("فيديو المخالفات المكتشفة")
        st.video(output_path)  # Show the processed video
    else:
        st.error("فشل في حفظ الفيديو المعالج.")