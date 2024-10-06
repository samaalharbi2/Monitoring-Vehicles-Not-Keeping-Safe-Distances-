import subprocess 
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from huggingface_hub import hf_hub_download
import tempfile 

# Suppressing output from subprocess
subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Load your logo image
logo_path = "Screen_Shot_1446-04-03_at_9.24.57_PM-removebg-preview.png"

# Add logo to the top left
col1, col2 = st.columns([1, 3])  # Adjust columns based on your layout
with col1:
    st.image(logo_path, width=200)  # Adjust the width as needed
with col2:
    st.header("Safe Distance Detection")



# Load the trained YOLO model from Hugging Face Hub
model_path = hf_hub_download(repo_id="samoooo/yolo-lane", filename="Copy of best.pt")
model = YOLO(model_path)

# Define class names
class_names = {0: 'IN LANE', 1: 'OUT OF LANE'}

# Upload video file (Single Upload Button)
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(tfile.name)

    # Check if video loaded successfully
    if not cap.isOpened():
        st.error("Error: Could not open video.")
    else:
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video path
        output_path = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Define safe distance and calibration factor
        safe_distance_threshold_meters = 2
        calibration_factor = 0.0175  # Adjust as needed
        lane_threshold = 50

        stframe = st.empty()  # Placeholder for displaying video frames

        # Process video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB and process with YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            centroids = []
            boxes_list = []

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confidences, classes):
                    if conf > 0.3:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, box)
                        class_name = class_names.get(int(cls), 'Unknown')

                        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                        centroids.append(centroid)
                        boxes_list.append((x1, y1, x2, y2, class_name))

            # Calculate minimum distance between cars in the same lane
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

            # Draw results on frame
            for i, (x1, y1, x2, y2, class_name) in enumerate(boxes_list):
                color = (0, 255, 0) if class_name == 'IN LANE' else (0, 165, 255)
                label = "IN LANE" if class_name == 'IN LANE' else "OUT OF LANE"

                if closest_pair and i in closest_pair:
                    # تعديل تحديد السيارة الخلفية بناءً على الموقع بالنسبة للمحور y
                    rear_car_index = closest_pair[1] if centroids[closest_pair[0]][1] > centroids[closest_pair[1]][1] else closest_pair[0]
                    if i == rear_car_index and min_distance < safe_distance_threshold_meters:
                        color = (0, 0, 255)  # Red for violation
                        label = "Violating Safe Distance"
                        cv2.line(frame, centroids[closest_pair[0]], centroids[closest_pair[1]], (0, 255, 255), 3)
                        midpoint = ((centroids[closest_pair[0]][0] + centroids[closest_pair[1]][0]) // 2,
                                     (centroids[closest_pair[0]][1] + centroids[closest_pair[1]][1]) // 2)
                        cv2.putText(frame, f"Dist: {min_distance:.2f}m", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                cv2.circle(frame, centroids[i], 10, (255, 0, 0), -1)

            # Display the current frame
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Write frame to output video
            out.write(frame)

        # Release resources
        cap.release()
        out.release()

        # Display the output video
        st.success("Processing complete. Displaying output video:")
        st.video(output_path)
else:
    st.write("Please upload a video to start processing.") 