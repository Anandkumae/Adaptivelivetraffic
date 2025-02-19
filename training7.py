import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Load YOLO model
model = YOLO("yolov8n.pt")

# Class IDs for different vehicle types
motorcycle_id = 3
car_id = 2
bus_id = 5
truck_id = 7

# Weight factors for different vehicles
k_motorcycle = 1.0
k_car = 2.0
k_truck_bus = 3.0

def calculate_green_time(frame):
    if frame is None:
        return 10, 0, 0, 0

    # Run YOLO model on the frame
    results = model(frame)

    # Check if results are valid
    if len(results) == 0 or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return 10, 0, 0, 0

    detected_objects = results[0].boxes.data

    motorcycle_count = 0
    car_count = 0
    truck_bus_count = 0

    for obj in detected_objects:
        class_id = int(obj[5])
        if class_id == motorcycle_id:
            motorcycle_count += 1
        elif class_id == car_id:
            car_count += 1
        elif class_id in [bus_id, truck_id]:
            truck_bus_count += 1

    green_time = (k_motorcycle * motorcycle_count) + (k_car * car_count) + (k_truck_bus * truck_bus_count)
    green_time = max(10, min(90, green_time))

    return green_time, motorcycle_count, car_count, truck_bus_count

st.title("Live Traffic Detection")

# Option to choose between live video or image capture
mode = st.radio("Select Mode:", ("Live Video", "Capture Image"))

### 🚀 1️⃣ Live Video Streaming with WebRTC
if mode == "Live Video":
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0

        def transform(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Calculate green time and vehicle counts
            green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(img)

            # Add text overlay
            cv2.putText(img, f"Green Time: {green_time:.2f} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Motorcycles: {motorcycle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Cars: {car_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Trucks/Buses: {truck_bus_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Debugging: Print frame count and detection results
            st.sidebar.write(f"Frame {self.frame_count}:")
            st.sidebar.write(f"Green Time: {green_time:.2f} sec")
            st.sidebar.write(f"Motorcycles: {motorcycle_count}")
            st.sidebar.write(f"Cars: {car_count}")
            st.sidebar.write(f"Trucks/Buses: {truck_bus_count}")

            return img

    webrtc_streamer(
        key="traffic-detection",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,  # Enable asynchronous processing
    )

### 📸 2️⃣ Capture Image
elif mode == "Capture Image":
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(frame)

        # Display processed image
        st.image(frame, channels="BGR", caption="Captured Image with Detections")

        # Show detection results
        st.write(f"Green Light Time: {green_time:.2f} sec")
        st.write(f"Motorcycles: {motorcycle_count}")
        st.write(f"Cars: {car_count}")
        st.write(f"Trucks/Buses: {truck_bus_count}")
