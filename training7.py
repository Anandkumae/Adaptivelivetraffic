import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

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

    results = model(frame)

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

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(frame)
    
    # Display processed image
    st.image(frame, channels="BGR", caption="Uploaded Image with Detections")
    
    # Show detection results
    st.write(f"Green Light Time: {green_time:.2f} sec")
    st.write(f"Motorcycles: {motorcycle_count}")
    st.write(f"Cars: {car_count}")
    st.write(f"Trucks/Buses: {truck_bus_count}")
