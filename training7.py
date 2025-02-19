import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO('yolov8n.pt')

motorcycle_id = 3
car_id = 2
bus_id = 5
truck_id = 7


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


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(frame)
    

    cv2.putText(frame, f"Motorcycles: {motorcycle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Cars: {car_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Trucks/Buses: {truck_bus_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Green Light Time: {green_time:.2f} sec", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    

    cv2.imshow("Live Traffic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
