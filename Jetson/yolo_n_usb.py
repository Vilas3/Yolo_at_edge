import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

# Check if CUDA is available and select the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the YOLOv8 model and move it to the GPU
model = YOLO("yolov8n.pt").to(device)  # Use .to(device) to move the model to GPU

# Open the USB camera (0 for default, change if multiple cameras are connected)
cap = cv2.VideoCapture(0)

# Set camera resolution (optional, adjust as needed)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Variables for FPS calculation
frame_count = 0
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Resize the frame to 640x640
    frame_resized = cv2.resize(frame, (640, 640))

    # Convert the frame to a tensor and move it to GPU
    frame_tensor = torch.from_numpy(frame_resized).float()  # Convert to float32
    frame_tensor = frame_tensor.permute(2, 0, 1)  # Convert from HWC to CHW
    frame_tensor /= 255.0  # Normalize the frame to [0, 1]
    frame_tensor = frame_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to GPU

    # Run YOLOv8 inference on the GPU
    results = model(frame_tensor)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Update frame count
    frame_count += 1

    # Calculate FPS
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  # Update FPS every second
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Display FPS on the frame
    cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output with FPS displayed
    cv2.imshow("YOLOv8 Detection with FPS", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
