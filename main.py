import cv2
import torch
from ultralytics import YOLO
from angle_calculation import calculate_gun_angle
from display import display_gun_angle

# Check if GPU is available
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("Using CPU")
    device = "cpu"

# Load YOLOv5 model
model = YOLO('yolov5s.pt')  # Path to YOLOv5 weights (small model for speed)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for processing (optional)
    scale = 0.3
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Calculate the frame center
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)  # (width // 2, height // 2)

    # Detect objects in the frame using YOLOv5
    results = model(frame, device=device)  # Perform detection on the frame

    # Extract the center of the detected object (e.g., person or eye)
    if results[0].boxes:
        # Get the first detected box (adjust logic for specific detections)
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        eye_center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Center of the detected object
    else:
        eye_center = None  # No detection found

    # Calculate gun angle if an object is detected
    if eye_center:
        gun_angle = calculate_gun_angle(eye_center, frame_center)
        display_gun_angle(gun_angle, frame)  # Pass both gun_angle and frame

    # Render the detections onto the frame
    annotated_frame = results[0].plot()  # Use results[0].plot() to render annotations on the frame

    # Resize the annotated frame for a larger display
    display_scale = 2.0  # Adjust this to make the window bigger or smaller
    display_frame = cv2.resize(annotated_frame, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)

    # Show the resized frame
    cv2.imshow("YOLOv5 Object Detection with Gun Angle Simulation", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
