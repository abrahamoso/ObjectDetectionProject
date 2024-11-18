import cv2
from Src.camera_setup import initialize_camera
from Src.object_detection import load_yolo_model, run_yolo_detection

# Initialize settings
camera_index = 1
camera_width = 640
camera_height = 480
camera_fps = 30
target_classes = [0, 67]  # Person=0, Phone=67

# Initialize camera and YOLO
cap = initialize_camera(camera_index, camera_width, camera_height, camera_fps)
model = load_yolo_model('yolov5nu.pt', device='cuda')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results, filtered_detections = run_yolo_detection(model, frame, target_classes)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv5 Object Detection", annotated_frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
