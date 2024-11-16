import cv2
from eye_detection import detect_faces_and_objects
from angle_calculation import calculate_gun_angle
from display import display_gun_angle

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for processing
    scale = 0.3
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Detect objects in the frame
    detected_objects = detect_faces_and_objects(frame)

    # Annotate the frame with detected objects
    for obj in detected_objects:
        x, y, w, h = obj["box"]
        label = f"{obj['class_name']}: {obj['confidence']:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Object Detection with Gun Angle Simulation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
