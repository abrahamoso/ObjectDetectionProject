import cv2
from eye_detection import detect_faces_and_eyes
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

    # Resize and grayscale the frame for faster processing
    scale = 0.5
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    eye_centers = detect_faces_and_eyes(gray, frame)

    # Calculate and display angles for each detected eye
    frame_center = (frame.shape[1] / 2, frame.shape[0] / 2)
    for eye_center in eye_centers:
        gun_angle = calculate_gun_angle(eye_center, frame_center)
        display_gun_angle(frame, gun_angle)
        print(f"Eye Center: {eye_center}, Gun Angle: {gun_angle}")

    # Show the output frame
    cv2.imshow("Eye Tracking with Gun Angle Simulation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
