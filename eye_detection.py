import cv2

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_faces_and_eyes(gray_frame, color_frame):
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    eye_centers = []

    for (x, y, w, h) in faces:
        cv2.rectangle(color_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = color_frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_center = (int(ex + ew / 2), int(ey + eh / 2))
            cv2.circle(roi_color, eye_center, 5, (0, 255, 255), -1)
            eye_centers.append(eye_center)

    return eye_centers
