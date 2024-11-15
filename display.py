import cv2

def display_gun_angle(frame, gun_angle):
    cv2.putText(frame, f"Gun Angle: {gun_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
