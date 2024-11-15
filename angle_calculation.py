def calculate_gun_angle(eye_center, frame_center):
    offset_x = eye_center[0] - frame_center[0]
    horizontal_angle = (offset_x / frame_center[0]) * 45  # Adjust if necessary

    gun_angle = int((horizontal_angle + 45) * 2)  # Maps to 0-180 degrees
    return gun_angle
