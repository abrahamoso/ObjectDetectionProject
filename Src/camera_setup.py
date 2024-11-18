import cv2

def initialize_camera(index=0, width=640, height=480, fps=30):
    """
    Initializes the camera with the given settings.
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise ValueError("Could not open camera.")
    
    return cap
