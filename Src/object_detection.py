from ultralytics import YOLO

def load_yolo_model(model_path='yolov5n.pt', device='cuda'):
    """
    Loads the YOLO model.
    """
    model = YOLO(model_path)
    model.to(device)
    return model

def run_yolo_detection(model, frame, target_classes):
    """
    Runs YOLO on a frame and filters detections by target classes.
    """
    results = model(frame)
    filtered_detections = [
        box for box in results[0].boxes if int(box.cls[0]) in target_classes
    ]
    return results, filtered_detections
