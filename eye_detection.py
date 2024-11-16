import cv2

def detect_faces_and_objects(frame):
    # Paths to the YOLO configuration and class files
    config_path = "config/yolov3.cfg"
    weights_path = "yolov3.weights"  # Use your actual file location
 
    names_path = "config/coco.names"

    # Load YOLO model
    net = cv2.dnn.readNet(weights_path, config_path)

    # Load class labels
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get detections
    layer_names = net.getLayerNames()
    
    if isinstance(net.getUnconnectedOutLayers(), list):
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    else:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    detections = net.forward(output_layers)

    # Parse YOLO detections
    height, width = frame.shape[:2]
    boxes = []
    confidences = []
    class_ids = []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = int(detection[4])  # Confidence of this detection
            confidence = scores[class_id]
            if confidence > 0.3:  # Adjust confidence threshold as needed
                box = detection[0:4] * [width, height, width, height]
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detected_objects.append({
                "box": (x, y, w, h),
                "confidence": confidences[i],
                "class_id": class_ids[i],
                "class_name": classes[class_ids[i]]
            })

    return detected_objects
