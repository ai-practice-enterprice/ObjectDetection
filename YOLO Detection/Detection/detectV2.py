import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

# Load Model
model = YOLO("RobotDetectionv8Large.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Callback-functie voor het verwerken van frames
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    
    # Annotatie toevoegen aan het frame
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    return label_annotator.annotate(annotated_frame, detections=detections, labels=labels)


# Open Webcam
cap = cv2.VideoCapture("./sample1.mp4")  # 0 = webcam

if not cap.isOpened():
    print("Can't open webcam/video!")
    exit()

# Save videp (if needed in a file)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("result.mp4", fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Manage frame with callback function
    processed_frame = callback(frame, 0)  # index = 0 but can be different

    # show processed frame
    cv2.imshow("Webcam", processed_frame)

    # save video
    out.write(processed_frame)

    # Stop 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything
cap.release()
out.release()
cv2.destroyAllWindows()