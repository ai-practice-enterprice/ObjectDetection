import os
import cv2
import torch
import time
import threading
import numpy as np
import cmapy
import supervision as sv
from ultralytics import YOLO
from midas_loader import load_midas_model, midas_predict, normalize_depth

# Models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YOLO("RobotDetectionv8Large.pt")
midas_model = load_midas_model("dpt_hybrid_384.pt", device)

# Tracking & Annotators
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Video Input (0 = webcam or use video file)
cap = cv2.VideoCapture("./sample1.mp4")
if not cap.isOpened():
    print("Video openen mislukt.")
    exit()

# FPS
prev_time = time.time()
frame_counter = 0

# Yolo every X frames and Midas every Y frames so they don't run at the same time
yolo_interval = 2000
midas_interval = 3000

# Memory for last results
last_yolo_detections = None
last_results = None
last_distances = {}
latest_frame = None

# Midas Threading
midas_lock = threading.Lock()
midas_thread_running = False

# Calculate the depth of the full image without disturbing the main loop
def run_midas_async(rgb_copy, detections_copy):
    global last_distances, midas_thread_running, colored_map_thread

    # Voorspel depthmap over het hele beeld
    depth_map = midas_predict(rgb_copy, midas_model, device)
    if depth_map is None or depth_map.size == 0:
        midas_thread_running = False
        return

    depth_map = normalize_depth(depth_map, bits=2)
    depth_map = 255 - ((depth_map / depth_map.max()) * 255).astype(np.uint8)

    # Bereken gemiddelde diepte per object
    local_distances = {}
    for i, box in enumerate(detections_copy.xyxy.astype(int)):
        if i >= len(detections_copy.tracker_id):
            continue
        x_min, y_min, x_max, y_max = box
        object_depth = depth_map[y_min:y_max, x_min:x_max]
        if object_depth.size == 0:
            continue
        distance_ratio = np.mean(object_depth) / 255
        distance_m = round(distance_ratio * 10, 2)
        tracker_id = int(detections_copy.tracker_id[i])
        local_distances[tracker_id] = distance_m

    # Kleur de hele depth map in
    colored_map = cv2.applyColorMap(depth_map, cmapy.cmap('coolwarm'))
    colored_map = colored_map.astype(np.float32) / 255.0

    with midas_lock:
        last_distances = local_distances
        colored_map_thread = colored_map
        midas_thread_running = False

colored_map_thread = None

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    frame_counter += 1

    frame = cv2.resize(frame, (640, 360))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO or Midas?
    run_yolo = frame_counter % yolo_interval == 0
    run_midas = frame_counter % midas_interval == 0 and not midas_thread_running

    # Run YOLO
    if run_yolo:
        results = model(rgb_frame)[0]
        last_results = results
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        last_yolo_detections = detections
    elif last_yolo_detections is not None:
        detections = tracker.update_with_detections(last_yolo_detections)
    else:
        continue

    # Run Midas
    if run_midas and last_results is not None:
        midas_thread_running = True
        rgb_copy = rgb_frame.copy()
        detections_copy = detections
        threading.Thread(target=run_midas_async, args=(rgb_copy, detections_copy)).start()

    base_frame = rgb_frame.astype(np.float32) / 255.0
    with midas_lock:
        blended = cv2.addWeighted(base_frame, 0.5, colored_map_thread if colored_map_thread is not None else np.zeros_like(base_frame), 0.5, 0)
    blended = (blended * 255).astype(np.uint8)

    # Draw Annotations
    annotated = box_annotator.annotate(blended.copy(), detections=detections)
    labels = []
    for i in range(len(detections.xyxy)):
        class_id = detections.class_id[i]
        tracker_id = int(detections.tracker_id[i]) if i < len(detections.tracker_id) else None
        distance_m = last_distances.get(tracker_id, None)
        distance_text = f" {distance_m}m" if distance_m is not None else ""
        label = f"#{tracker_id} {last_results.names[class_id]}{distance_text}" if tracker_id is not None else f"{last_results.names[class_id]}"
        labels.append(label)

    final = label_annotator.annotate(annotated, detections=detections, labels=labels)

    # Show FPS
    fps_text = f"{fps:.1f} FPS"
    cv2.putText(final, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Depth + YOLO + FPS + Distance", cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
