import cv2
import torch
import numpy as np
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # gebruik tijdelijk yolov8n.pt om snel te testen

# 2. Laad MiDaS DPT_Hybrid model (correcte versie!)
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
midas.to(device).eval()

# 3. Juiste transform ophalen voor DPT_Hybrid
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform  # ❗ past bij DPT_Hybrid

# 4. Videobron
cap = cv2.VideoCapture(0)  # of pad naar .mp4 bestand

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO-predictie op frame
    results = yolo_model(frame, classes=[0])
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    # MiDaS input klaarzetten
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Depth prediction
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=input_image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    # Bounding boxes verwerken
    for (x1, y1, x2, y2) in boxes:
        if x2 > x1 and y2 > y1:
            crop = depth_map[y1:y2, x1:x2]
            crop = crop[(crop > 0) & (crop < 10000)]
            if crop.size > 0:
                depth = np.median(crop)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{depth:.2f} depth", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Toon resultaat
    cv2.imshow("YOLO + MiDaS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
