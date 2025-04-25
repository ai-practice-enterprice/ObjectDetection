import os
import shutil
from ultralytics import YOLO
import cv2

# 🔧 Settings
model_path = "./my_model.pt"
photos_folder = "./TrainPhotos"
no_detection_folder = "./NoDetections"
confidence_threshold = 0.6
start_batch_index = 34  # 👈 only start from batch_015 and up

# 📦 Load model
model = YOLO(model_path)

# 📂 Create output folder if it doesn't exist
os.makedirs(no_detection_folder, exist_ok=True)

# 🔁 Walk through subfolders
for root, dirs, files in os.walk(photos_folder):
    # Check if we're inside a batch folder
    folder_name = os.path.basename(root)
    if folder_name.startswith("batch_"):
        try:
            batch_number = int(folder_name.split("_")[1])
        except (IndexError, ValueError):
            print(f"Skipping folder (invalid name): {folder_name}")
            continue

        if batch_number < start_batch_index:
            print(f"⏩ Skipping {folder_name} (before batch_{start_batch_index:03d})")
            continue

    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(root, file)
        image = cv2.imread(image_path)

        results = model(image)[0]
        boxes = results.boxes

        if len(boxes) == 0:
            print(f"❌ No detection: {file}")
            shutil.move(image_path, os.path.join(no_detection_folder, file))
            continue

        label_lines = []
        low_confidence_boxes = []

        h, w = image.shape[:2]
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]

            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            line = f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

            if conf >= confidence_threshold:
                label_lines.append(line)
            else:
                low_confidence_boxes.append((int(x1), int(y1), int(x2), int(y2), line))

        if label_lines:
            label_path = os.path.splitext(image_path)[0] + ".txt"
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))
            print(f"✅ Labeled automatically: {file}")
            continue

        if low_confidence_boxes:
            preview = image.copy()
            for x1, y1, x2, y2, _ in low_confidence_boxes:
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Resize preview for screen
            max_dim = 900
            scale = min(max_dim / w, max_dim / h)
            resized = cv2.resize(preview, (int(w * scale), int(h * scale)))

            cv2.imshow("Low confidence - accept [y] / reject [n]", resized)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord('y'):
                label_path = os.path.splitext(image_path)[0] + ".txt"
                with open(label_path, "w") as f:
                    for _, _, _, _, line in low_confidence_boxes:
                        f.write(line + "\n")
                print(f"✅ Labeled (user confirmed): {file}")
            else:
                shutil.move(image_path, os.path.join(no_detection_folder, file))
                print(f"➡️ Rejected by user — moved to NoDetections: {file}")
