import cv2
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

input_folder = "raw_images"       # map met onbewerkte foto's
output_folder = "yolo_dataset/images"   # hier komen je foto's
label_folder = "yolo_dataset/labels"    # hier komen YOLO labels

os.makedirs(output_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if not file.lower().endswith((".jpg", ".png")):
        continue

    path = os.path.join(input_folder, file)
    image = cv2.imread(path)
    height, width, _ = image.shape

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        boxes = []

        for hand_landmarks in results.multi_hand_landmarks:
            # Vind de bounding box rond de hand
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = max(min(x_coords), 0)
            x_max = min(max(x_coords), 1)
            y_min = max(min(y_coords), 0)
            y_max = min(max(y_coords), 1)

            # YOLO-format: class_id x_center y_center width height
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            box_width = x_max - x_min
            box_height = y_max - y_min

            boxes.append([0, x_center, y_center, box_width, box_height])

        # Sla afbeelding + labels op
        out_img_path = os.path.join(output_folder, file)
        out_label_path = os.path.join(label_folder, os.path.splitext(file)[0] + ".txt")
        cv2.imwrite(out_img_path, image)

        with open(out_label_path, "w") as f:
            for box in boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
