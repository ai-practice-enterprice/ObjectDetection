import cv2
import os

def extract_frames(video_path, output_folder, target_fps):
    # Zorg dat de output map bestaat
    os.makedirs(output_folder, exist_ok=True)

    # Open de video
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Alleen frames opslaan op basis van target FPS
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"Eframe_{saved_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"frame count: {frame_count}")
    print(f"Frames opgeslagen in {output_folder}")


video_pad = "./sourceFolder/videoJetank1.mp4"  # Vervang met videobestand
output_map = "./sourceFolder2"
doel_fps = 4  # Pas aan naar gewenst FPS

extract_frames(video_pad, output_map, doel_fps)
