import cv2
import os
import random
import string

def generate_random_filename(length=8):
    """Genereert een willekeurige bestandsnaam met letters en cijfers."""
    letters_and_digits = string.ascii_lowercase + string.digits
    return ''.join(random.choices(letters_and_digits, k=length)) + ".jpg"

def video_to_frames(video_path, output_folder, start_batch_index=1):
    """Splits een video op in frames, roteert ze 90° en slaat ze op in submappen van 100 stuks."""
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error opening video file.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    folder_index = start_batch_index - 1

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Subfolder every 100 files
        if frame_count % 100 == 0:
            folder_index += 1
            current_subfolder = os.path.join(output_folder, f"batch_{folder_index:03d}")
            if not os.path.exists(current_subfolder):
                os.makedirs(current_subfolder)

        filename = generate_random_filename()
        frame_path = os.path.join(current_subfolder, filename)

        # Rotate Frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite(frame_path, frame)

        frame_count += 1

    video.release()
    print(f"{frame_count} frames saved across batches starting from batch_{start_batch_index:03d} in: {output_folder}")

# Custom Parameters
video_path = "./TrainVideos/IMG_6823.mp4"
output_folder = "./TrainPhotos"
start_batch = 34

# Start
video_to_frames(video_path, output_folder, start_batch)
