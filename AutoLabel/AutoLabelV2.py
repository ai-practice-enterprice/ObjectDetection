import os
import cv2
import shutil
import logging
from ultralytics import YOLO
from ftplib import FTP, error_perm

# Delete all files in local source folder after processing
def cleanup_source_folder(source_folder):
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print(f"Source folder {source_folder} is emptied.")


# Function to connect to FTP-server
def ftp_connect(hostname, username, password):
    try:
        ftp = FTP(hostname)
        ftp.login(username, password)
        return ftp
    except Exception as e:
        print(f"Failed to connect: {str(e)}")
        return None


def upload_to_ftp(ftp, local_image_path, local_label_path, remote_image_path, remote_label_path):
    try:
        with open(local_image_path, 'rb') as f:
            ftp.storbinary(f"STOR {remote_image_path}", f)
        with open(local_label_path, 'rb') as f:
            ftp.storbinary(f"STOR {remote_label_path}", f)
        print(f"Uploaded: {local_image_path} and {local_label_path}")
    except Exception as e:
        print(f"Error uploading {local_image_path} and {local_label_path}: {str(e)}")


def download_file(ftp, remote_path, local_path):
    try:
        print(f"Downloading file from {remote_path} to {local_path}")
        with open(local_path, 'wb') as f:
            ftp.retrbinary(f"RETR {remote_path}", f.write)
        print(f"File downloaded successfully: {remote_path}")
    except error_perm as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"Error downloading file: {str(e)}")


# Function to extract frames from video
def extract_frames(video_path, output_folder, target_fps):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)
    
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if saved_count % frame_interval == 0:
            cv2.imwrite(os.path.join(output_folder, f"Eframe_{saved_count:04d}.png"), frame)
        saved_count += 1
    cap.release()

    print(f"Total frames extracted: {saved_count}")


# YOLO Annotation tool
class YoloAnnotationTool:
    def __init__(self, source_folder, target_folder, classes, model_path='yolov8n.pt'):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.classes = classes
        self.model_path = model_path
        self.saved_count = 0
        
        print(f"Initializing YoloAnnotationTool with model: {self.model_path}")
        self.setup_directories()
        self.load_yolo_model()

    def setup_directories(self):
        os.makedirs(self.target_folder, exist_ok=True)
        self.images_folder = os.path.join(self.target_folder, 'images')
        self.labels_folder = os.path.join(self.target_folder, 'labels')
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.labels_folder, exist_ok=True)

    def load_yolo_model(self):
        self.model = YOLO(self.model_path)

    def predict(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None, [], 0
        results = self.model(image)
        print(f"Predictions made for {image_path}")
        detections, total_confidence, count = [], 0, 0
        annotated_image = image.copy()
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id < len(self.classes):
                    label = self.classes[class_id]
                else:
                    label = f"unknown_{class_id}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                x_center = ((x1 + x2) / 2) / image.shape[1]
                y_center = ((y1 + y2) / 2) / image.shape[0]
                bbox_width = (x2 - x1) / image.shape[1]
                bbox_height = (y2 - y1) / image.shape[0]
                detections.append({'class_id': class_id, 'coordinates': [x_center, y_center, bbox_width, bbox_height], 'confidence': confidence})
                total_confidence += confidence
                count += 1

        avg_confidence = total_confidence / count if count > 0 else 0
        return annotated_image, detections, avg_confidence

    def save_annotation(self, image_path, detections):
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)
        dest_image_path = os.path.join(self.images_folder, filename)
        shutil.copy2(image_path, dest_image_path)

        annotation_path = os.path.join(self.labels_folder, f"{name}.txt")
        with open(annotation_path, 'w') as f:
            for detection in detections:
                f.write(f"{detection['class_id']} {' '.join(map(str, detection['coordinates']))}\n")

    def save_annotation_to_ftp(self, image_path, detections, ftp):
        print(f"Saving annotations for: {image_path}")
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        remote_image_path = f"/home/ai-team-user/AutoLabeling/Target/images/{filename}"
        remote_label_path = f"/home/ai-team-user/AutoLabeling/Target/labels/{name}.txt"
        
        annotated_image, detections, avg_confidence = self.predict(image_path)

        local_image_path = os.path.join(self.images_folder, filename)
        cv2.imwrite(local_image_path, annotated_image)

        annotation_path = os.path.join(self.labels_folder, f"{name}.txt")
        with open(annotation_path, 'w') as f:
            for detection in detections:
                f.write(f"{detection['class_id']} {' '.join(map(str, detection['coordinates']))}\n")
        
        upload_to_ftp(ftp, local_image_path, annotation_path, remote_image_path, remote_label_path)
        
        os.remove(local_image_path)
        os.remove(annotation_path)

    def process_images(self):
        print(f"Start file processing in: {self.source_folder}")
        
        ftp = ftp_connect('192.168.1.11', 'ai-team-user', '7Ob9rg')
        if not ftp:
            return  # Exit if connection fails
        
        ftp.cwd('/home/ai-team-user/AutoLabeling/Source')
        files = ftp.nlst()
        
        for file in files:
            local_file_path = os.path.join(self.source_folder, file)
            
            download_file(ftp, file, local_file_path)
            
            print(f"Downloaded {file} to {local_file_path}")
            
            if file.lower().endswith(('.mp4', '.avi', '.mov')):  # Process video files
                print(f"Processing video: {file}")
                output_folder = os.path.join(self.source_folder, file.split('.')[0])
                extract_frames(local_file_path, output_folder, 4)
                
                image_files = [f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for image_file in image_files:
                    image_path = os.path.join(output_folder, image_file)
                    image_with_boxes, detections, avg_confidence = self.predict(image_path)
                    
                    if avg_confidence == 0:
                        os.remove(image_path)
                        print(f"Frame {image_file} auto-deleted (no detection).")
                        continue  # Skip if no detection
                    
                    cv2.imshow("YOLO Annotatie", image_with_boxes)
                    print(f"Avg confidence: {avg_confidence:.2f}")
                    self.save_annotation_to_ftp(image_path, detections, ftp)
                    
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('y'):
                        self.saved_count += 1
                    elif key == ord('q'):
                        break
                    else:
                        continue
            
            elif file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process image files
                print(f"Processing file: {file}")
                image_with_boxes, detections, avg_confidence = self.predict(local_file_path)
                
                if avg_confidence == 0:
                    os.remove(local_file_path)
                    print(f"Image {file} auto-deleted (no detection).")
                    continue  # Skip if no detection
                
                cv2.imshow("YOLO Annotatie", image_with_boxes)
                print(f"Avg confidence: {avg_confidence:.2f}")
                self.save_annotation_to_ftp(local_file_path, detections, ftp)
                
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    self.saved_count += 1
                elif key == ord('q'):
                    break
                else:
                    continue
        
        cv2.destroyAllWindows()
        ftp.quit()

        # After processing all files -> cleanup the source folder
        print(f"Cleaning up source folder: {self.source_folder}")
        cleanup_source_folder(self.source_folder)


# Main execution
if __name__ == "__main__":
    SOURCE_FOLDER = "sourceFolder"
    TARGET_FOLDER = "targetFolder"
    CLASSES = ["je-tank", "jetracer"]
    MODEL_PATH = "C:/Users/rayen/OneDrive/Documenten/GitHub/PE_Rayen/RobotDetectionModel/RobotDetectionv8Large.pt"
    
    tool = YoloAnnotationTool(SOURCE_FOLDER, TARGET_FOLDER, CLASSES, MODEL_PATH)
    tool.process_images() 
