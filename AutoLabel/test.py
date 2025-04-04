import os
import cv2
import numpy as np
import shutil
import logging
from datetime import datetime
from ultralytics import YOLO
from ftplib import FTP, error_perm

# Function to connect with FTP-server
def ftp_connect(hostname, username, password):
    try:
        print(f"Attempting to connect to FTP server at {hostname}...")
        ftp = FTP(hostname)
        ftp.login(username, password)
        print(f"Successfully connected to FTP server at {hostname}.")
        return ftp
    except Exception as e:
        print(f"Failed to connect to FTP server at {hostname}. Error: {str(e)}")
        return None

# Function to download file from FTP-server
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

# Function to upload file to FTP-server
def upload_file(ftp, local_path, remote_path):
    try:
        print(f"Uploading file from {local_path} to {remote_path}")
        with open(local_path, 'rb') as f:
            ftp.storbinary(f"STOR {remote_path}", f)
        print(f"File uploaded successfully: {remote_path}")
    except error_perm as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
    
def upload_to_ftp(ftp, local_image_path, local_label_path, remote_image_path, remote_label_path):
    """Upload image and label to FTP server."""
    try:
        # Upload image
        upload_file(ftp, local_image_path, remote_image_path)
        
        # Upload label
        upload_file(ftp, local_label_path, remote_label_path)
        
        print(f"Successfully uploaded {local_image_path} and {local_label_path} to FTP server.")
    except Exception as e:
        print(f"Error uploading files: {str(e)}")


# Function to extract frames from video
def extract_frames(video_path, output_folder, target_fps):
    print(f"Start frame extraction from video: {video_path}")
    # Output
    os.makedirs(output_folder, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frames based on target FPS
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"Eframe_{saved_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"frame count: {frame_count}")
    print(f"Frames saved in {output_folder}")

# YOLO Annotation tool
class YoloAnnotationTool:
    def __init__(self, source_folder, target_folder, classes, confidence_threshold=0.5, model_path='yolov8n.pt'):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # Remember stats
        self.saved_count = 0
        self.user_rejected_count = 0
        self.auto_rejected_count = 0
        
        # Logger
        logging.basicConfig(
            filename=os.path.join(self.target_folder, 'annotation_log.txt'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger()
        
        # Directory structure
        self.setup_directories()
        
        # Load YOLO model
        self.load_yolo_model()
    
    def setup_directories(self):
        """Create necessary directory structure"""
        print(f"Setup directories in {self.target_folder}")
        os.makedirs(self.target_folder, exist_ok=True)
        self.images_folder = os.path.join(self.target_folder, 'images')
        self.labels_folder = os.path.join(self.target_folder, 'labels')
        
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.labels_folder, exist_ok=True)
        
        # Class file
        with open(os.path.join(self.target_folder, 'classes.txt'), 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
    
    def load_yolo_model(self):
        """Load YOLO model"""
        print(f"Load YOLO model from {self.model_path}")
        self.model = YOLO(self.model_path)
    
    def predict(self, image_path):
        print(f"Predict for image: {image_path}")
        image = cv2.imread(image_path)
        height, width, _ = image.shape
   
        results = self.model(image)
        
        detections = []
        total_confidence = 0
        count = 0
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id < len(self.classes):
                    label = self.classes[class_id]
                else:
                    label = f"unknown_{class_id}"
                    self.logger.warning(f"Unknown class ID: {class_id} in {image_path}")
                
                thickness = 2  # Thin lines for boxes
                
                # Draw smaller bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                
                # Smaller text size
                font_scale = 0.5 
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 255, 0)  # Green
                thickness = 1

                # Smaller text label
                cv2.putText(annotated_image, f"{label} {confidence:.2f}", 
                            (x1, y1 - 10), font, font_scale, color, thickness)

                # Calculate normalized coordinates for YOLO format
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                detections.append({
                    'class_id': class_id,
                    'coordinates': [x_center, y_center, bbox_width, bbox_height],
                    'confidence': confidence
                })
                
                total_confidence += confidence
                count += 1
        
        avg_confidence = total_confidence / count if count > 0 else 0
        return annotated_image, detections, avg_confidence


    def save_annotation(self, image_path, detections):
        print(f"Save annotations for: {image_path}")
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        dest_image_path = os.path.join(self.images_folder, filename)
        shutil.copy2(image_path, dest_image_path)
        
        annotation_path = os.path.join(self.labels_folder, f"{name}.txt")
        with open(annotation_path, 'w') as f:
            for detection in detections:
                class_id = detection['class_id']
                x_center, y_center, width, height = detection['coordinates']
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    def save_annotation_to_ftp(self, image_path, detections, ftp):
        print(f"Saving annotations for: {image_path}")
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        # Use FTP server to directly save the image and annotations
        # For example: /home/ai-team-user/AutoLabeling/Target/images/{filename}
        remote_image_path = f"/home/ai-team-user/AutoLabeling/Target/images/{filename}"
        remote_label_path = f"/home/ai-team-user/AutoLabeling/Target/labels/{name}.txt"
        
        # Use the annotated image (which contains bounding boxes)
        annotated_image, detections, avg_confidence = self.predict(image_path)  # This should be defined

        # Temporarily save the image and annotation file locally and upload
        local_image_path = os.path.join(self.images_folder, filename)
        cv2.imwrite(local_image_path, annotated_image)  # Temporarily save the image with boxes

        # Save label to a temporary text file and upload
        annotation_path = os.path.join(self.labels_folder, f"{name}.txt")
        with open(annotation_path, 'w') as f:
            for detection in detections:
                class_id = detection['class_id']
                x_center, y_center, width, height = detection['coordinates']
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        # Upload the files to the FTP server
        upload_to_ftp(ftp, local_image_path, annotation_path, remote_image_path, remote_label_path)
        
        # Remove temporary files
        os.remove(local_image_path)
        os.remove(annotation_path)


    
    def process_images(self):
        print(f"Start file processing in: {self.source_folder}")
        
        # Connect to FTP-server
        ftp = ftp_connect('192.168.1.11', 'ai-team-user', '7Ob9rg')
        if not ftp:
            return  # Exit if connection fails
        
        # Get files from FTP-server (source_folder)
        ftp.cwd('/home/ai-team-user/AutoLabeling/Source')
        files = ftp.nlst()  # Get list of files in directory
        
        for file in files:
            local_file_path = os.path.join(self.source_folder, file)
            
            # Download file
            download_file(ftp, file, local_file_path)
            
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process image files
                print(f"Processing file: {file}")
                image_with_boxes, detections, avg_confidence = self.predict(local_file_path)
                
                # Auto delete image if confidence = 0
                if avg_confidence == 0:
                    print(f"Image with no detection (avg. confidence: {avg_confidence:.2f}) is getting auto-deleted.")
                    os.remove(local_file_path)  # delete image
                    self.auto_rejected_count += 1
                    self.logger.info(f"Image {file} deleted (no detection).")
                    continue  # Next
                
                # Show image with bounding boxes
                cv2.imshow("YOLO Annotatie", image_with_boxes)
                print(f"Avg confidence: {avg_confidence:.2f}")
                print("Press 'y' to Save, 'n' to Ignore, 'q' to Quit")
                
                # Upload directly to FTP server
                self.save_annotation_to_ftp(local_file_path, detections, ftp)
                
                # Wait for user
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('y'):
                    self.saved_count += 1
                    self.logger.info(f"Image {file} saved with {len(detections)} annotations")
                elif key == ord('q'):
                    break
                else:
                    self.user_rejected_count += 1
                    self.logger.info(f"Image {file} ignored by user.")
            
            elif file.lower().endswith(('.mp4', '.avi', '.mov')):  # Process video files
                print(f"Process video: {file}")
                output_folder = os.path.join(self.source_folder, file.split('.')[0])
                extract_frames(local_file_path, output_folder, 4)

                image_files = [f for f in os.listdir(output_folder) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for image_file in image_files:
                    image_path = os.path.join(output_folder, image_file)
                    image_with_boxes, detections, avg_confidence = self.predict(image_path)
                    
                    # Delete frame if confidence = 0
                    if avg_confidence == 0:
                        print(f"Frame {image_file} with no detection (avg. confidence: {avg_confidence:.2f}) is getting auto-deleted.")
                        os.remove(image_path)
                        self.auto_rejected_count += 1
                        self.logger.info(f"Frame {image_file} auto-deleted (no detection).")
                        continue
                    
                    # Show image with bounding boxes
                    cv2.imshow("YOLO Annotatie", image_with_boxes)
                    print(f"Avg. confidence: {avg_confidence:.2f}")
                    print("Press 'y' to Save, 'n' to Ignore, 'q' to Quit")
                    
                    # Upload to FTP
                    self.save_annotation_to_ftp(image_path, detections, ftp)
                    
                    # Wait for user
                    key = cv2.waitKey(0) & 0xFF
                    
                    if key == ord('y'):
                        self.saved_count += 1
                        self.logger.info(f"Image {image_file} saved with {len(detections)} annotations")
                    elif key == ord('q'):
                        break
                    else:
                        self.user_rejected_count += 1
                        self.logger.info(f"Image {image_file} ignored by user.")
                
            # Close all open windows after processing
            cv2.destroyAllWindows()
        
        # Close FTP connection
        ftp.quit()




# Main execution
if __name__ == "__main__":
    SOURCE_FOLDER = "sourceFolder"
    TARGET_FOLDER = "targetFolder"
    CLASSES = ["je-tank", "jetracer"]
    CONFIDENCE_THRESHOLD = 0.7
    MODEL_PATH = "C:/Users/rayen/OneDrive/Documenten/GitHub/PE_Rayen/RobotDetectionModel/RobotDetectionv8Large.pt"

    
    tool = YoloAnnotationTool(SOURCE_FOLDER, TARGET_FOLDER, CLASSES, CONFIDENCE_THRESHOLD, MODEL_PATH)
    tool.process_images()