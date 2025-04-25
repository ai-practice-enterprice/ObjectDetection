import os
import cv2
import shutil
import logging
from ultralytics import YOLO
import random
import string
from datetime import datetime

# Generate a random string of a specified length
def generate_random_name(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


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
            random_name = generate_random_name()
            cv2.imwrite(os.path.join(output_folder, f"{random_name}.png"), frame)
        saved_count += 1
    cap.release()

    print(f"Total frames extracted: {saved_count}")


# YOLO Annotation tool
class YoloAnnotationTool:
    def __init__(self, source_folder, target_folder, classes, confidence_threshold=0.5, model_path='yolov8n.pt'):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.classes = classes
        self.confidence_threshold = confidence_threshold  # Added confidence threshold
        self.model_path = model_path
        self.saved_count = 0
        self.user_rejected_count = 0
        self.auto_rejected_count = 0
        self.total_processed_count = 0  # To track all processed images
        
        print(f"Initializing YoloAnnotationTool with model: {self.model_path}")
        self.setup_directories()
        self.load_yolo_model()

        # Set up logger
        logging.basicConfig(
            filename=os.path.join(self.target_folder, 'annotation_log.txt'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger()

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

    def process_images(self):
        print(f"Start file processing in: {self.source_folder}")
        
        # Process images from local folder
        image_files = [f for f in os.listdir(self.source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing {len(image_files)} images locally.")
        
        for file in image_files:
            local_file_path = os.path.join(self.source_folder, file)
            
            print(f"Processing file: {file}")
            image_with_boxes, detections, avg_confidence = self.predict(local_file_path)
            
            # Check for "unknown" detections and skip them
            if any('unknown' in self.classes[d['class_id']] for d in detections):
                os.remove(local_file_path)
                print(f"Image {file} auto-deleted (unknown detection).")
                self.auto_rejected_count += 1
                continue  # Skip if any unknown detections
            
            # Skip image if low confidence or no detection
            if avg_confidence < self.confidence_threshold or not detections:
                os.remove(local_file_path)
                print(f"Image {file} auto-deleted (low confidence: {avg_confidence:.2f})")
                self.auto_rejected_count += 1
                continue  # Skip if no detection or below threshold
            
            cv2.imshow("YOLO Annotatie", image_with_boxes)
            print(f"Avg confidence: {avg_confidence:.2f}")
            self.save_annotation(local_file_path, detections)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                self.saved_count += 1
            elif key == ord('q'):
                break
            else:
                self.user_rejected_count += 1
                continue
        
        cv2.destroyAllWindows()
        
        # Process video files if any are in the source folder
        video_files = [f for f in os.listdir(self.source_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        print(f"Processing {len(video_files)} video files locally.")
        
        for file in video_files:
            local_file_path = os.path.join(self.source_folder, file)
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
                    self.auto_rejected_count += 1
                    continue  # Skip if no detection
                
                cv2.imshow("YOLO Annotatie", image_with_boxes)
                print(f"Avg confidence: {avg_confidence:.2f}")
                self.save_annotation(image_path, detections)
                
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    self.saved_count += 1
                elif key == ord('q'):
                    break
                else:
                    self.user_rejected_count += 1
                    continue
        
        # Log results
        self.log_results()

    def log_results(self):
        """Log the final results"""
        summary = f"""
        Annotation completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ---------------------------------------------------
        Images saved: {self.saved_count}
        User rejected images: {self.user_rejected_count}
        Automatically rejected (low confidence or unknown detections): {self.auto_rejected_count}
        Total processed: {self.saved_count + self.user_rejected_count + self.auto_rejected_count}
        """
        
        self.logger.info(summary)
        print(summary)

# Main execution
if __name__ == "__main__":
    SOURCE_FOLDER = "./sourceFolder"  # Set the local source folder path
    TARGET_FOLDER = "./targetFolder"  # Set the target folder path
    CLASSES = ["je-tank", "jetracer"]  # Define classes
    CONFIDENCE_THRESHOLD = 0.5  # Set the confidence threshold here
    MODEL_PATH = "RobotDetectionv8Large.pt"  # Set the model path
    
    tool = YoloAnnotationTool(SOURCE_FOLDER, TARGET_FOLDER, CLASSES, CONFIDENCE_THRESHOLD, MODEL_PATH)
    tool.process_images()
