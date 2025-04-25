import os
import cv2
import numpy as np
import shutil
import logging
from datetime import datetime
from ultralytics import YOLO
class YoloAnnotationTool:
    def __init__(self, source_folder, target_folder, classes, confidence_threshold=0.5, model_path='yolov8n.pt'):
        """
        Initialiseer de YOLO annotatie tool met Ultralytics
        
        Args:
            source_folder (str): Map met bronafbeeldingen
            target_folder (str): Doelmap voor output
            classes (list): Lijst met klassenamen
            confidence_threshold (float): Minimale confidence score om een detectie te tonen
            model_path (str): Pad naar het YOLO model
        """
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        # Statistieken bijhouden
        self.saved_count = 0
        self.user_rejected_count = 0
        self.auto_rejected_count = 0
        
        # Logger instellen
        logging.basicConfig(
            filename=os.path.join(self.target_folder, 'annotation_log.txt'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger()
        
        # Directory structuur aanmaken
        self.setup_directories()
        
        # YOLO model laden
        self.load_yolo_model()
    
    def setup_directories(self):
        """Maak de benodigde directory structuur aan"""
        os.makedirs(self.target_folder, exist_ok=True)
        self.images_folder = os.path.join(self.target_folder, 'images')
        self.labels_folder = os.path.join(self.target_folder, 'labels')
        
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.labels_folder, exist_ok=True)
        
        # Klassenbestand aanmaken
        with open(os.path.join(self.target_folder, 'classes.txt'), 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
    
    def load_yolo_model(self):
        """Laad het Ultralytics YOLO model"""
        # Ultralytics YOLO model laden
        self.model = YOLO(self.model_path)
    
    def predict(self, image_path):
        """
        Voer YOLO predictie uit op de gegeven afbeelding
        
        Args:
            image_path (str): Pad naar de afbeelding
            
        Returns:
            tuple: (afbeelding met bounding boxes, detecties, gemiddelde confidence)
        """
        # Afbeelding laden
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
        # Voorspelling uitvoeren met Ultralytics YOLO
        results = self.model(image)
        
        # Resultaten verwerken
        detections = []
        total_confidence = 0
        count = 0
        
        # Annotated afbeelding verkrijgen
        annotated_image = image.copy()
        
        # Voor elk resultaat
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Bounding box coördinaten (in pixels)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Class ID en confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Controleer of class_id geldig is
                if class_id < len(self.classes):
                    label = self.classes[class_id]
                else:
                    label = f"unknown_{class_id}"
                    self.logger.warning(f"Onbekende class ID: {class_id} in {image_path}")
                
                # Teken bounding box en label
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 8)
                cv2.putText(annotated_image, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)
                
                # Bereken YOLO formaat (genormaliseerd naar 0-1)
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                # Detectie opslaan
                detections.append({
                    'class_id': class_id,
                    'coordinates': [x_center, y_center, bbox_width, bbox_height],
                    'confidence': confidence
                })
                
                total_confidence += confidence
                count += 1
        
        # Gemiddelde confidence berekenen
        avg_confidence = total_confidence / count if count > 0 else 0
        
        return annotated_image, detections, avg_confidence
    
    def save_annotation(self, image_path, detections):
        """
        Sla afbeelding en annotaties op
        
        Args:
            image_path (str): Pad naar bronafbeelding
            detections (list): Lijst met detecties
        """
        # Bestandsnaam zonder pad
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        # Afbeelding kopiëren
        dest_image_path = os.path.join(self.images_folder, filename)
        shutil.copy2(image_path, dest_image_path)
        
        # Annotatie opslaan in YOLO-formaat
        annotation_path = os.path.join(self.labels_folder, f"{name}.txt")
        with open(annotation_path, 'w') as f:
            for detection in detections:
                class_id = detection['class_id']
                x_center, y_center, width, height = detection['coordinates']
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    def process_images(self):
        #Verwerk alle afbeeldingen in de bronmap
        image_files = [f for f in os.listdir(self.source_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.logger.info(f"Start verwerking van {len(image_files)} afbeeldingen")
        
        for image_file in image_files:
            image_path = os.path.join(self.source_folder, image_file)
            
            try:
                # YOLO predictie uitvoeren
                image_with_boxes, detections, avg_confidence = self.predict(image_path)
                
                # Controleer op de gemiddelde confidence
                if avg_confidence < self.confidence_threshold or not detections:
                    self.logger.info(f"Afbeelding {image_file} automatisch genegeerd (confidence: {avg_confidence:.2f})")
                    self.auto_rejected_count += 1
                    continue
                
                # Toon afbeelding met bounding boxes
                cv2.namedWindow("YOLO Annotatie", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLO Annotatie", 1200, 1000)
                cv2.moveWindow("YOLO Annotatie", 0, 0)
                cv2.imshow("YOLO Annotatie", image_with_boxes)
                print(f"Afbeelding: {image_file}, Gemiddelde confidence: {avg_confidence:.2f}")
                print("Druk 'y' om op te slaan, 'n' om te negeren, 'q' om te stoppen")
                
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('y'):
                    # Opslaan
                    self.save_annotation(image_path, detections)
                    self.saved_count += 1
                    self.logger.info(f"Afbeelding {image_file} opgeslagen met {len(detections)} annotaties")
                elif key == ord('q'):
                    # Stoppen
                    break
                else:
                    # Negeren (inclusief 'n')
                    self.user_rejected_count += 1
                    self.logger.info(f"Afbeelding {image_file} genegeerd door gebruiker")
                
            except Exception as e:
                self.logger.error(f"Fout bij verwerken van {image_file}: {str(e)}")
        
        cv2.destroyAllWindows()
        
        # Log resultaten
        self.log_results()
    
    def log_results(self):
        """Log de eindresultaten"""
        summary = f"""
        Annotatie voltooid op {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ---------------------------------------------------
        Opgeslagen afbeeldingen: {self.saved_count}
        Door gebruiker verwijderd: {self.user_rejected_count}
        Automatisch genegeerd (lage confidence): {self.auto_rejected_count}
        Totaal verwerkt: {self.saved_count + self.user_rejected_count + self.auto_rejected_count}
        """
        
        self.logger.info(summary)
        print(summary)


if __name__ == "__main__":
    # Configuratie
    SOURCE_FOLDER = "sourceFolder"
    TARGET_FOLDER = "targetFolder"
    CLASSES = ["je-tank","jetracer"]
    CONFIDENCE_THRESHOLD = 0              
    MODEL_PATH = "RobotDetectionv8Large.pt"               
    
    # Annotatie tool initialiseren en uitvoeren
    tool = YoloAnnotationTool(SOURCE_FOLDER, TARGET_FOLDER, CLASSES, CONFIDENCE_THRESHOLD, MODEL_PATH)
    tool.process_images()