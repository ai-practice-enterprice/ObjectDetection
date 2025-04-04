# YOLO Robot Model 🦾

This project contains a trained YOLO model for detecting robots (`je-tank` and `jet-racer`). The model was trained using Ultralytics YOLOv5.

---

## 🚀 Installation
Follow these steps to set up the project and run the model.

### 1. Create a Virtual Environment (recommended)
To avoid conflicts with other Python packages, create a virtual environment:

python -m venv yolo-env
source yolo-env/bin/activate   # Linux/Mac
yolo-env\Scripts\activate      # Windows


### 2. Install Required Packages

  ```bash
pip install ultralytics opencv-python
  ```



### 3. Running The Model

Change the model if needed:
- RobotDetectionv5 - YOLOv5
- RobotDetectionv8 - YOLOv8
- RobotDetectionv8Large - Yolov8 with a larger dataset

```bash
python detect.py
```

(to run with video)
cap = cv2.VideoCapture("videofile.mp4")

Press "q" to quit




