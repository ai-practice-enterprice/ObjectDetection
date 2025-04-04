# YOLO Auto Labeling Tool

This tool automates the process of labeling images and videos using the YOLOv8 model. The images and videos are processed, and frames from videos are extracted. The tool allows automatic label generation and provides an option to confirm or accept auto-generated labels based on a trust threshold. After labeling, the images and annotations are saved in a specified folder.

## Prerequisites

Before running the tool, make sure you have the following installed:

- **Python 3.7+**
- **OpenCV** for video processing: pip install opencv-python
- **Ultralytics YOLO** for object detection: pip install ultralytics
- **ftplib** for FTP connections (this should come with Python by default)
  
### Files Needed
- **AutoLabelV2.py**: The main script that processes the images and videos.
- **YOLO Model**: You need to have a YOLOv8 model trained on your custom dataset (e.g., `yolov8n.pt`).

## Setup

### 1. FTP Server Configuration
To upload images and videos to the `/Source` folder on the FTP server:

- Make sure your FTP server is running and accessible.
- Edit the FTP connection settings in the code by replacing the values in the following function:
  
  ftp_connect('ftp_server_address', 'username', 'password')

### 2. Define Source and Target Folders

- **Source Folder**: The folder where the images and videos are downloaded from the FTP server.
- **Target Folder**: The folder where processed images and labels are saved. Images and labels will be stored under `Target/images` and `Target/labels`.

### 3. Choose Trust Threshold
You can define a trust threshold for the confidence score in the script. Auto-generated labels with confidence scores above this threshold will be automatically accepted. Otherwise, you will need to manually confirm the labels.

- Modify the **confidence threshold** in the code:
  
  self.confidence_threshold = 0.7  # Set your desired confidence threshold

### 4. Set Up the YOLO Model
Ensure that you have a trained YOLOv8 model (e.g., `yolov8n.pt`) ready for use in the tool. Update the `MODEL_PATH` in the script:

MODEL_PATH = "path_to_your_yolov8_model/yolov8n.pt"

---

## How to Use

### Step 1: Upload Files to the FTP Server

To upload images or videos to the `/Source` folder on the FTP server, you can use any FTP client or the script. Ensure the files are placed inside the `/Source` folder.

### Step 2: Start the Auto Labeling Process

Once the files are uploaded to the FTP server, follow these steps to run the labeling process:

1. **Run the Script**:

    To start the auto-labeling process, run the `AutoLabelV2.py` script:

    ```bash
    python AutoLabelV2.py
    ```

2. **Processing and Frame Extraction**:

    - If the uploaded file is a video, it will automatically be converted into frames.
    - If the uploaded file is an image, it will directly be processed for annotation.

3. **Labeling Process**:

    - For each image, the script will use YOLOv8 to generate predictions (labels).
    - If the confidence score for an auto-generated label exceeds the trust threshold, it will be automatically accepted.
    - If the confidence score is below the threshold, you will need to confirm or reject the label manually. Press `'y'` to accept the label, `'n'` to reject, or `'q'` to quit the process.

4. **Saving Results**:

    - **Images** and **labels** will be saved under the `/Target` folder:
        - `/Target/images`: Contains the images with bounding boxes drawn by YOLO.
        - `/Target/labels`: Contains the corresponding `.txt` label files in YOLO format.
    
    The labels are saved with the same filename as the image (e.g., `image.jpg` → `image.txt`).

---

## Workflow

1. **Upload images/videos** to the `/Source` folder on the FTP server.
2. **Run the script** to automatically process the files.
3. **Videos will be converted to frames**, and the YOLO model will predict and generate labels.
4. **Label validation**: The auto-generated labels will be automatically accepted if the confidence is above the threshold. If not, you need to validate each label manually.
5. **Results** will be stored in the `/Target` folder.

---

## Notes

- The script assumes that the videos are in a format like `.mp4`, `.avi`, or `.mov`, and images are `.png`, `.jpg`, or `.jpeg`.
- The model path should point to your pre-trained YOLOv8 model. If you don't have a custom model, you can use a general pre-trained model, but for best results, fine-tuning it on your dataset is recommended.
- The FTP connection will use passive mode, so make sure your firewall and network settings allow FTP connections.

---

## Troubleshooting

- **"FTP connection failed" error**: Ensure your FTP server is running, and the credentials (username, password) are correct.
- **"Failed to load image" error**: This could be caused by unsupported image formats or corrupted files. Verify the files are correct.
- **Model prediction issues**: If the model isn't detecting objects well, consider training the model on your custom dataset.

---

## Conclusion

This tool simplifies the process of automatically labeling images and videos for object detection tasks using YOLO. It can be configured to automatically accept or manually confirm labels based on the confidence score.