import os
import shutil

# Source folder with all batches and images
source_folder = "./TrainPhotos"

# Output folders
output_images = "./data/images"
output_labels = "./data/labels"

# Create output folders if they don't exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

count = 0

# Recursively walk through TrainPhotos and its subfolders
for root, dirs, files in os.walk(source_folder):
    for file in files:
        # Only look for image files
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(root, file)
        label_path = os.path.splitext(image_path)[0] + ".txt"

        # Check if the corresponding label file exists
        if os.path.exists(label_path):
            # Copy image to data/images/
            shutil.copy(image_path, os.path.join(output_images, file))

            # Copy label to data/labels/
            label_name = os.path.basename(label_path)
            shutil.copy(label_path, os.path.join(output_labels, label_name))

            count += 1

print(f"✅ Done! {count} labeled image-label pairs copied to 'data/images' and 'data/labels'.")
