import os
import shutil
import json

# Paths to my dataset
dataset_root = r'C:\Users\Dell\Documents\Desktop\A. ĐỒ ÁN TỔNG HỢP AI\code\dataset\train' 
coco_file = r'C:\Users\Dell\Documents\Desktop\A. ĐỒ ÁN TỔNG HỢP AI\code\dataset\train\_annotations.coco.json'   
output_originals_0 = os.path.join(dataset_root, 'originals/0')
output_originals_1 = os.path.join(dataset_root, 'originals/1')
output_ground_truth_0 = os.path.join(dataset_root, 'ground_truth/0')
output_ground_truth_1 = os.path.join(dataset_root, 'ground_truth/1')

# Create necessary directories if they don't exist
os.makedirs(output_originals_0, exist_ok=True)
os.makedirs(output_originals_1, exist_ok=True)
os.makedirs(output_ground_truth_0, exist_ok=True)
os.makedirs(output_ground_truth_1, exist_ok=True)

# Load the COCO JSON file
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

# Iterate through images in the COCO JSON
for image_info in coco_data['images']:
    img_filename = image_info['file_name']
    img_id = image_info['id']
    img_class = None

    # Determine if this image is class 1 (contaminated)
    for ann in coco_data['annotations']:
        if ann['image_id'] == img_id:
            img_class = ann['category_id']
            break

    # Check if the image is ground truth ("GTM" in filename)
    src_path = os.path.join(dataset_root, img_filename)
    if "GTM" in img_filename:
        # Move to ground_truth folder
        if img_class == 1:
            shutil.move(src_path, os.path.join(output_ground_truth_1, img_filename))
        else:
            shutil.move(src_path, os.path.join(output_ground_truth_0, img_filename))
    else:
        # Move to originals folder
        if img_class == 1:
            shutil.move(src_path, os.path.join(output_originals_1, img_filename))
        else:
            shutil.move(src_path, os.path.join(output_originals_0, img_filename))

print("Images organized into originals and ground truth folders!")
