import os
import shutil
import json

# Paths to dataset
dataset_root = r'C:\Users\Dell\Documents\Desktop\A. ĐỒ ÁN TỔNG HỢP AI\code\dataset\test' #Path to test folder
coco_file = r'C:\Users\Dell\Documents\Desktop\A. ĐỒ ÁN TỔNG HỢP AI\code\dataset\test\_annotations.coco.json'   #Path to _annotations.coco.json file of test folder
output_originals_0 = os.path.join(dataset_root, 'originals/0') #Path to save original non-contaminated images
output_originals_1 = os.path.join(dataset_root, 'originals/1') #Path to save original contaminated images

# Create necessary directories if they don't exist
os.makedirs(output_originals_0, exist_ok=True)
os.makedirs(output_originals_1, exist_ok=True)


# Load the COCO JSON file
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

# Iterate through images in the COCO JSON FILE
for image_info in coco_data['images']: #iterate each image in the "images[]" in the COCO JSON FILE
    img_filename = image_info['file_name'] #get file name of the image
    img_id = image_info['id'] #get id of the image
    img_class = None

    # Determine if this image is class 1 (contaminated)
    for ann in coco_data['annotations']:
        if ann['image_id'] == img_id:  #if the image_id in the "annotations[]" in the COCO JSON FILE is the image being iterated
            img_class = ann['category_id'] #get the img_class of the image
            break

    #get the original path/source path of the image
    src_path = os.path.join(dataset_root, img_filename)
    # Move to originals folder
    if img_class == 1:
        shutil.move(src_path, os.path.join(output_originals_1, img_filename)) # if img_class is 1 -> move the image to the contaminated folder
    else:
        shutil.move(src_path, os.path.join(output_originals_0, img_filename)) #else move the image to the non-contaminated folder

print("Images organized into originals and ground truth folders!")
