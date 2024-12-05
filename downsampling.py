import os
import random
import shutil

## Downsample Class 1
#Initialize folder paths
dataset_root = r'C:\Users\Dell\Documents\Desktop\A. ĐỒ ÁN TỔNG HỢP AI\code\dataset\train' 
class_1_dir = r'C:\Users\Dell\Documents\Desktop\A. ĐỒ ÁN TỔNG HỢP AI\code\dataset\train\originals\1'  
output_dir = os.path.join(dataset_root, 'originals/1_downsampled')  

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all Class 1 images
class_1_images = os.listdir(class_1_dir)

# Shuffle and select only 861 images randomly
selected_images = random.sample(class_1_images, 861)

# Copy selected images to the output directory
for img in selected_images:
    shutil.copy(os.path.join(class_1_dir, img), os.path.join(output_dir, img))

# Check the number of copied images in the downsampled folder
print(f"Selected {len(selected_images)} images for Class 1.")

##Rename the folders for easy of use in YOLOv8m-cls
#Rename the folder 1 to 1_archived
os.rename(class_1_dir, r'C:\Users\Dell\Documents\Desktop\A. ĐỒ ÁN TỔNG HỢP AI\code\dataset\train\originals\1_archived')
#Rename the folder 1_downsampled to 1
os.rename(output_dir, r'C:\Users\Dell\Documents\Desktop\A. ĐỒ ÁN TỔNG HỢP AI\code\dataset\train\originals\1')




