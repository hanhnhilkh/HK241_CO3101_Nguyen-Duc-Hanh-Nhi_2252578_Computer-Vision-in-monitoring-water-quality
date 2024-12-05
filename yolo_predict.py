from ultralytics import YOLO
import os
import shutil

# Load the trained model having the best performance
model = YOLO('C:/Users/Dell/Documents/Desktop/A_AI_Project/code/runs/classify/train16/weights/best.pt')  


##Define the directories
# Directory containing the images to predict
source_train_dir = 'C:/Users/Dell/Documents/Desktop/A_AI_Project/code/dataset_predict_yolo/train'
source_val_dir = 'C:/Users/Dell/Documents/Desktop/A_AI_Project/code/dataset_predict_yolo/val'
source_test_dir = 'C:/Users/Dell/Documents/Desktop/A_AI_Project/code/dataset_predict_yolo/test'
# Directory to save contaminated images
save_train_dir = 'C:/Users/Dell/Documents/Desktop/A_AI_Project/code/yolo_predict_results/train_predictions_1'  
save_val_dir = 'C:/Users/Dell/Documents/Desktop/A_AI_Project/code/yolo_predict_results/val_predictions_1'
save_test_dir = 'C:/Users/Dell/Documents/Desktop/A_AI_Project/code/yolo_predict_results/test_predictions_1'

# Ensure the all the three save directories exists
os.makedirs(save_train_dir, exist_ok=True)
os.makedirs(save_val_dir, exist_ok=True)
os.makedirs(save_test_dir, exist_ok=True)

##Run predictions

# Predict on the val set  
results_val = model.predict(source=source_val_dir, imgsz=512)


# Iterate over the results of the val prediction
for result in results_val:
    # Check if the predicted class is "contaminated" 
    if result.probs.top1 == 1:  # index if class "contaminated" is 1 in file yolo_dataset.yaml
        img_path = result.path  # Path to the original image
        img_name = os.path.basename(img_path)  # Extract the image name
        
        # Copy the image to the save directory
        shutil.copy(img_path, os.path.join(save_val_dir, img_name))

print(f"Saved contaminated images to {save_val_dir}")



# Predict on the train set    
results_train = model.predict(source=source_train_dir, imgsz=512)


# Iterate over the results of the train prediction
for result in results_train:
    # Check if the predicted class is "contaminated" 
    if result.probs.top1 == 1:  # index if class "contaminated" is 1 in file yolo_dataset.yaml
        img_path = result.path  # Path to the original image
        img_name = os.path.basename(img_path)  # Extract the image name
        
        # Copy the image to the save directory
        shutil.copy(img_path, os.path.join(save_train_dir, img_name))

print(f"Saved contaminated images to {save_train_dir}")


# Predict on the test set   DONE!!
results_test = model.predict(source=source_test_dir, imgsz=512)


# Iterate over the results of the train prediction
for result in results_test:
    # Check if the predicted class is "contaminated" 
    if result.probs.top1 == 1:  #In binary classification, the class with the higher probability is the predicted class. 
                                #So here it means if the class with the higher probability is detected as "contaminated" 
                                # => the image is classified as class "contaminated", whose index is 1 in file yolo_dataset.yaml
        img_path = result.path  # Path to the original image
        img_name = os.path.basename(img_path)  # Extract the image name
        
        # Copy the image to the save directory
        shutil.copy(img_path, os.path.join(save_test_dir, img_name))

print(f"Saved contaminated images to {save_test_dir}")





