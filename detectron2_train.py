if __name__ == "__main__": #solve the problem of runtime error from multiprocessing, which is common common on Windows when using multiprocessing in PyTorch or Detectron2. It arises because Windows uses the spawn method for starting processes, whereas Unix systems use fork.
    ##4 lines for checking if pytorch and detectron2 are installed correctly by printing their versions
    # import torch
    # import detectron2
    # print(f"PyTorch version: {torch.__version__}")
    # print(f"Detectron2 version: {detectron2.__version__}")

    from detectron2 import model_zoo
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg

    # Register train and val datasets
    # name, dictionary for metadata, json_file, image_root 
    register_coco_instances("train_dataset", {}, "C:/Users/Dell/Documents/Desktop/A_AI_Project/code/dataset_detectron2/train/_annotations_train.coco.json", "C:/Users/Dell/Documents/Desktop/A_AI_Project/code/dataset_detectron2/train")
    register_coco_instances("val_dataset", {}, "C:/Users/Dell/Documents/Desktop/A_AI_Project/code/dataset_detectron2/val/_annotations_val.coco.json", "C:/Users/Dell/Documents/Desktop/A_AI_Project/code/dataset_detectron2/val")
    register_coco_instances("test_dataset", {}, "C:/Users/Dell/Documents/Desktop/A_AI_Project/code/dataset_detectron2/test/_annotations_test.coco.json", "C:/Users/Dell/Documents/Desktop/A_AI_Project/code/dataset_detectron2/test")

    # Configure Detectron2
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #load weights from the Detectron2 model zoo without needing to know the exact URL.
    cfg.DATASETS.TRAIN = ("train_dataset",) #I'm training the model on the train set
    cfg.DATASETS.TEST = ("val_dataset",) #I'm evaluating the model on the val set
    cfg.TEST.EVAL_PERIOD = 10 #This will do evaluation once after 10 iterations on the cfg.DATASETS.TEST, which should be my val set.
    cfg.MODEL.DEVICE = "cpu" #I want to use CPU to train first, then I'll consider using GPU and downloading CUDA and use pytorch+CUDA

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let weight taken from training initialize from model zoo
    cfg.SOLVER.BASE_LR = 0.001  #pick a Learning Rate for the model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class: "1" (contaminated)


    cfg.DATALOADER.NUM_WORKERS = 2 #enable multiprocesing with 2 parralel subprocesses to load a batch of data
    cfg.SOLVER.IMS_PER_BATCH = 2 #number of images per batch - batch size - refers to the number of training examples utilized in one iteration.
    cfg.SOLVER.MAX_ITER = 100 #number of iterations to train the model
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 #number of regions of interest (ROIs) per image in a batch. Each image will take 128 ROIs to train, reduce computational load -> faster (default: 512)
    # cfg.SOLVER.STEPS = [] # do not decay learning rate at any iteration/step decay 

    # Initialize Trainer and train the model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

