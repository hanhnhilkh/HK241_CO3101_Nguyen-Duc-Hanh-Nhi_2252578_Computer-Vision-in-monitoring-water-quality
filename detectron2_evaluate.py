from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg


cfg = get_cfg()  # Load your config
cfg.merge_from_file("path/to/config.yaml")
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # Use trained weights

evaluator = COCOEvaluator("dataset_name_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "dataset_name_val")
# inference_on_dataset(DefaultTrainer.build_model(cfg), val_loader, evaluator)


## Evaluation vẻrsion 2
# from detectron2.config import get_cfg
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.engine import DefaultTrainer
# from detectron2.data import build_detection_test_loader

# # Load the configuration and weights
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = "./output/model_final.pth"  # Adjust to your saved model path
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Ensure NUM_CLASSES matches training setup
# cfg.DATASETS.TEST = ("test_dataset",)

# # Build the evaluator and data loader
# evaluator = COCOEvaluator("test_dataset", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "test_dataset")

# # Run evaluation
# print("Evaluating on the test dataset...")
# inference_on_dataset(DefaultTrainer.build_model(cfg), val_loader, evaluator)
