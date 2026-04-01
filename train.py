DATA_ROOT = "export"
ANNOTATIONS = "export/result.json"
IMAGES = "export/images"

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

register_coco_instances(
    "books_train",
    {},
    "export/result.json",
    "export/images"
)

metadata = MetadataCatalog.get("books_train")

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
import os

cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)

cfg.DATASETS.TRAIN = ("books_train",)
cfg.DATASETS.TEST = ()

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.MODEL.WEIGHTS = "model_final.pth"

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 2000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

cfg.OUTPUT_DIR = "./output_books"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.DEVICE = "cuda"

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
