import os
import cv2
import json
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

MODEL_PATH = "output_books/model_final.pth"
TEST_IMAGE = "test_page_3.jpg"
OUTPUT_DIR = "inference_output"
NUM_CLASSES = 1
SCORE_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)

cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

image = cv2.imread(TEST_IMAGE)
outputs = predictor(image)

instances = outputs["instances"].to("cpu")

boxes = instances.pred_boxes.tensor.numpy()
classes = instances.pred_classes.numpy()
scores = instances.scores.numpy()

# Visualization
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("__unused"))
out = v.draw_instance_predictions(instances)

vis_path = os.path.join(OUTPUT_DIR, "page.jpg")
cv2.imwrite(vis_path, out.get_image()[:, :, ::-1])

print(f"Визуализация сохранена: {vis_path}")

for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
    if score < SCORE_THRESHOLD:
        continue

    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]

    crop_path = os.path.join(OUTPUT_DIR, f"class_{cls}_score_{score:.2f}_{i}.png")
    cv2.imwrite(crop_path, crop)

print("Кропы сохранены")

predictions = []

for box, cls, score in zip(boxes, classes, scores):
    if score < SCORE_THRESHOLD:
        continue

    x1, y1, x2, y2 = box
    predictions.append({
        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        "category_id": int(cls),
        "score": float(score)
    })

json_path = os.path.join(OUTPUT_DIR, "boxes.json")
with open(json_path, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"JSON сохранён: {json_path}")
