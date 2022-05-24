import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import subprocess
import time
from pathlib import Path


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, BoxMode 
import pycocotools.mask as mask_util # tobbox(rle)  


# utilities
from pprint import pprint # For beautiful print!
from collections import OrderedDict
import os 

# For data visualisation
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import pandas as pd
from collections import OrderedDict
import json
from pycocotools.coco import COCO


TRAIN_ANNOTATIONS_PATH = "data/train/annotations.json"
TRAIN_IMAGE_DIRECTIORY = "data/train/images/"

VAL_ANNOTATIONS_PATH = "data/val/annotations.json"
VAL_IMAGE_DIRECTIORY = "data/val/images/"
    
    
def create_cfg():
    train_annotations_path = 'data/train/annotations.json'
    train_images_path = 'data/train/images'

    val_annotations_path = 'data/val/annotations.json'
    val_images_path = 'data/val/images'
    
    register_coco_instances("training_dataset", {},train_annotations_path, train_images_path)
    register_coco_instances("validation_dataset", {},val_annotations_path, VAL_IMAGE_DIRECTIORY)
    # Select your config from model_zoo, we have released pre-trained models for x101 and r50.

    # Download available here: https://drive.google.com/drive/folders/10_JiikWP59vm2eGIxRenAXxvYLDjUOz0?usp=sharing (10k iters)
    # Pre-trained with score of (0.030 AP, 0.050 AR)
    # MODEL_ARCH = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

    # Download available here: https://drive.google.com/drive/folders/1-LLFE8xFGOKkzPXF1DKF45c6O4W-38hu?usp=sharing (110k iters)
    # Pre-trained with score of (0.082 AP, 0.128 AR)
    MODEL_ARCH = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    #"https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg = get_cfg()
    # Check the model zoo and use any of the models ( from detectron2 github repo)

    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_ARCH))

    cfg.DATASETS.TRAIN = ("training_dataset",)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 2

    # Loading pre trained weights
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ARCH)
    
    # No. of Batchs
    cfg.SOLVER.IMS_PER_BATCH = 4     #for 16 GB GPU, reduce it to 2 for 12 GB GPU if you face CUDA memory error

    # Learning Rate: 
    cfg.SOLVER.BASE_LR = 0.0025

    # No of Interations
    cfg.SOLVER.MAX_ITER = 150000

    # Options: WarmupMultiStepLR, WarmupCosineLR.
    # See detectron2/solver/build.py for definition.
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

    #save every 1000 steps
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000

    # Images per batch (Batch Size) 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  

    # No of Categories(Classes) present
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 323 #498



    #Output directory
    #### NOTE: You can also download pre-trained folder from Google Drive and upload in your drive; links are shared in above cell.
    # cfg.OUTPUT_DIR = "/content/drive/MyDrive/logs_detectron2_x101"
    cfg.OUTPUT_DIR = "/home/polovinkope/output"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg
        
    
def train_model(cfg, train_coco):
    RESUME = True
    #print("tutA")
    trainer = DefaultTrainer(cfg) 
    #print("tutA")
    if RESUME:
        trainer.resume_or_load(resume=True)
    else:
        trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = '/home/polovinkope/output/model_final.pth'

    evaluator = COCOEvaluator("validation_dataset", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "validation_dataset")
    valResults = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 323

    cfg.DATASETS.TEST = ("validation_dataset", )
    predictor = DefaultPredictor(cfg)
    
    val_metadata = MetadataCatalog.get("val_dataset")

    '''#sample image 
    image_id = '008536'
    im = cv2.imread(f"data/val/images/{image_id}.jpg")

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],
                       metadata=val_metadata, 
                       scale=2,
                       instance_mode=ColorMode.IMAGE_BW
        )

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])'''
    
    category_ids = sorted(train_coco.getCatIds())
    categories = train_coco.loadCats(category_ids)

    class_to_category = { int(class_id): int(category_id) for class_id, category_id in enumerate(category_ids) }

    with open("class_to_category.json", "w") as fp:
      json.dump(class_to_category, fp)
    
    
def main():

    

    train_coco = COCO(TRAIN_ANNOTATIONS_PATH)
    
    

    with open(TRAIN_ANNOTATIONS_PATH) as f:
      train_annotations_data = json.load(f)

    with open(VAL_ANNOTATIONS_PATH) as f:
      val_annotations_data = json.load(f)
    
    category_ids = train_coco.loadCats(train_coco.getCatIds())
    category_names = [_["name_readable"] for _ in category_ids]
    #print(len(category_names))
    #print("## Categories\n-", "\n- ".join(category_names))
    

    # Getting all categoriy with respective to their total images

    no_images_per_category = {}

    for n, i in enumerate(train_coco.getCatIds()):
      imgIds = train_coco.getImgIds(catIds=i)
      label = category_names[n]
      no_images_per_category[label] = len(imgIds)

    img_info = pd.DataFrame(train_coco.loadImgs(train_coco.getImgIds()))
    no_images_per_category = OrderedDict(sorted(no_images_per_category.items(), key=lambda x: -1*x[1]))

    # Top 30 categories, based on number of images
    i = 0
    for k, v in no_images_per_category.items():
      #print(k, v)
      i += 1
      if i > 30:
        break
    fig = go.Figure([go.Bar(x=list(no_images_per_category.keys())[:50], y=list(no_images_per_category.values())[:50])])
    fig.update_layout(
        title="No of Image per class",)

    #fig.show()

    fig = go.Figure([go.Bar(x=list(no_images_per_category.keys())[50:200], y=list(no_images_per_category.values())[50:200])])
    fig.update_layout(
        title="No of Image per class",)

    #fig.show()

    fig = go.Figure([go.Bar(x=list(no_images_per_category.keys())[200:], y=list(no_images_per_category.values())[200:])])
    fig.update_layout(
        title="No of Image per class",)

    #fig.show()
    
    #print(f"Average number of image per class : { sum(list(no_images_per_category.values())) / len(list(no_images_per_category.values())) }")
    #print(f"Highest number of image per class is : { list(no_images_per_category.keys())[0]} of { list(no_images_per_category.values())[0] }")
    #print(f"Lowest number of image per class is : Veggie Burger of { sorted(list(no_images_per_category.values()))[0] }")
    img_no = 4557

    annIds = train_coco.getAnnIds(imgIds=train_annotations_data['images'][img_no]['id'])
    anns = train_coco.loadAnns(annIds)

    # load and render the image
    

    # Going thought every cell in rows and cols
    '''for i in range(1, cols * rows+1):
      annIds = train_coco.getAnnIds(imgIds=img_info['id'][i])
      anns = train_coco.loadAnns(annIds)

      fig.add_subplot(rows, cols, i)

      # Show the image

      img = plt.imread(TRAIN_IMAGE_DIRECTIORY+img_info['file_name'][i])
      for i in anns:
        [x,y,w,h] = i['bbox']
        #create rectagle bbox of size given in dataset
        cv2.rectangle(img, (int(x), int(y)), (int(x+h), int(y+w)), (255,0,0), 2)
      plt.imshow(img)

      # Render annotations on top of the image
      train_coco.showAnns(anns)

      # Setting the axis off
      plt.axis("off")

    # Showing the figure
    plt.show()'''
    cfg = create_cfg()
    train_model(cfg, train_coco)
    

if __name__ == "__main__":
    main()