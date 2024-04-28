import asyncio
from argparse import ArgumentParser
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import mmcv
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
import os.path as osp
import argparse
import os
import numpy as np
import json

from tqdm import tqdm

# Define the arguments
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
    ]

# Path to config file:
config_file = "./configs/coco/ai_city_challenge_2024_train.py"

# Path to checkpoint
checkpoint_file = "../../../checkpoints/InternImage_best_checkpoint.pth"

# Configurations
device = "cuda:0"
palette = "coco"
conf_thres = 0.01

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InternImage inferencing")
    parser.add_argument('--source', type=str, help='source images')
    parser.add_argument('--out', type=str, help='output json file')
    
    args = parser.parse_args()
    
    source = args.source
    output_file = args.out
    
    #test_pipeline = Compose(test_pipeline)
    # initialize model
    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
    
    if not os.path.isdir(source):
        results = []
        result = inference_detector(model, source)
    
    else:
        _, _, img_list = next(os.walk(source))
        results = []
        print("[INFO] Predicting images ...")
        for img_file in tqdm(img_list):
            image_id = get_image_Id(img_file)
            img_path = os.path.join(source, img_file)
            result = inference_detector(model, img_path)
            bboxes = []
            scores = []
            labels = []
            
            #print(len(result))
            for i in range(len(result)):
                category_id = i
                #print(len(result[i]))
                for annotation in result[i]:
                    #print(annotation)
                    bbox = (int(annotation[0]), int(annotation[1]), int(annotation[2]), int(annotation[3]))
                    score = round(annotation[4], 6)
                    bboxes.append(bbox)
                    scores.append(score)
                    labels.append(category_id)
            
            num_det = len(scores)
            for i in range(num_det):
                bbox_data = {}
                bbox_data["image_id"] = image_id
                x1, y1, x2, y2 = bboxes[i]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                bbox_data["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                bbox_data["category_id"] = int(labels[i])
                bbox_data["score"] = float(scores[i])
                results.append(bbox_data)

    with open(output_file, "w") as f:
        json.dump(results, f)