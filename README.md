# AI City Challenge 2024 - Track 4

# Robust Data Augmentation and Ensemble Method for Object Detection in Fisheye Camera Images

Team name: VNPT AI

Team ID: 9

# Descriptions:

This repository contains the solution proposed by the SmartVision team from VNPT AI to tackle the problem stated in the AI City Challenge 2024 - Track 4: "Road Object Detection in Fish-Eye Cameras". 
For reproduction, please follow the **Instructions** below.

# Instructions:

## Installation

## Data preparation
1. Download the Fisheye8K dataset, and put the data into './dataset/fisheye8k/'

2. Download the VisDrone dataset, and put the data into './dataset/visdrone/'.

3. Download the FisheyeEval1k test dataset, and put the data into './dataset/fisheye_test/'

```
- dataset
    - fisheye8k
        - ms_coco-format_labels
        - test
        - train
    - visdrone
        - VisDrone2019-DET-test_dev
        - VisDrone2019-DET-train
        - VisDrone2019-DET-val
        - test_dev.json
        - train.json
        - val.json
    - fisheye_test
        - images
        - images1
```

4. Convert the VisDrone dataset to YOLO format using the following command. Note that when converting the VisDrone dataset, we also map each category to their corresponding one in the Fisheye8k dataset, other categories are ignored. The labels will be saved in the "labels" directory under the corresponding sub-dataset

```
python ./dataprocessing/format_conversion/visdrone2yolo.py --data_path ./dataset/visdrone/VisDrone2019-DET-train
```

5. Transform the VisDrone images to fisheye images. The images are split into two squared images, and each of them is converted individually. You can use the code to visualize the labels after conversion, the results will look like below:

```
# Fisheye Conversion
python ./dataprocessing/ifish_augmentation/convert_visdrone.py --src_path ./dataset/visdrone --trg_path ./dataset/synthetic_visdrone --distortion 0.5 --data_type VisDrone2019-DET-train --crop True

# Visualize 4 images for checking
python ./dataprocessing/visualization/visualization.py --image_dir ./dataset/synthetic_visdrone/VisDrone2019-DET-train/images --label_dir ./dataset/synthetic_visdrone/VisDrone2019-DET-train/labels --num_imgs 4 --save_dir ./dataprocessing/visualization
```

6. Convert the Visdrone and Synthetic VisDrone datasets to MS-COCO annotation format. Note that for the VisDrone dataset, the old train.json file will be replaced by the new one. If you want to keep the old file, modify the --output argument in the command.

```
# Convert VisDrone dataset
python ./dataprocessing/format_conversion/yolo2coco.py --images_dir ./dataset/visdrone/VisDrone2019-DET-train/images --labels_dir ./dataset/visdrone/VisDrone2019-DET-train/labels --output ./dataset/visdrone/train.json

# Convert Synthetic VisDrone dataset
python ./dataprocessing/format_conversion/yolo2coco.py --images_dir ./dataset/synthetic_visdrone/VisDrone2019-DET-train/images --labels_dir ./dataset/synthetic_visdrone/VisDrone2019-DET-train/labels --output ./dataset/synthetic_visdrone/train.json
```

7. Merge the VisDrone and the Synthetic VisDrone datasets with the Fisheye8k dataset individually. We use a third party tool called COCO_merger for this task. For convenience, all images are copied into a single directory. We create a script for automatic images copying, however, the process can also be done manually.

```
# Merge Fisheye8k, FisheyeEval1k and VisDrone images

```

## Models Training
### Co-DETR

### YOLOR-W6

### YOLOv9-e

### InternImage





