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
1. Download the Fisheye8K dataset, and put the data into `./dataset/fisheye8k/`. Link to the fisheye8k dataset: [link](https://scidm.nchc.org.tw/en/dataset/fisheye8k/resource/f6e7500d-1d6d-48ea-9d38-c4001a17170e/nchcproxy)

2. Download the VisDrone dataset, and put the data into `./dataset/visdrone/`. In our experiments, we only work with the VisDrone2019-DET-train subdataset. So downloading only the train set is suffice.

3. Download the FisheyeEval1k test dataset, and put the data into `./dataset/fisheye_test/`. For convenience, all test images should be put into one folder named `images`. The `./dataset/` directory will look like below:

```
- dataset
    - fisheye8k
        - test
        - train
    - visdrone
        - VisDrone2019-DET-test_dev
        - VisDrone2019-DET-train
        - VisDrone2019-DET-val
    - fisheye_test
        - images
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

7. Merge the VisDrone and the Synthetic VisDrone datasets with the Fisheye8k dataset individually. For convenience, all images are copied into a single directory. We create a script for automatic images copying, however, the process can also be done manually. The script creates a new directory `./dataset/all_images` and save all the images into it, because the names of the images are different, this won't affect the training and the evaluation process.

```
# Merge Fisheye8k, FisheyeEval1k and VisDrone images
python ./dataprocessing/data_combine/copy_images.py

# Merge coco files
# Remember to change the list of files and output files in the script
python ./dataprocessing/data_combine/merge_coco_files.py
```

## Models Training
### Co-DETR
```
cd ./train/CO-DETR

# Create the conda environment
conda create -n codetr python=3.8 pip cudatoolkit-dev=11.7 gxx=11.4 cudnn

# Install mmdet and dependencies
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
pip install -v -e .

```
1. Train the Co-DETR model on the VisDrone and Fisheye8k fold 0 dataset.
```
tools/dist_train.sh projects/CO-DETR/configs/codino/train_fold0.py 4
```
2. Train the Co-DETR model on the VisDrone and Fisheye8k fold 1 dataset.
```
tools/dist_train.sh projects/CO-DETR/configs/codino/train_fold1.py 4
```
3. Train the Co-DETR model on the VisDrone and Fisheye8k fold 2 dataset.
```
tools/dist_train.sh projects/CO-DETR/configs/codino/train_fold2.py 4
```
4. Train the Co-DETR model on the VisDrone and Fisheye8k merge training and testing dataset.
```
tools/dist_train.sh projects/CO-DETR/configs/codino/train_all.py 4
```
5. Train the Co-DETR model on the VisDrone and Fisheye8k merge training and testing and pseudo dataset.
```
tools/dist_train.sh projects/CO-DETR/configs/codino/train_pseudo.py 4
```

6. Train the Co-DETR model on the VisDrone and Fisheye8k merge training and testing and pseudo dataset.
```
tools/dist_train.sh projects/CO-DETR/configs/codino/train_syn_vis_fis.py 4
```

### YOLOR-W6
Follow these instructions to train the YOLOR-W6:
1. Create the conda environment
```
conda create -n yolor python=3.8 pip cudatoolkit-dev=11.7 gxx=11.4
conda activate yolor
```

2. Download the COCO-pretrained YOLOr-W6 model released by the authors and put the checkpoint in `./train/YoloR/`. Pretrained link: [yolor-w6-paper-555.pt](https://github.com/WongKinYiu/yolor/releases/download/weights/yolor-w6-paper-555.pt)

3. Install the dependencies
```
pip install -r requirements.txt
```

4. Train the YOLOr-W6 model on the VisDrone+Fisheye8k dataset using the following command

```
# Move to the YOLO-R directory
cd ./train/YoloR

# Train the yolor-w6 model for 250 epochs
python train.py --batch-size 8 --img 1280 1280 --data ../../dataset/visdrone_fisheye8k.yaml --cfg models/yolor-w6.yaml --weights './yolor-w6-paper-555.pt' --device 0 --name yolor_w6 --hyp hyp.scratch.1280.yaml --epochs 250
```
The checkpoints will be saved in `./train/YoloR/runs/train/`

### YOLOv9-e
Follow these instructions to train the YOLOv9-e:
1. Create the conda environment
```
conda create -n yolov9 python=3.8 pip cudatoolkit-dev=11.7 gxx=11.4
conda activate yolov9
```

2. Install the dependecies
```
pip install -r requirements.txt
```

3. Train the YOLOv9-e model on the VisDrone+Fisheye8k dataset using the following command

```
# Move to the YOLOv9 directory
cd ./train/YoloV9

# Train the yolov9-e model from sratch for 250 epochs
python train_dual.py --workers 8 --device 0 --batch 4 --data ../../dataset/visdrone_fisheye8k.yaml --img 1280 --cfg models/detect/yolov9-e.yaml --weights '' --name yolov9-e --hyp hyp.scratch-high.yaml --min-items 0 --epochs 250 --close-mosaic 15
```
The checkpoints will be saved in './train/YoloV9/runs/train/'

### InternImage
Follow these instructions to train the InternImage
1. Create conda environment
```
conda create -n internimage python=3.8 pip cudatoolkit-dev=11.7 gxx=11.4 cudnn
conda activate internimage
# Move to the InternImage directory
cd ./train/InternImage/
```

2. Install `torch==1.13.0` and `cuda=11.7`
``` 
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

3. Install `timm==0.6.11` and `mmcv-full==1.5.0`:
```
pip install -U openmim
mim install mmcv-full==1.5.0
pip install timm==0.6.11 mmdet==2.28.1
```

4. Install other requirements:
```
pip install opencv-python termcolor yacs pyyaml scipy

# To bypass the 'verify' TypeError in mmdet
pip install yapf==0.40.1 
```

5. Compile the CUDA operators
```
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
cd ..
```

6. Train the InternImage model on the VisDrone+Fisheye8k dataset using 8 GPU
```
# bash dist_train.sh <config-file> <gpu-num>
bash dist_train.sh ./configs/coco/ai_city_challenge_2024_train.py 8
```



## Models Inferencing

### Checkpoints
For quick reproduction, download checkpoints from this link below and put them in the `./checkpoints` directory: 
- [checkpoints](https://1drv.ms/f/s!AqGcdYmA92Q_m8Yg2hOB1PAk_15WBw?e=dFbNte)
### Co-DETR
```
cd ./infer/CO-DETR
```
1. Infer the Co-DETR model on the VisDrone and Fisheye8k fold 0.
```
tools/dist_test.sh projects/CO-DETR/configs/codino/infer_all.py ../../checkpoints/best_vis_fish_fold0.pth 4
```
2. Infer the Co-DETR model on the VisDrone and Fisheye8k fold 1.
```
tools/dist_test.sh projects/CO-DETR/configs/codino/infer_fold1.py ../../checkpoints/best_vis_fish_fold1.pth 4
```
3. Infer the Co-DETR model on the VisDrone and Fisheye8k fold 2.
```
tools/dist_test.sh projects/CO-DETR/configs/codino/infer_fold2.py ../../checkpoints/best_vis_fish_fold2.pth 4
```
4. Infer the Co-DETR model on the VisDrone and Fisheye8k merge training and testing.
```
tools/dist_test.sh projects/CO-DETR/configs/codino/infer_all.py ../../checkpoints/best_vis_fish_all.pth 4
```
5. Infer the Co-DETR model on the VisDrone and Fisheye8k merge training and testing and pseudo.
```
tools/dist_test.sh projects/CO-DETR/configs/codino/infer_pseudo.py ../../checkpoints/best_vis_fish_pseudo.pth 4
```

6. Infer the Co-DETR model on the Synthetic VisDrone and Fisheye8k merge training.
```
tools/dist_test.sh projects/CO-DETR/configs/codino/infer_syn_vis_fis.py ../../checkpoints/codetr_syn_vis_fish.pth 4
```

### YOLOR-W6
For inferencing, follow these instructions
1. Move to the YOLOR-W6 directory and activate the yolor conda environment created in the training phase. If you haven't, see the **Training** section for instructions.
```
cd ./infer/YoloR

# Activate the yolor environment
conda activate yolor
```

2. Infer using the yolor model, note that the iou threshold is set to 0.65:
python detect.py --source ../../dataset/fisheye_test/images --weights ../../checkpoints/yolor_w6_best_checkpoint.pt --conf 0.01 --iou 0.65 --img-size 1280 --device 0 --save-txt --save-conf

3. Convert to submission format. Remember to modify the path to the corresponding labels_dir
```
python ../../dataprocessing/format_conversion/yolo2coco.py --images_dir ../../dataset/fisheye_test/images --labels_dir ./runs/detect/exp/labels --output ./yolor_w6.json --conf 1 --submission 1 --is_fisheye8k 1
```

### YOLOv9-e
For inferencing, follow these instructions
1. Move to the YOLO directory and activate the yolov9 conda environment created in the training phase. If you haven't created the conda environment, see the **Training** section for instructions.
```
cd ./infer/YoloV9

# Activate the yolov9 conda environment
conda activate yolov9
```

2. Run inference using the yolov9 model. Note that the iou threshold is set to 0.75.
```
python detect_dual.py --source '../../dataset/fisheye_test/images' --img 1280 --device 0 --weights '../../checkpoints/yolov9_e_best_checkpoint.pt' --name yolov9_e --iou 0.75 --save-txt --save-conf
```

3. Convert to submission format. Remember to modify the path to the corresponding labels_dir
```
python ../../dataprocessing/format_conversion/yolo2coco.py --images_dir ../../dataset/fisheye_test/images --labels_dir ./runs/detect/yolov9_e/labels --output ./yolov9.json --conf 1 --submission 1 --is_fisheye8k 1
```

### InternImage
For inferencing, follow these instructions
1. Move to the InternImage directory
```
cd ./infer/InternImage/detection

# activate the conda environment
conda activate internimage
```

2. Run inference (modify the path to checkpoint in the demo_images.py file if necessary). The result is saved in submission format by default
```
python demo_images.py --source ../../../dataset/fisheye_test/images --out ./internimage.json
```


### Model ensembling
```
cd ./infer/
python fuse_results.py
```
