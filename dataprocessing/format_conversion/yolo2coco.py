import os
import json
import cv2
import argparse
from tqdm import tqdm


def get_image_Id(img_name):
    """
        Calculate imageId from image file
        - Params:
            img_name: image file
        
        - Returns:
            imageId: the id of the image
    """
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId


def yolo_2_coco(images_dir, labels_dir, output_file, use_fisheye8k_id=False):
    """
        Convert YOLO dataset to COCO json format
        - Params:
            images_dir          :
            labels_dir          :
            output_file         :
            use_fisheye8k_id    :
    """
    categories = []
    images = []
    annotations = []

    # Add categories' ids
    categories.append({"id": 0, "name": "Bus"})
    categories.append({"id": 1, "name": "Bike"})
    categories.append({"id": 2, "name": "Car"})
    categories.append({"id": 3, "name": "Pedestrian"})
    categories.append({"id": 4, "name": "Truck"})

    # Loop through the image directory
    _, _, images_list = next(os.walk(images_dir))
    image_id = 0
    annotation_id = 0

    for image_file in tqdm(images_list):
        image_path = os.path.join(images_dir, image_file)
        img = cv2.imread(image_path)
        img_h, img_w, img_c = img.shape
        
        if use_fisheye8k_id:
            id = get_image_Id(image_file)
        else:
            id = image_id
            image_id += 1

        images.append({
            "id": id,
            "file_name": image_file,
            "width": img_w,
            "height": img_h
        })
    
        label_file = image_file.split(".")[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            bboxes = f.readlines()

        for bbox in bboxes:
            args = bbox.split(" ")
            category_id = int(args[0])
            center_x = int(float(args[1]) * img_w)
            center_y = int(float(args[2]) * img_w)
            bbox_w = int(float(args[3]) * img_w)
            bbox_h = int(float(args[4]) * img_h)

            left = center_x - bbox_w//2
            top = center_y - bbox_h//2

            annotations.append({
                "id": annotation_id,
                "category_id": category_id,
                "image_id": id,
                "bbox": [left, top, bbox_w, bbox_h],
                "segmentation": [],
                "area": bbox_w * bbox_h,
                "iscrowd": 0
            })

            annotation_id += 1
    
    data_dict = {}
    data_dict["categories"] = categories
    data_dict["images"] = images
    data_dict["annotations"] = annotations

    with open(output_file, "w") as f:
        json.dump(data_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset from YOLO format to COCO format")
    parser.add_argument("--images_dir", type=str, default="../visdrone", help="Path to images directory")
    parser.add_argument("--labels_dir", type=str, default="../visdrone", help="Path to labels directory")
    parser.add_argument("--output", type=str, default="./output.json", help="Path to the output json file")
    parser.add_argument("--is_fisheye8k", type=bool, default=False, help="Whether to use the Fisheye8k imageId")

    args = parser.parse_args()
    images_dir = args.images_dir
    labels_dir = args.labels_dir
    output = args.output
    is_fisheye8k = args.is_fisheye8k

    print(is_fisheye8k)

    yolo_2_coco(images_dir, labels_dir, output, is_fisheye8k)