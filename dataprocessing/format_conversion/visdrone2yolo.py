import os
import argparse
from tqdm import tqdm
import cv2


def visdrone_2_yolo(src_label_path, des_label_path, img_w, img_h):
    """
        Convert one label file from VisDrone-2019 format to standard YOLO format
        - Params:
            src_label_path: path to VisDrone formated label file
            des_label_path: path to destination YOLO label file
            img_w: image width, needed to normalize the bounding boxes' coordinates
            img_h: image height, needed to normalize the bounding boxes' coordinates   
    """
    with open(src_label_path, "r") as f:
        bboxes = f.readlines()

    # Open the new label file
    f = open(des_label_path, "w")

    for bbox in bboxes:
        args = bbox.split(",")
        left = int(args[0])
        top = int(args[1])
        bbox_w = int(args[2])
        bbox_h = int(args[3])
        score = int(args[4])
        category = str(args[5])
        category = map_categories(category)

        # Ignored classes
        if category == -1:
            continue

        # VisDrone Ignored bounding boxes
        if score == 0:
            continue

        center_x = left + bbox_w//2
        center_y = top + bbox_h//2

        center_x = round(float(center_x)/img_w, 6)
        center_y = round(float(center_y)/img_h, 6)
        bbox_w = round(float(bbox_w)/img_w, 6)
        bbox_h = round(float(bbox_h)/img_h, 6)
        f.write("{} {} {} {} {}\n".format(category, center_x, center_y, bbox_w, bbox_h))

    f.close()
    

def map_categories(category_id):
    """
        Mapping VisDrone categories to corresponding Fisheye8k categories
        - Params:
            category_id: VisDrone category_id
        
        - Returns:
            category_id: Fisheye 8k category id
    """
    interested_cls = {
        "0": -1,    # VisDrone ignored regions  -> ignore
        "1": 3,     # VisDrone pedestrian       -> YOLO pedestrian
        "2": -1,    # VisDrone human            -> ignore
        "3": -1,    # VisDrone bicycle          -> ignore
        "4": 2,     # VisDrone car              -> YOLO car
        "5": 2,     # VisDrone van              -> YOLO car
        "6": 4,     # VisDrone truck            -> YOLO truck
        "7": -1,    # VisDrone tricycle         -> ignore
        "8": -1,    # VisDrone awning-tricycle  -> ignore
        "9": 0,     # VisDrone bus              -> YOLO bus
        "10": 1,    # VisDrone motor            -> YOLO bike
        "11": -1,   # VisDrone others           -> ignore
    }

    return interested_cls[category_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset from VisDrone format to YOLO format")
    parser.add_argument("--data_path", type=str, default="../visdrone", help="Path to VisDrone dataset")
    
    args = parser.parse_args()

    data_path = args.data_path

    src_images_path = os.path.join(data_path, "images")
    src_labels_path = os.path.join(data_path, "annotations")
    trg_labels_path = os.path.join(data_path, "labels")

    assert os.path.exists(src_images_path)
    assert os.path.exists(src_labels_path)

    if not os.path.exists(trg_labels_path):
        os.mkdir(trg_labels_path)

    _, _, img_list = next(os.walk(src_images_path))

    for image_file in tqdm(img_list):
        image_name = image_file.split(".")[0]
        label_file = image_name + ".txt"

        src_image_path = os.path.join(src_images_path, image_file)
        src_label_path = os.path.join(src_labels_path, label_file)
        trg_label_path = os.path.join(trg_labels_path, label_file)

        img = cv2.imread(src_image_path)
        height, width, channel = img.shape

        visdrone_2_yolo(src_label_path, trg_label_path, width, height)