import os
import json
import cv2
import argparse
from tqdm import tqdm


def coco_2_yolo(coco_path, des_label_dir):
    """
        Convert
    """
    assert(os.path.exists(coco_path))
    if not os.path.exists(des_label_dir):
        os.mkdir(des_label_dir)

    with open(coco_path, "r") as f:
        coco_data = json.load(f)

    # initialize the dictionaries
    id2img = {}
    img2anno = {}

    categories = coco_data["categories"]
    images = coco_data["images"]
    annotations = coco_data["annotations"]

    print("[INFO]: Loading images ...")
    for image in tqdm(images):
        image_id = image["id"]
        image_file = image["file_name"]
        # Initialize the annotations list
        img2anno[image_file] = []
        img_w = image["width"]
        img_h = image["height"]
        id2img[image_id] = [image_file, img_w, img_h]

    print("[INFO]: Loading annotations ...")
    for annotation in tqdm(annotations):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]

        # Convert to YOLO format
        bbox[0] = bbox[0] - bbox[2]//2
        bbox[1] = bbox[1] - bbox[3]//2

        img_file = id2img[image_id][0]
        img_w = id2img[image_id][1]
        img_h = id2img[image_id][2]

        bbox[0] = round(float(bbox[0])/img_w, 6)
        bbox[1] = round(float(bbox[1])/img_h, 6)
        bbox[2] = round(float(bbox[2])/img_w, 6)
        bbox[3] = round(float(bbox[3])/img_h, 6)

        new_bbox = [category_id, bbox[0], bbox[1], bbox[2], bbox[3]]
        img2anno[img_file].append(new_bbox)

    print("[INFO] Writing annotations to file")
    for image_file in tqdm(img2anno.keys()):
        image_name = image_file.split(".")[0]
        label_file = image_name + ".txt"
        label_path = os.path.join(des_label_dir, label_file)
        with open(label_path, "w") as f:
            for bbox in img2anno[image_file]:
                f.write("{} {} {} {} {}\n".format(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset from COCO format to YOLO format")
    parser.add_argument("--coco_path", type=str, help="Path to coco json file")
    parser.add_argument("--labels_dir", type=str, help="Path to YOLO labels directory")

    args = parser.parse_args()

    coco_path = args.coco_path
    des_label_dir = args.labels_dir

    coco_2_yolo(coco_path, des_label_dir)