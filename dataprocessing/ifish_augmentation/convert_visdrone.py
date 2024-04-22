import os
from ifisheye import *
from utils import *
import cv2
import argparse
from tqdm import tqdm


def convert_one_image_and_box(src_image_path, src_label_path, des_image_dir, des_label_dir, distortion_coefficient=0.5, crop=True):
    """
        Convert one VisDrone image to 2 Fisheye images
        - Params:
            src_image_path: path to the image
            src_label_path: path to the corresponding label, given that the label is in YOLO format (center_x, center_y, bbox_w, bbox_h)
    """
    assert os.path.exists(src_image_path)
    assert os.path.exists(src_label_path)
    assert os.path.exists(des_image_dir)
    assert os.path.exists(des_label_dir)

    image_name = src_image_path.split("/")[-1].split(".")[0]

    img = cv2.imread(src_image_path)
    height, width, channel = img.shape
    with open(src_label_path, "r") as f:
        bboxes = f.readlines()

    categories = []
    new_bboxes = []
    for bbox in bboxes:
        args = bbox.split(" ")
        categories.append(int(args[0]))
        center_x = int(float(args[1]) * width)
        center_y = int(float(args[2]) * height)
        bbox_w = int(float(args[3]) * width)
        bbox_h = int(float(args[4]) * height)

        left = int(center_x - bbox_w//2)
        top = int(center_y - bbox_h//2)
        right = int(center_x + bbox_w//2)
        bot = int(center_y + bbox_h//2)
        
        new_bboxes.append([left, top, right, bot])

    # Split the image into 2
    img1, img2 = split_image(img)
    bboxes1, bboxes2, categories1, categories2 = split_bboxes(categories, new_bboxes, img)

    # Convert each image and label into fisheye image
    new_img1 = convert_image(img1, distortion_coefficient, crop)
    new_img2 = convert_image(img2, distortion_coefficient, crop)
    old_h, old_w, _ = img1.shape
    new_h, new_w, _ = new_img1.shape
    old_size = (old_w, old_h)
    new_size = (new_w, new_h)
    new_bboxes1 = convert_bboxes(bboxes1, old_size, new_size, distortion_coefficient, crop)
    new_bboxes2 = convert_bboxes(bboxes2, old_size, new_size, distortion_coefficient, crop)

    img1_name = image_name + "_1"
    img2_name = image_name + "_2"

    img1_path = os.path.join(des_image_dir, img1_name + ".jpg")
    img2_path = os.path.join(des_image_dir, img2_name + ".jpg")

    cv2.imwrite(img1_path, new_img1)
    cv2.imwrite(img2_path, new_img2)

    # Write the labels
    label1_path = os.path.join(des_label_dir, img1_name + ".txt")
    label2_path = os.path.join(des_label_dir, img2_name + ".txt")

    write_bboxes(categories1, new_bboxes1, label1_path, "YOLO", new_w, new_h)
    write_bboxes(categories2, new_bboxes2, label2_path, "YOLO", new_w, new_h)


def convert_images(src_image_paths, src_label_paths, des_image_dir, des_label_dir, distortion_coefficient=0.5, crop=True):
    """
        Convert multiple images and labels
    """
    num_images = len(src_image_paths)
    num_labels = len(src_label_paths)

    assert num_images == num_labels
    print("[INFO]: Converting images and labels ...")

    for i in tqdm(range(num_images)):
        convert_one_image_and_box(src_image_paths[i], src_label_paths[i], des_image_dir, des_label_dir, distortion_coefficient, crop)
    
    print("[INFO] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Synthetic VisDrone dataset")
    parser.add_argument("--src_path", type=str, default="../visdrone", help="Path to VisDrone dataset")
    parser.add_argument("--trg_path", type=str, default="../synthetic_visdrone", help="Path to Synthetic VisDrone dataset")
    parser.add_argument("--distortion", type=float, default=0.5, help="Distortion Coefficient")
    parser.add_argument("--data_type", type=str, default="train", help="Which VisDrone dataset to convert, train or val")
    parser.add_argument("--crop", type=bool, default=True, help="Whether to crop the result images")

    args = parser.parse_args()
    src_data_dir = args.src_path
    trg_data_dir = args.trg_path
    data_type = args.data_type
    distortion_coefficient = args.distortion
    crop = args.crop

    if not os.path.exists(trg_data_dir):
        os.mkdir(trg_data_dir)

    src_data_dir = os.path.join(src_data_dir, data_type)
    trg_data_dir = os.path.join(trg_data_dir, data_type)

    assert(os.path.exists(src_data_dir))
    if not os.path.exists(trg_data_dir):
        os.mkdir(trg_data_dir)

    src_images_dir = os.path.join(src_data_dir, "images")
    src_labels_dir = os.path.join(src_data_dir, "labels")
    assert(os.path.exists(src_images_dir))
    assert(os.path.exists(src_labels_dir))

    trg_images_dir = os.path.join(trg_data_dir, "images")
    trg_labels_dir = os.path.join(trg_data_dir, "labels")

    if not os.path.exists(trg_images_dir):
        os.mkdir(trg_images_dir)

    if not os.path.exists(trg_labels_dir):
        os.mkdir(trg_labels_dir)

    # Loop through the image dir
    print("[INFO]: Getting original images and labels' paths")
    _, _, img_list = next(os.walk(src_images_dir))
    src_images_path = []
    src_labels_path = []
    for img_file in tqdm(img_list):
        img_name = img_file.split(".")[0]
        label_file = img_name + ".txt"
        img_path = os.path.join(src_images_dir, img_file)
        label_path = os.path.join(src_labels_dir, label_file)
        src_images_path.append(img_path)
        src_labels_path.append(label_path)
    
    convert_images(src_images_path, src_labels_path, trg_images_dir, trg_labels_dir, distortion_coefficient, crop)