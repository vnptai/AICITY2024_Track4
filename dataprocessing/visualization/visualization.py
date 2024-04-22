import cv2
import os
from tqdm import tqdm
import numpy as np
import argparse


class Visualizer():
    def __init__(self, labels, colors):
        self.labels = labels
        self.colors = colors

    """
        Visualize one bbox on the current image
        - Params:
            box: box coordinates (expect to be in xyxy format)
            img: current image

    """
    def visualize_bbox(self, box, img, show_label=False):
        box_coords = box[:4].astype(int)
        box_cls = int(box[4])
        if box_cls >= len(self.labels):
            return img
        if len(box) <= 5:
            box_conf = 1.0
        else: 
            box_conf = round(box[5], 2)

        color = self.colors[box_cls]
        label = self.labels[box_cls]

        # draw the bounding box
        upper_left = (box_coords[0], box_coords[1])
        lower_right = (box_coords[2], box_coords[3])

        cv2.rectangle(img, upper_left, lower_right, color, thickness=1, lineType=cv2.LINE_AA)
        
        # draw the label
        if show_label:
            (w, h), _ = cv2.getTextSize("{} {}".format(label, box_conf), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            img = cv2.rectangle(img, (box_coords[0], box_coords[1] - h), (box_coords[0] + w, box_coords[1]), color, thickness=-1)
            img = cv2.putText(img, "{} {}".format(label, box_conf), (box_coords[0], box_coords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
        
        return img

    """
        Visualize list of bounding boxes
    """
    def visualize_bboxes(self, boxes, img):
        for box in boxes:
            img = self.visualize_bbox(box, img)
        return img
    
# For testing
if __name__ == "__main__":
    # Visualize one image for testing
    parser = argparse.ArgumentParser(description="Visualization for YOLO-formated labels")
    parser.add_argument("--image_dir", type=str, help="image directory")
    parser.add_argument("--label_dir", type=str, help="label directory")
    parser.add_argument("--num_imgs", type=int, default=4, help="number of images to visualize")
    parser.add_argument("--save_dir", type=str, default="./data_processing/visualization", help="save directory")
    args = parser.parse_args()
    image_dir = args.image_dir
    label_dir = args.label_dir
    num_imgs = args.num_imgs

    save_dir = args.save_dir

    _, _, img_list = next(os.walk(image_dir))
    
    labels = [
        "bus",
        "bike",
        "car",
        "pedestrian",
        "truck"
    ]
    
    colors = [
        (255, 0, 0), # blue
        (0, 255, 0), # green
        (0, 0, 255), # red
        (255, 0, 255), # fuchsia
        (0, 255, 255), # yellow
        (128, 0, 128), # purple
        (128, 0, 0), # navy
    ]
    
    visualizer = Visualizer(labels, colors)
    
    for img_file in tqdm(img_list[:num_imgs]):
        image_name = img_file.split(".")[0]
        image_path = os.path.join(image_dir, image_name + ".jpg")
        label_path = os.path.join(label_dir, image_name + ".txt")
        img = cv2.imread(image_path)
        img_height, img_width, img_channel = img.shape
    
        with open(label_path, "r") as f:
            bboxes = f.readlines()
        
        for bbox in bboxes:
            args = bbox.split(" ")
            cls = int(args[0])
            center_x = float(args[1])
            center_y = float(args[2])
            width = float(args[3])
            height = float(args[4])
        
            left = int((center_x - width/2)*img_width)
            top = int((center_y - height/2)*img_height)
            right = int((center_x + width/2)*img_width)
            bottom = int((center_y + height/2)*img_height)
        
            bbox = np.array([left, top, right, bottom, cls])
        
            visualizer.visualize_bbox(bbox, img)
        
        cv2.imwrite(os.path.join(save_dir, img_file), img)