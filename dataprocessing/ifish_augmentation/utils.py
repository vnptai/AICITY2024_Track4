import os
from ifisheye import *
import numpy as np
import cv2
from tqdm import tqdm


def convert_image(img, distortion_coefficient, crop=True):
    """
        Convert an ordinary image to fisheye image
        - Params:
            img                     : the original image
            distortion_coefficient  : distortion coefficient
            crop                    : whether to crop the dark area around images

        - Returns:
            new_img                 : newly generated fisheye image

    """
    new_img = fish(img, distortion_coefficient)
    if not crop:
        return new_img

    height, width, channel = img.shape

    # calculate the coordinates of the furthest point to the left and up
    left = (0.0, float(height/2))
    top = (float(width/2), 0.0)

     # normalize the coordinates
    left = ((2*left[0] - width)/width, (2*left[1] - height)/height)
    top = ((2*top[0] - width)/width, (2*top[1] - height)/height)

    # calculate the new coordinates
    new_left_x, new_left_y = reverse_fish_xn_yn(left[0], left[1], np.sqrt(left[0]**2 + left[1]**2), distortion_coefficient)
    new_top_x, new_top_y = reverse_fish_xn_yn(top[0], top[1], np.sqrt(top[0]**2 + top[1]**2), distortion_coefficient)
    
    # un-normalize the new coordinates
    left = (int((new_left_x + 1)*width/2), int((new_left_y + 1)*height/2))
    top = (int((new_top_x + 1)*width/2), int((new_top_y + 1)*height/2))
    
    new_img = new_img[top[1]:(height-top[1]), left[0]:(width-left[0]), :]

    return new_img


def convert_bboxes(bboxes, old_size, new_size, distortion_coefficient, crop=True):
    """
        Convert bboxes coordinates in ordinary images to corresponding fisheye images
        - Params:
            - bboxes: list of bbox in xyxy format [left, top, right, bottom] (unnormalized)
            - old_size: original size of the image (w, h)
            - new_size: the size of the newly converted image (w, h)
            - distortion_coefficient:
            - crop: whether to crop the dark area around images

        - Returns:
            - new_bboxes: new bounding boxes' coordinates
    """
    old_w, old_h = old_size
    new_w, new_h = new_size
    left_margin = int((old_w - new_w)//2)
    top_margin = int((old_h - new_h)//2)

    new_bboxes = []
    for bbox in bboxes:
        # top_left, top_right, bottom_left, bottom_right
        bbox_x = np.array([bbox[0], bbox[2], bbox[0], bbox[2]]).astype(float)
        bbox_y = np.array([bbox[1], bbox[1], bbox[3], bbox[3]]).astype(float)

        rd = np.zeros_like(bbox_x)
        bbox_x_fish = np.zeros_like(bbox_x)
        bbox_y_fish = np.zeros_like(bbox_y)

        # Calculate the new coodinates individually
        for i in range(4):
            bbox_x[i], bbox_y[i] = (2*bbox_x[i] - old_w)/old_w, (2*bbox_y[i] - old_h)/old_h
            rd[i] = np.sqrt(bbox_x[i]**2 + bbox_y[i]**2)
            bbox_x_fish[i], bbox_y_fish[i] = reverse_fish_xn_yn(bbox_x[i], bbox_y[i], rd[i], distortion_coefficient)
            bbox_x_fish[i], bbox_y_fish[i] = int(((bbox_x_fish[i] + 1)*old_w)/2), int(((bbox_y_fish[i] + 1)*old_h)/2)
        

        if crop:
            left_fish = int(min(bbox_x_fish)) - left_margin
            top_fish = int(min(bbox_y_fish)) - top_margin
            right_fish = int(max(bbox_x_fish)) - left_margin
            bot_fish = int(max(bbox_y_fish)) - top_margin
            new_bboxes.append([left_fish, top_fish, right_fish, bot_fish])
        else:
            left_fish = int(min(bbox_x_fish))
            top_fish = int(min(bbox_y_fish))
            right_fish = int(max(bbox_x_fish))
            bot_fish = int(max(bbox_y_fish))
            new_bboxes.append([left_fish, top_fish, right_fish, bot_fish])

    return new_bboxes


def split_image(img):
    """
        Split an image into 2 squared images
        - Params:
            img: the original image (np.ndarray)

        - Return:
            img1, img2: newly splited images
    """
    height, width, channel = img.shape
    if width == height:
        return img
    
    elif width > height:
        img1 = img[:, :height, :]
        img2 = img[:, -height:, :]
        return img1, img2
    
    else:
        img1 = img[:width, :, :]
        img2 = img[-width:, :, :]
        return img1, img2
    

def split_bboxes(categories, bboxes, img):
    """
        Split the bounding boxes
        - Params:

    """
    height, width, channel = img.shape
    if width == height:
        return bboxes
    
    elif width > height:
        bboxes1 = []
        bboxes2 = []
        categories1 = []
        categories2 = []
        num_bboxes = len(bboxes)
        for i in range(num_bboxes):
            bbox = bboxes[i]
            if bbox[0] <= height:
                bboxes1.append([bbox[0], bbox[1], min(bbox[2], height), bbox[3]])
                categories1.append(categories[i])
            if bbox[2] >= (width - height):
                left = max(bbox[0] - width + height, 0)
                bboxes2.append([left, bbox[1], bbox[2] - width + height, bbox[3]])
                categories2.append(categories[i])

        return bboxes1, bboxes2, categories1, categories2
    else:
        # Implement later
        pass


def write_bboxes(categories, bboxes, out_file, format="YOLO", img_w=None, img_h=None):
    """
        Write label to file:
        - Params:
            categories: list of bounding boxes' classes
            bboxes: bounding boxes coordinates in xyxy format
            out_file: path to the output_file
            format: dataset format
    """
    if format == "YOLO":
        assert img_w is not None
        assert img_h is not None
        f = open(out_file, "w")
        num_bboxes = len(bboxes)
        for i in range(num_bboxes):
            category = categories[i]
            left = bboxes[i][0]
            top = bboxes[i][1]
            right = bboxes[i][2]
            bot = bboxes[i][3]

            # Normalize bboxes
            center_x = float((left + right)/2)/img_w
            center_y = float((top + bot)/2)/img_h
            bbox_w = float(right - left)/img_w
            bbox_h = float(bot - top)/img_h

            center_x = round(center_x, 6)
            center_y = round(center_y, 6)
            bbox_w = round(bbox_w, 6)
            bbox_h = round(bbox_h, 6)

            f.write("{} {} {} {} {}\n".format(category, center_x, center_y, bbox_w, bbox_h))

        f.close()