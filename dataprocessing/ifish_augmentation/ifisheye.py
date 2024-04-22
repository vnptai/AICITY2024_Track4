"""iFish utils.

Company: VNPT-IT.
Filename: ifisheye.py.
Datetime: 10/04/2024.
Description: Utilities for applying fisheye effects to ordinary images. Inspired by https://github.com/Gil-Mor/iFish.git
"""
import cv2
import numpy as np
import os


def get_fish_xn_yn(source_x, source_y, radius, distortion):
    """
        Converting a pixel's coordinates to its corresponding coordinates in fisheye image
        - Params:
            source_x    : pixel's x-coordinate
            source_y    : pixel's y-coordinate
            radius      : pixel's distance from the image center
            distortion  : distortion coefficient

        - Returns:
            fish_x      : pixel's new x-coordinate 
            fish_y      : pixel's new y-coordinate
    """
    if 1 - distortion*(radius**2) == 0:
        fish_x = source_x 
        fish_y = source_y
    else:
        fish_x = source_x/(1 - distortion*(radius**2))
        fish_y = source_y/(1 - distortion*(radius**2))

    return fish_x, fish_y


def img_pad_square(img, pad_value=0):
    """
        Add padding to the image to make it become a squared image
        - Params:
            img         : the original image
            pad_value   : padding value
        
        - Returns:
            img         : padded image
    """
    height, width, channel = img.shape
    if width >= height:
        border_width = (width - height)//2
        img = cv2.copyMakeBorder(img, border_width, border_width, 0, 0, cv2.BORDER_CONSTANT, value=pad_value)
    else:
        border_width = (height - width)//2
        img = cv2.copyMakeBorder(img, 0, 0, border_width, border_width, cv2.BORDER_CONSTANT, value=pad_value)
    return img


def fish(img, distortion_coefficient):
    """
        Convert normal image to fisheye image
        - Params:
            img                     : the original image
            distortion_coefficient  : distortion coefficient (should be between 0-1)
        - Returns:
    """
    width, height, channel = img.shape

    # RGB to RGBA
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.dstack((img, np.full((width, height), 255)))
    
    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # floats and calculations
    w, h = float(width), float(height)

    # easier calculation if we traverse x, y in dst image
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):

            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)

            # get xn and yn distance from normalized center
            rd = np.sqrt(xnd**2 + ynd**2)

            # new normalized pixel coordinates
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)

            # convert the normalized distorted xdn and ydn back to image pixels
            xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

            # if new pixel is in bounds copy from source pixel to destination pixel
            if (0 <= xu) and (xu < img.shape[0]) and (0 <= yu) and (yu < img.shape[1]):
                dstimg[x][y] = img[xu][yu]
    return dstimg.astype(np.uint8)


def reverse_fish_xn_yn(source_x, source_y, radius, distortion):
    """
        Converting a pixel's coordinates in fisheye image to its corresponding coordinates in the original image
        (The reverse function of get_fish_xn_yn)
        - Params:
            source_x    : pixel's x-coordinate
            source_y    : pixel's y-coordinate
            radius      : pixel's distance from the image center
            distortion  : distortion coefficient

        - Returns:
            fish_x      : pixel's new x-coordinate 
            fish_y      : pixel's new y-coordinate
    """
    if radius == 0:
        return source_x, source_y
    coefficient = (np.sqrt(1 + 4*distortion*(radius**2)) - 1)/(2*distortion*(radius**2))
    
    return source_x*coefficient, source_y*coefficient