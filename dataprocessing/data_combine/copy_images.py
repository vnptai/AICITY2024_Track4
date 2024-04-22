import os
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy Fisheye8k and VisDrone images into a single folder")
    parser.add_argument("--fisheye8k", type=str, default="./dataset/fisheye8k", help="Path to fisheye8k dataset")
    parser.add_argument("--visdrone", type=str, default="./dataset/visdrone/VisDrone2019-DET-train", help="Path to visdrone dataset")
    parser.add_argument("--fisheye_eval", type=str, default="./dataset/fisheye_test", help="Path to fisheyeEval1k test dataset")
    parser.add_argument("--des_dir", type=str, default="./dataset/fisheye_visdrone" , help="Path to destination directory")

    args = parser.parse_args()

    fisheye_8k_dir = args.Fisheye8k
    visdrone_dir = args.visdrone
    fisheye_eval_dir = args.fisheye_eval
    des_dir = args.des_dir

    
