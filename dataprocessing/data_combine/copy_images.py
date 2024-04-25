import os
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy Fisheye8k and VisDrone images into a single folder")
    parser.add_argument("--fisheye8k", type=str, default="./dataset/fisheye8k", help="Path to fisheye8k dataset")
    parser.add_argument("--visdrone", type=str, default="./dataset/visdrone/VisDrone2019-DET-train", help="Path to visdrone dataset")
    parser.add_argument("--synthetic", type=str, default="./dataset/synthetic_visdrone/VisDrone2019-DET-train", help="Path to synthetic visdrone dataset")
    parser.add_argument("--fisheye_eval", type=str, default="./dataset/fisheye_test", help="Path to fisheyeEval1k test dataset")
    parser.add_argument("--des_dir", type=str, default="./dataset/all_images" , help="Path to destination directory")

    args = parser.parse_args()

    fisheye_8k_dir = args.Fisheye8k
    visdrone_dir = args.visdrone
    synthetic_dir = args.synthetic
    fisheye_eval_dir = args.fisheye_eval
    des_dir = args.des_dir

    img_dirs = [
        os.path.join(fisheye_8k_dir, "train", "images"),
        os.path.join(fisheye_8k_dir, "test", "images"),
        os.path.join(visdrone_dir, "images"),
        os.path.join(synthetic_dir, "images"),
        os.path.join(fisheye_eval_dir, "images"),
        os.path.join(fisheye_eval_dir, "images1"),
    ]

    if not os.path.exists(des_dir):
        os.mkdir(des_dir)

    for img_dir in img_dirs:
        _, _, img_list = next(os.walk(img_dir))
        for img_file in img_list:
            src_img_path = os.path.join(img_dir, img_file)
            des_img_path = os.path.join(des_dir, img_file)
            os.system("cp {} {}".format(src_img_path, des_img_path))