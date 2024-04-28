import json
import os
from tqdm import tqdm

json_list = [
    "./dataset/visdrone/train.json",
    "./dataset/fisheye8k/train/train.json",
]

new_json_file = "./dataset/visdrone_fisheye.json"
img2anno = {}
img2wh = {}

num_json_files = len(json_list)
for file_id, json_file in enumerate(json_list):
    print("INFO: Processing file id {}/{}".format(file_id, num_json_files))
    assert(os.path.exists(json_file))
    with open(json_file, "r") as f:
        json_data = json.load(f)

    categories = json_data["categories"]
    images = json_data["images"]
    annotations = json_data["annotations"]

    print("[INFO] Mapping annotations to image file")
    id2img = {}
    for image in tqdm(images):
        id2img[image["id"]] = image["file_name"]
        img2wh[image["file_name"]] = [image["width"], image["height"]]
        img2anno[image["file_name"]] = []

    for annotation in tqdm(annotations):
        image_id = annotation["image_id"]
        img_file = id2img[image_id]
        img2anno[img_file].append(annotation)

print("[INFO]: Composing new json file...")
print("[WARNING]: The images and annotations' ids will be reset!")
image_id = 1
anno_id = 1
images = []
annotations = []
for img_file in tqdm(img2anno.keys()):
    images.append({
        "id": image_id,
        "file_name": img_file,
        "width": img2wh[img_file][0],
        "height": img2wh[img_file][1], 
    })
    
    for annotation in img2anno[img_file]:
        annotation["image_id"] = image_id
        annotation["id"] = anno_id
        anno_id += 1
        annotations.append(annotation)
    
    image_id += 1

new_json_data = {}
new_json_data["categories"] = categories
new_json_data["images"] = images
new_json_data["annotations"] = annotations

with open(new_json_file, "w") as f:
    json.dump(new_json_data, f)