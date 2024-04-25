from mmengine.utils import ProgressBar
from pycocotools.coco import COCO
from mmengine.fileio import dump, load
from mmdet.models.utils import weighted_boxes_fusion
import json

annotation = '../dataset/json_labels/val.json'

pred_results = ['./CO-DETR/work_dirs/infer_fold0.bbox.json',
                './CO-DETR/work_dirs/infer_fold1.bbox.json',
                './CO-DETR/work_dirs/infer_fold2.bbox.json']###

out_file = 'pseudo_label.json'
weights = [3,1,1]

fusion_iou_thr = 0.75
skip_box_thr = 0.15
# conf_type = 'avg'

cocoGT = COCO(annotation)

predicts_raw = []

models_name = ['model_' + str(i) for i in range(len(pred_results))]

for model_name, path in \
            zip(models_name, pred_results):
        pred = load(path)
        predicts_raw.append(pred)

predict = {
        str(image_id): {
            'bboxes_list': [[] for _ in range(len(predicts_raw))],
            'scores_list': [[] for _ in range(len(predicts_raw))],
            'labels_list': [[] for _ in range(len(predicts_raw))]
        }
        for image_id in cocoGT.getImgIds()
    }

for i, pred_single in enumerate(predicts_raw):
        for pred in pred_single:
            p = predict[str(pred['image_id'])]
            p['bboxes_list'][i].append([pred['bbox'][0], pred['bbox'][1], pred['bbox'][0] + pred['bbox'][2], pred['bbox'][1] + pred['bbox'][3]])
            # p['bboxes_list'][i].append(pred['bbox'])
            p['scores_list'][i].append(pred['score'])
            p['labels_list'][i].append(pred['category_id'])

result = []
prog_bar = ProgressBar(len(predict))
for image_id, res in predict.items():
    bboxes, scores, labels = weighted_boxes_fusion(
        res['bboxes_list'],
        res['scores_list'],
        res['labels_list'],
        weights=weights,
        iou_thr=fusion_iou_thr,
        skip_box_thr=skip_box_thr)

    for bbox, score, label in zip(bboxes, scores, labels):
        bbox_copy = bbox.numpy().tolist()
        bbox_copy[2] = bbox_copy[2] - bbox_copy[0]
        bbox_copy[3] = bbox_copy[3] - bbox_copy[1]
        result.append({
            'bbox': bbox_copy,
            'category_id': int(label),
            'image_id': int(image_id),
            'score': float(score)
        })
    prog_bar.update()

val_json_data = json.load(open(annotation))
predict_json_data = result
for predict_data in predict_json_data:
    if predict_data['score'] < 0.4: continue
    bbox_data = predict_data
    bbox_data['id'] = len(val_json_data['annotations']) + 1
    width, height = predict_data['bbox'][2], predict_data['bbox'][3]
    bbox_data['area'] = width * height
    bbox_data['iscrowd'] = 0
    val_json_data['annotations'].append(bbox_data)

with open(out_file, 'w') as f:
    json.dump(val_json_data, f)




