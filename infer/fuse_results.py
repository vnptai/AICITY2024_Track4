from mmengine.utils import ProgressBar
from pycocotools.coco import COCO
from mmengine.fileio import dump, load
from mmdet.models.utils import weighted_boxes_fusion

annotation = '../dataset/json_labels/val.json'

pred_results = ['./CO-DETR/work_dirs/infer_pseudo.bbox.json',
                './CO-DETR/work_dirs/infer_all.bbox.json',
                './CO-DETR/work_dirs/infer_fold0.bbox.json',
                '../evaluation/yolor_w6_vis+fis_conf_001_iou_065.json',
                '../evaluation/interimage_vis_fis.json',
                '../evaluation/yolov9_vis+fis_conf_001_iou_075_val.json',
                '../evaluation/codetr_synthetic_visdrone.json']###
out_file = 'final.json'
weights = [9,4,4,3,2,3,4]

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
dump(result, file=out_file)

