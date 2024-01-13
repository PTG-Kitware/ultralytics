# Inference
## Save to kwcoco
```bash
$ python ptg_detect.py --tasks m2  --split val  --weights /data/PTG/medical/bbn_data/M2_M3_M5_R18_v9/Model/weights/best.pt   --device 0  --project /data/PTG/medical/training/yolo_object_detector/detect/  --name bbn_model_m2_m3_m5_r18_v9  --save-img
```

## Default CLI
```bash
$ yolo detect predict model=<path_to_best.pt> project=<string_id> conf=<threshold> source=<path_to_image>
```
