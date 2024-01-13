import os
import random
import kwcoco
import glob
import argparse
import cv2

import ubelt as ub

from ultralytics import YOLO
from PIL import Image
from pathlib import Path

from angel_system.data.common.load_data import Re_order
from angel_system.data.common.load_data import time_from_name
from angel_system.data.medical.data_paths import grab_data # Domain specific


def data_loader(tasks, split):
    """Create a list of all videos in the tasks for the given split

    :return: List of absolute paths to video folders
    """
    training_split = {
        split: []
    }

    for task in tasks:
        ( ptg_root,
        task_data_dir,
        task_activity_config_fn,
        task_activity_gt_dir,
        task_ros_bags_dir,
        task_training_split,
        task_obj_dets_dir,
        task_obj_config ) = grab_data(task, "gyges")
        training_split = {key: value + task_training_split[key] for key, value in training_split.items()}

    print("\nTraining split:")
    for split_name, videos in training_split.items():
        print(f"{split_name}: {len(videos)} videos")
        print([os.path.basename(v) for v in videos])
    print("\n")

    videos = training_split[split]
    return videos

def detect(args):
    save_dir = f"{args.project}/{args.name}"
    save_images_dir = f"{save_dir}/images"
    Path(save_images_dir).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    dset = kwcoco.CocoDataset()

    # Add categories
    for i, object_label in names.items():
        if object_label == "background":
            continue
        dset.add_category(name=object_label, id=i)

    videos = data_loader(args.tasks, args.split)
    for video in videos:
        video_name = os.path.basename(video)
        video_data = {
            "name": video_name,
            #"task": video_task,
        }
        vid = dset.add_video(**video_data)

        images = glob.glob(f"{video}/images/*.png")
        if not images:
            warnings.warn(f"No images found in {video_name}")
        images = Re_order(images, len(images))
        for image_fn in ub.ProgIter(images, desc=f"images in {video_name}"):
            fn = os.path.basename(image_fn)
            img0 = cv2.imread(image_fn)  # BGR
            assert img0 is not None, 'Image Not Found ' + image_fn
            height, width = img0.shape[:2]

            frame_num, time = 0, 0#time_from_name(image_fn)

            image = {
                "file_name": image_fn,
                "video_id": vid,
                "frame_index": frame_num,
                "width": width,
                "height": height,
            }
            img_id = dset.add_image(**image)

            results = model.predict(
                source=image_fn,
                conf=args.conf_thr,
                imgsz=args.img_size,
                device=args.device,
                verbose=False
            )[0] # list of length=num images
            
            if args.save_img:
                im_array = results.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                #im.show()  # show image
                video_images_dir = f"{save_images_dir}/video_{vid}"
                Path(video_images_dir).mkdir(parents=True, exist_ok=True)
                im.save(f"{video_images_dir}/{fn}")  # save image
            
            for bbox in results.boxes:
                cxywh = bbox.xywh.tolist()[0] # center, w, h
                xywh = [cxywh[0] - (cxywh[2] / 2), cxywh[1] - (cxywh[3] / 2),
                        cxywh[2], cxywh[3]]
                cls_id = int(bbox.cls.item())
                cls_name = names[cls_id]
                conf = bbox.conf.item()

                ann = {
                    "area": xywh[2] * xywh[3],
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": xywh,
                    "confidence": float(conf),
                }
                dset.add_annotation(**ann)

    # Save
    dset.fpath = f"{save_dir}/{args.name}_{args.split}_obj_results.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved predictions to {dset.fpath}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default='coffee',
        help='Dataset(s)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Data split to run on'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='yolov7.pt',
        help='model.pt path(s)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='inference size (pixels)'
    )
    parser.add_argument(
        '--conf-thr',
        type=float,
        default=0.7,
        help='object confidence threshold'
    )
    parser.add_argument(
        '--device',
        default='',
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    )
    parser.add_argument(
        '--project',
        default='runs/detect',
        help='save results to project/name'
    )
    parser.add_argument(
        '--name',
        default='exp',
        help='save results to project/name'
    )
    parser.add_argument(
        '--save-img',
        action='store_true',
        help='save results to *.png'
    )

    args = parser.parse_args()
    print(args)

    detect(args)

if __name__ == '__main__':
    main()