import os
import numpy as np
import random
import tensorflow as tf
from PIL import Image
import tqdm
from defaults import get_cfg_defaults
import logging
import sys
import torch
import torch.nn.functional as F

def prepare_evox(cfg, logger, train=True):
    # Define directory
    directory = os.path.dirname(cfg.DATASET.PATH) if train else os.path.dirname(cfg.DATASET.PATH_TEST)
    os.makedirs(directory, exist_ok=True)
    print(f"Saving TFRecords to: {cfg.DATASET.PATH if train else cfg.DATASET.PATH_TEST}")

    # Get list of PNG files
    if train:
        image_folder = os.path.dirname(cfg.DATASET.PATH)
    else:
        image_folder = os.path.dirname(cfg.DATASET.PATH_TEST)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Shuffle the list of image files
    random.seed(0)
    random.shuffle(image_files)
    count = len(image_files)
    print(f"Total images: {count}")

    # Number of folds
    folds = cfg.DATASET.PART_COUNT
    evox_folds = [[] for _ in range(folds)]

    # Evenly distribute images into folds
    count_per_fold = count // folds
    for i in range(folds):
        evox_folds[i] = image_files[i * count_per_fold: (i + 1) * count_per_fold]

    # Resize function
    def resize_image(image, size):
        return np.array(image.resize((size, size), Image.ANTIALIAS))

    # Process and save each fold
    for i in range(folds):
        images = []
        for img_file in tqdm.tqdm(evox_folds[i], desc=f"Processing fold {i+1}/{folds}"):
            img_path = os.path.join(image_folder, img_file)
            try:
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                image_resized = resize_image(image, size=cfg.DATASET.MAX_RESOLUTION_LEVEL)  # Resize to max resolution 
                image_resized = np.expand_dims(image_resized, axis=0)  # Ensure (1, MAX_RESOLUTION_LEVEL, MAX_RESOLUTION_LEVEL)
                images.append((img_file, image_resized))  # Store (filename, image)
            except Exception as e:
                print(f"Skipping image {img_file} due to error: {e}")

        # Write high-res images to TFRecords
        tfr_opt = tf.io.TFRecordOptions(compression_type="")
        part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i) if train else cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i)
        tfr_writer = tf.io.TFRecordWriter(part_path, tfr_opt)

        for img_file, image in images:
            label = str(img_file.split('_')[1])  # Extract label from filename
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()])),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))  
            }))
            tfr_writer.write(ex.SerializeToString())

        tfr_writer.close()
        print(f"Fold {i+1} saved at max resolution to {part_path}")

        # ✅ **Save Lower Resolution Versions**
        upper_bound = cfg.DATASET.MAX_RESOLUTION_LEVEL - 1
        for res_power in range(upper_bound, 1, -1):  # Loop for 2^6 (64x64) → 2^2 (4x4)
            images_down = []
            for img_file, image in tqdm.tqdm(images, desc=f"Downscaling fold {i+1} to {2**res_power}x{2**res_power}"):
                h, w = image.shape[1], image.shape[2]
                image_tensor = torch.tensor(image, dtype=torch.float32).view(1, 1, h, w)  # Convert to Tensor

                # Downscale using average pooling
                image_down = F.avg_pool2d(image_tensor, 2, 2).clamp_(0, 255).to(torch.uint8)
                image_down = image_down.view(cfg.MODEL.CHANNELS, h // 2, w // 2).numpy()

                images_down.append((img_file, image_down))

            # Save lower resolution version
            part_path = cfg.DATASET.PATH % (res_power, i) if train else cfg.DATASET.PATH_TEST % (res_power, i)
            tfr_writer = tf.io.TFRecordWriter(part_path, tfr_opt)

            for img_file, image in images_down:
                label = str(img_file.split('_')[1])  # Extract label from filename
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))  
                }))
                tfr_writer.write(ex.SerializeToString())

            tfr_writer.close()
            print(f"Fold {i+1}, resolution {2**res_power} saved to {part_path}")

            images = images_down  # Use lower-resolution images for next step

def run():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare TFRecords for Evox Cars dataset (Grayscale).")
    parser.add_argument(
        "--config-file",
        default="configs/evox_64x64_1.yaml",  # Ensure this points to your config file
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # Load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Set up logging
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("Running with config:\n{}".format(cfg))

    # Prepare dataset for both training and testing
    prepare_evox(cfg, logger, train=True)
    prepare_evox(cfg, logger, train=False)

if __name__ == '__main__':
    run()
