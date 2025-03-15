import os
import numpy as np
import random
import tensorflow as tf
from PIL import Image
import tqdm
from defaults import get_cfg_defaults
import logging
import sys

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
    def resize_image(image, size=(128, 128)):
        return np.array(image.resize(size, Image.ANTIALIAS))

    # Process and save each fold
    for i in range(folds):
        images = []
        for img_file in tqdm.tqdm(evox_folds[i], desc=f"Processing fold {i+1}/{folds}"):
            img_path = os.path.join(image_folder, img_file)
            try:
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                image_resized = resize_image(image, size=(128, 128))  # Resize
                images.append((img_file, image_resized[np.newaxis, ...]))  # Add an extra channel
            except Exception as e:
                print(f"Skipping image {img_file} due to error: {e}")

        # Write fold to TFRecord
        tfr_opt = tf.io.TFRecordOptions(compression_type="")
        part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i) if train else cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i)
        tfr_writer = tf.io.TFRecordWriter(part_path, tfr_opt)

        for img_file, image in images:
            label = str(img_file.split('_')[1])  # Extract brand from filename and use as a label
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()])),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))  
            }))
            tfr_writer.write(ex.SerializeToString())

        tfr_writer.close()
        print(f"Fold {i+1} saved to {part_path}")

def run():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare TFRecords for Evox Cars dataset (Grayscale).")
    parser.add_argument(
        "--config-file",
        default="configs/evox.yaml",  # Ensure this points to your config file
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
