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

    # Get list of PNG files
    image_folder = 'data/datasets/evox/cars'  # Replace this with the correct folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    # Shuffle the list of image files
    random.shuffle(image_files)
    count = len(image_files)
    print(f"Total images: {count}")
    
    # Function to resize images to desired size (e.g., 128x128 or 256x256)
    def resize_image(image, size=(128, 128)):
        return np.array(image.resize(size, Image.ANTIALIAS))

    images = []

    # Process images
    for img_file in tqdm.tqdm(image_files):
        img_path = os.path.join(image_folder, img_file)
        
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale ('L' mode for grayscale)
            image_resized = resize_image(image, size=(128, 128))  # Resize to desired size (128x128)
            images.append(image_resized[np.newaxis, ...])  # Add an extra dimension for channel (1, height, width)
        except Exception as e:
            print(f"Skipping image {img_file} due to error: {e}")

    # Prepare TFRecord Writer
    tfr_opt = tf.io.TFRecordOptions(compression_type="GZIP")
    #tfr_opt = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.NONE)
    tfr_writer = tf.io.TFRecordWriter(cfg.DATASET.PATH if train else cfg.DATASET.PATH_TEST, tfr_opt)

    # Write images to TFRecord
    for image in images:
        # Example features
        ex = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))
        }))
        tfr_writer.write(ex.SerializeToString())

    tfr_writer.close()


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

    # Prepare the dataset (train)
    prepare_evox(cfg, logger, train=True)


if __name__ == '__main__':
    run()
