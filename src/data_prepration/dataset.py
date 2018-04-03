from image_downloader import parse_data
import os
import numpy as np
import scipy.misc
import tensorflow as tf
import sys
sys.path.append("/data/slim/models/research/slim/")
from preprocessing import inception_preprocessing


def load_image_batch(key_path_pairs, height, width):
    """
    Load image batch from CSV file with imageIds and 4-D ndarray
    :return:
    """
    image_list = []
    id_list = []
    for (key, path) in key_path_pairs:
        if os.path.exists(path):
            image_raw = tf.cast(np.array(scipy.misc.imread(path, flatten=False, mode='RGB')), tf.float32)

            image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=False)
            print(image.shape)
            image_list.append(image)
            id_list.append(key)
        else:
            continue
    print(image_list[0])
    return id_list, image_list


def load_image_list(csv_file, mode):
    key_url_pairs = parse_data(csv_file)
    image_ids = []
    image_paths = []
    for (key, url) in key_url_pairs:
        image_path = os.path.join("/data/landmarks", mode, key + ".jpg")
        if os.path.exists(image_path):
            image_ids.append(key)
            image_paths.append(image_path)
        else:
            continue
    return image_ids, image_paths


if __name__ == '__main__':
    ids, paths = load_image_list("/data/landmarks/test.csv", "test")
    print(len(ids))
