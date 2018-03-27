from image_downloader import parse_data
import os
import numpy as np
import scipy.misc
import tensorflow as tf
import sys
sys.path.append("/data/slim/models/research/slim/")
from preprocessing import inception_preprocessing


def load_image_batch(csv_file, height, width, mode):
    """
    Load image batch from CSV file with imageIds and 4-D ndarray
    :return:
    """
    key_url_pairs = parse_data(csv_file)
    image_list = []
    id_list = []
    for (key, url) in key_url_pairs:
        image_path = os.path.join("/data/landmarks", mode, key+".jpg")
        if os.path.exists(image_path):
            image_raw = tf.cast(np.array(scipy.misc.imread(image_path, flatten=False, mode='RGB')), tf.float32)

            image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=False)
            print(image.shape)
            image_list.append(image)
            id_list.append(key)
        else:
            #print("bad")
            continue
    print(image_list[0])
    return None


if __name__ == '__main__':
    load_image_batch("/data/landmarks/test.csv", 299, 299, "test")
