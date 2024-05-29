import tensorflow as tf
from config import *
import os

AUTOTUNE = tf.data.AUTOTUNE

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.cast(tf.equal(parts[-2], 'Real'), tf.float32)

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img /= 255.0
    return img, label

def data_augmentation(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label

def create_dataset(data_dir, training=True):
    data_dir = os.path.join(data_dir)
    list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*'), shuffle=True)

    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    if training and DATA_AUGMENTATION:
        labeled_ds = labeled_ds.map(data_augmentation, num_parallel_calls=AUTOTUNE)

    dataset = labeled_ds.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def get_datasets():
    train_dataset = create_dataset(TRAIN_DIR, training=True)
    validation_dataset = create_dataset(VALIDATION_DIR, training=False)
    return train_dataset, validation_dataset
