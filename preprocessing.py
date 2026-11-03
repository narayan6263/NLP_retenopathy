"""
Preprocessing + augmentation pipeline for retinal fundus images.
Expects a directory structure:
    data/train/<class_0..4>/*.png
    data/val/<class_0..4>/*.png
"""

import tensorflow as tf

IMG_SIZE = (224, 224)
NUM_CLASSES = 5  # 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR


def _preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


layers = tf.keras.layers

augmenter = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ],
    name="augmentation",
)


def get_datasets(train_dir, val_dir, batch_size=32, img_size=IMG_SIZE):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
    )

    train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y),
                             num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
