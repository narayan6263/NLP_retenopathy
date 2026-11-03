"""
Builds the DR-detection model: EfficientNetB0 (pretrained) backbone
-> CBAM attention -> Local Attention Pooling -> classifier head.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from .layers import CBAM, LocalAttentionPooling


def build_model(input_shape=(224, 224, 3), num_classes=5, dropout=0.3, freeze_backbone=True):
    inputs = layers.Input(shape=input_shape)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs
    )
    backbone.trainable = not freeze_backbone

    x = backbone.output                 # (H, W, C) feature map
    x = CBAM(reduction_ratio=8, name="cbam")(x)
    x = LocalAttentionPooling(name="lap")(x)   # -> (batch, C) instead of GAP

    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dr_grade")(x)

    model = models.Model(inputs, outputs, name="EfficientNet_CBAM_LAP")
    return model, backbone


def compile_model(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=True),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_for_finetune(model, backbone, num_layers_to_unfreeze=30):
    """Call after initial head-training to fine-tune the top of the backbone."""
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    return model
