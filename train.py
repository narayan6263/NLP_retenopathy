"""
Train the EfficientNet + CBAM + LAP model on the DR dataset.
Usage:
    python -m src.train --train_dir data/train --val_dir data/val
"""

import argparse
import tensorflow as tf
from .model import build_model, compile_model, unfreeze_for_finetune
from .preprocessing import get_datasets


def main(args):
    train_ds, val_ds = get_datasets(args.train_dir, args.val_dir, batch_size=args.batch_size)

    model, backbone = build_model(num_classes=args.num_classes, freeze_backbone=True)
    model = compile_model(model, lr=1e-3)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("outputs/best_model.keras", save_best_only=True, monitor="val_auc", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    print("Stage 1: training classifier head (backbone frozen)")
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=callbacks)

    print("Stage 2: fine-tuning top of EfficientNet backbone")
    model = unfreeze_for_finetune(model, backbone, num_layers_to_unfreeze=30)
    model = compile_model(model, lr=1e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs, callbacks=callbacks)

    model.save("outputs/final_model.keras")
    print("Saved model to outputs/final_model.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="data/train")
    parser.add_argument("--val_dir", default="data/val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--head_epochs", type=int, default=10)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    main(parser.parse_args())
