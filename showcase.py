"""
Loads the trained model, evaluates it on the validation set, and saves:
  - outputs/confusion_matrix.png  (Accuracy/F1/Precision/Recall/ROC-AUC/QWK printed to console)
  - outputs/gradcam_examples.png  (a few real predictions with attention overlays)

Usage:
    python -m src.showcase --val_dir data/val --model_path outputs/final_model.keras
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .layers import CUSTOM_OBJECTS
from .evaluate import evaluate_predictions
from .gradcam import make_gradcam_heatmap


def main(args):
    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path, custom_objects=CUSTOM_OBJECTS)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.val_dir, image_size=(224, 224), batch_size=32,
        label_mode="categorical", shuffle=False,
    )
    class_names = val_ds.class_names
    val_ds_preprocessed = val_ds.map(
        lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(x), y)
    )

    print("Running predictions on the validation set...")
    y_true, y_probs = [], []
    for x_batch, y_batch in val_ds_preprocessed:
        preds = model.predict(x_batch, verbose=0)
        y_probs.append(preds)
        y_true.append(np.argmax(y_batch.numpy(), axis=1))
    y_true = np.concatenate(y_true)
    y_probs = np.concatenate(y_probs)

    print("\n=== Evaluation Metrics ===")
    results, cm = evaluate_predictions(y_true, y_probs, class_names=class_names)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("Saved confusion matrix to outputs/confusion_matrix.png")

    print("\nGenerating Grad-CAM examples...")
    fig, axes = plt.subplots(1, args.num_examples, figsize=(4 * args.num_examples, 4))
    shown = 0
    for x_batch, y_batch in val_ds.take(1):
        for i in range(min(args.num_examples, x_batch.shape[0])):
            img = x_batch[i].numpy()
            img_input = tf.keras.applications.efficientnet.preprocess_input(
                np.expand_dims(img.copy(), axis=0)
            )
            heatmap, pred_class = make_gradcam_heatmap(img_input, model, last_conv_layer_name="top_conv")

            ax = axes[shown] if args.num_examples > 1 else axes
            ax.imshow(img.astype("uint8"))
            ax.imshow(heatmap, cmap="jet", alpha=0.4, extent=(0, 224, 224, 0))
            true_label = class_names[np.argmax(y_batch[i].numpy())]
            pred_label = class_names[pred_class]
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
            ax.axis("off")
            shown += 1
        break

    plt.tight_layout()
    plt.savefig("outputs/gradcam_examples.png", dpi=150)
    print("Saved Grad-CAM examples to outputs/gradcam_examples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", default="data/val")
    parser.add_argument("--model_path", default="outputs/final_model.keras")
    parser.add_argument("--num_examples", type=int, default=4)
    main(parser.parse_args())
