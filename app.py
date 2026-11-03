
# import os
# from pathlib import Path

# import gradio as gr
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# from src.layers import CBAM, LocalAttentionPooling

# # ================= PROJECT PATHS =================

# ROOT_DIR = Path(__file__).resolve().parent.parent

# MODEL_PATH = ROOT_DIR / "outputs" / "final_model.keras"
# TRAIN_DIR = ROOT_DIR / "data" / "train"

# CLASS_LABELS = {
#     "0": "No Diabetic Retinopathy",
#     "1": "Mild Diabetic Retinopathy",
#     "2": "Moderate Diabetic Retinopathy",
#     "3": "Severe Diabetic Retinopathy",
#     "4": "Proliferative Diabetic Retinopathy"
# }

# # ================= LOAD MODEL =================

# if not MODEL_PATH.exists():
#     raise FileNotFoundError(
#         f"Trained model not found: {MODEL_PATH}"
#     )

# print("Loading Diabetic Retinopathy model...")

# model = tf.keras.models.load_model(
#     MODEL_PATH,
#     custom_objects={
#         "CBAM": CBAM,
#         "LocalAttentionPooling": LocalAttentionPooling
#     },
#     compile=False
# )

# print("Model loaded successfully!")
# print("Model input shape:", model.input_shape)

# # Read class folders in the same sorted order
# # used by image_dataset_from_directory.

# if TRAIN_DIR.exists():
#     class_folders = sorted(
#         [p.name for p in TRAIN_DIR.iterdir() if p.is_dir()]
#     )
# else:
#     class_folders = ["0", "1", "2", "3", "4"]

# class_names = [
#     CLASS_LABELS.get(name, name)
#     for name in class_folders
# ]

# # Get trained model input dimensions
# IMG_HEIGHT = model.input_shape[1] or 224
# IMG_WIDTH = model.input_shape[2] or 224

# # ================= PREDICTION =================

# def predict_retinopathy(image):

#     if image is None:
#         return {}, "Please upload a retinal fundus image."

#     try:
#         image = image.convert("RGB")
#         image = image.resize((IMG_WIDTH, IMG_HEIGHT))

#         img_array = np.array(
#             image, dtype=np.float32
#         )

#         img_array = np.expand_dims(img_array, axis=0)

#         # EfficientNet preprocessing
#         img_array = tf.keras.applications.efficientnet.preprocess_input(
#             img_array
#         )

#         predictions = model.predict(
#             img_array, verbose=0
#         )[0]

#         if len(predictions) != len(class_names):
#             raise ValueError(
#                 "Number of model outputs does not match class folders."
#             )

#         scores = {
#             class_names[i]: float(predictions[i])
#             for i in range(len(class_names))
#         }

#         predicted_index = int(np.argmax(predictions))
#         predicted_class = class_names[predicted_index]
#         confidence = float(predictions[predicted_index]) * 100

#         result = (
#             f"Prediction: {predicted_class}\n"
#             f"Confidence: {confidence:.2f}%\n\n"
#             "This is an AI-generated screening result, "
#             "not a medical diagnosis. Consult an eye specialist."
#         )

#         return scores, result

#     except Exception as e:
#         return {}, f"Prediction error: {str(e)}"


# # ================= GRADIO INTERFACE =================

# with gr.Blocks(title="Diabetic Retinopathy Detection") as demo:

#     gr.Markdown(
#         """
#         # Diabetic Retinopathy Detection

#         ### EfficientNetB0 + CBAM + Local Attention Pooling

#         Upload a retinal fundus image to predict the
#         diabetic retinopathy severity grade.
#         """
#     )

#     with gr.Row():

#         with gr.Column():
#             image_input = gr.Image(
#                 type="pil",
#                 label="Upload Retinal Image"
#             )

#             predict_button = gr.Button(
#                 "Predict Retinopathy",
#                 variant="primary"
#             )

#         with gr.Column():
#             prediction_output = gr.Label(
#                 label="Prediction Probabilities",
#                 num_top_classes=5
#             )

#             result_output = gr.Textbox(
#                 label="Detection Result",
#                 lines=5
#             )

#     predict_button.click(
#         fn=predict_retinopathy,
#         inputs=image_input,
#         outputs=[prediction_output, result_output]
#     )

#     gr.Markdown(
#         """
#         **Model:** EfficientNetB0 + CBAM + LAP

#         **Classes:** No DR, Mild, Moderate, Severe,
#         Proliferative DR.

#         **Disclaimer:** This project is for educational
#         and research purposes only.
#         """
#     )


# # ================= START SERVER =================

# if __name__ == "__main__":

#     print("Starting Diabetic Retinopathy application...")

#     demo.launch(
#         server_name="127.0.0.1",
#         server_port=7860
#     )




import os
import numpy as np
import gradio as gr
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.report_generator import create_report
from src.model import CBAM, LocalAttentionPooling


# -------------------------------
# CONFIGURATION
# -------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "outputs", "final_model.keras"
)

IMG_SIZE = (224, 224)

GRADE_LABELS = {
    0: "No DR",
    1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR",
    3: "Severe Non-Proliferative DR",
    4: "Proliferative DR"
}


# -------------------------------
# LOAD MODEL
# -------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "CBAM": CBAM,
        "LocalAttentionPooling": LocalAttentionPooling
    },
    compile=False
)

print("Model loaded successfully!")


# -------------------------------
# PREPROCESS IMAGE
# -------------------------------

def prepare_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# -------------------------------
# PREDICTION + NLG
# -------------------------------

def predict_and_generate_report(image_path):

    if image_path is None:
        raise gr.Error("Please upload a retinal image.")

    try:

        processed_image = prepare_image(image_path)

        prediction = model.predict(
            processed_image,
            verbose=0
        )

        probabilities = np.asarray(
            prediction
        ).reshape(-1)

        if len(probabilities) != 5:
            raise ValueError(
                "Expected 5 output classes from the model."
            )

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Invalid model output.")

        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Model output is not a probability vector.")

        predicted_grade = int(np.argmax(probabilities))

        predicted_label = GRADE_LABELS[predicted_grade]

        confidence = float(
            probabilities[predicted_grade] * 100
        )

        screening_category = (
            "Referable DR"
            if predicted_grade >= 2
            else "Non-Referable DR"
        )

        prediction_text = f"""
Predicted Grade: {predicted_grade}

Classification: {predicted_label}

Screening Category: {screening_category}

Highest Class Probability: {confidence:.2f}%
"""

        probability_text = "\n".join([
            f"Grade {i} - {GRADE_LABELS[i]}: "
            f"{probabilities[i] * 100:.2f}%"
            for i in range(5)
        ])

        # NLG report generation
        report_text, pdf_path = create_report(
            grade=predicted_grade,
            probabilities=probabilities,
            image_name=os.path.basename(image_path)
        )

        return (
            prediction_text,
            probability_text,
            report_text,
            pdf_path
        )

    except gr.Error:
        raise

    except Exception as error:
        raise gr.Error(
            f"Prediction or report generation failed: {error}"
        )


# -------------------------------
# GRADIO INTERFACE
# -------------------------------

with gr.Blocks(
    title="DR Detection with NLG"
) as demo:

    gr.Markdown("""
    # Diabetic Retinopathy Detection
    ### Deep Learning + Natural Language Generation

    Upload a retinal fundus image to predict the DR grade
    and automatically generate a screening report.
    """)

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="filepath",
                label="Upload Retinal Image"
            )

            predict_button = gr.Button(
                "Predict & Generate Report",
                variant="primary"
            )

        with gr.Column():

            prediction_output = gr.Textbox(
                label="Prediction Results",
                lines=6
            )

            probability_output = gr.Textbox(
                label="Class-wise Probabilities",
                lines=7
            )

    report_output = gr.Textbox(
        label="Generated NLG Report",
        lines=15
    )

    pdf_output = gr.File(
        label="Download PDF Report"
    )

    predict_button.click(
        fn=predict_and_generate_report,
        inputs=image_input,
        outputs=[
            prediction_output,
            probability_output,
            report_output,
            pdf_output
        ]
    )

    gr.Markdown("""
    **Disclaimer:** This is an academic AI research prototype.
    It does not provide a medical diagnosis. Please consult
    a qualified ophthalmologist.
    """)


# -------------------------------
# RUN APP
# -------------------------------

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )
