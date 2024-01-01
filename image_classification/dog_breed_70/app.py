import gradio as gr
import tensorflow as tf
import numpy as np

H, W = 128, 128

model = tf.keras.models.load_model("b1_model_ft.h5")
class_names = np.load("dog_breed_class_names.npy")


def preprocess_image(image):
    image = tf.cast(tf.image.resize(image, size=[H, W]), dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)
    return image


def classify_image(image):
    if image is not None:
        image = preprocess_image(image)
        preds_probs = tf.squeeze(model.predict(image)).numpy()
        top_3_preds_idx = preds_probs.argsort()[-3:][::-1]
        top_3_preds_conf = np.round(preds_probs[top_3_preds_idx], decimals=2)
        top_3_preds_class = class_names[top_3_preds_idx]

        return {top_3_preds_class[i]: top_3_preds_conf[i] for i in range(3)}

    else:
        return ""


iface = gr.Interface(
    fn=classify_image,
    inputs="image",
    outputs=gr.Label(num_top_classes=3),
    live=True
)

iface.launch()
