import tensorflow as tf
import numpy as np
import gradio as gr

model = tf.keras.models.load_model("model_1.h5")
class_names = np.load("dog_breed_class_names.npy")

H, W = 224, 224

def preprocess_img(image):
    return tf.cast(tf.image.resize(image, size=[H, W]), dtype=tf.float32)

def classify_img(image):
    image = preprocess_img(image)
    expanded_image = tf.expand_dims(image, axis=0)
    model_preds = tf.squeeze(model.predict(expanded_image)).numpy()
    model_top_3_preds_idx = model_preds.argsort()[-3:][::-1]
    model_top_3_preds_class = class_names[model_top_3_preds_idx]
    model_top_3_preds_conf = np.round(model_preds[model_top_3_preds_idx], decimals=2)
    return {model_top_3_preds_class[i]: model_top_3_preds_conf[i] for i in range(3)}

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Dog Image")
        with gr.Column():
            output_lbl = gr.Label(label="Top 3 Results", num_top_classes=3)
            predict_btn = gr.Button("Predict")

    predict_btn.click(fn=classify_img,
                      inputs=input_img,
                      outputs=output_lbl)

iface.launch(share=True, debug=True, server_port=7860)