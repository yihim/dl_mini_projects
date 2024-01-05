import tensorflow as tf
import gradio as gr
from joblib import load
import re

config_weights = load("sentiment140_text_vectorizer.joblib")
text_vectorizer = tf.keras.layers.TextVectorization().from_config(config_weights["config"])
text_vectorizer.adapt(["xyz"])
text_vectorizer.set_weights(config_weights["weights"])

preprocess_text_components = load("preprocess_text.joblib")

def preprocess_text(text, stem=False):
    text = re.sub(preprocess_text_components["text_cleaning_re"], " ", str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in preprocess_text_components["stop_words"]:
            if stem:
                tokens.append(preprocess_text_components["stemmer"].stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

model = tf.keras.models.load_model("model_2.h5")

def classify_text(text):
    print(text)
    cleaned_text = preprocess_text(text)
    print(cleaned_text)
    text_vectorized = text_vectorizer([cleaned_text])
    preds = tf.squeeze(tf.cast(tf.round(model.predict(text_vectorized)), dtype=tf.int64))
    sentiment = "Negative" if preds == 0 else "Positive"
    return sentiment

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            input_text = gr.Text(label="Tweets")
            analyze_btn = gr.Button("Analyze")
        with gr.Column():
            output_label = gr.Label(label="Sentiment")

    analyze_btn.click(fn=classify_text, inputs=input_text, outputs=output_label)
    examples = gr.Examples(examples=["Just got a promotion at work! Feeling grateful and excited for the new opportunities. #Blessed",
                                      "Stuck in traffic again. Can't believe how frustrating the commute is every day. #TrafficWoes"],
                           inputs=input_text)

iface.launch()