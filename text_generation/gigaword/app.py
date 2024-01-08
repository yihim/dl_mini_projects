import tensorflow as tf
import gradio as gr
from transformers import AutoTokenizer, TFT5ForConditionalGeneration

model = TFT5ForConditionalGeneration.from_pretrained("Yihim/flan_t5_small_ggw_v1")
tokenizer = AutoTokenizer.from_pretrained("Yihim/flan_t5_small_ggw_v1")

MAX_LENGTH = 48

def summarize_statement(statement):
    inputs = tokenizer(statement, return_tensors="tf", max_length=MAX_LENGTH, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=MAX_LENGTH, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            input_statement = gr.Textbox(label="Statement", placeholder="Statement for summarization")
            summarize_btn = gr.Button("Summarize")
            examples = gr.Examples(examples=[
                "Advancements in artificial intelligence are reshaping industries, with applications ranging from healthcare to finance. The integration of AI technologies is transforming how businesses operate and make decisions, leading to increased efficiency and innovation.",
                "Climate change poses a critical threat to our planet, affecting ecosystems, weather patterns, and sea levels. Urgent action is needed to mitigate its impact, emphasizing sustainable practices, renewable energy adoption, and global cooperation."],
                                   inputs=[input_statement])
        with gr.Column():
            output_summary = gr.Text(label="Summary")
    summarize_btn.click(fn=summarize_statement, inputs=input_statement, outputs=output_summary)


iface.launch()
