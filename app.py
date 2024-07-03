import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import gradio as gr
from agent.model import get_model_and_tokenizer

# Load pre-trained model and tokenizer
model, tokenizer = get_model_and_tokenizer()

# Create a pipeline for sentiment analysis
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define a function to make predictions
def predict_sentiment(text):
    results = nlp(text)
    return results[0]

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
)

if __name__ == "__main__":
    iface.launch()
