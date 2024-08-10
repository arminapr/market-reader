import gradio as gr
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from bs4 import BeautifulSoup
import requests

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model_save_path = "./saved_model2"
tokenizer_save_path = "./saved_tokenizer2"

model = BertForSequenceClassification.from_pretrained(model_save_path).to(device)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_save_path)

sentiment_mapping = {0: 'neutral', 1: 'negative', 2: 'positive'}

# function to predict the sentiment of the text
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return sentiment_mapping[predictions[0]]

# function to scrape headlines and predict their sentiment
def scrape_and_predict():
    url = "https://finance.yahoo.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    headlines = []
    h3_tags = soup.find_all('h3', class_='clamp yf-1044anq')
    for h3 in h3_tags:
        headline = h3.get_text(strip=True)
        if headline:
            sentiment = predict_sentiment(headline)
            headlines.append({'headline': headline, 'sentiment': sentiment})

    return headlines

def sentiment_interface(text):
    return predict_sentiment(text)

def scrape_interface():
    return scrape_and_predict()

# gradio app
with gr.Blocks() as demo:
    gr.Markdown("# Sentiment Analysis and News Scraping")
    
    with gr.Tab("Predict Sentiment"):
        text_input = gr.Textbox(label="Input Text")
        sentiment_output = gr.Textbox(label="Sentiment")
        predict_button = gr.Button("Predict Sentiment")
        predict_button.click(sentiment_interface, inputs=text_input, outputs=sentiment_output)

    with gr.Tab("Scrape Yahoo Finance"):
        scrape_button = gr.Button("Scrape and Predict Sentiment")
        headlines_output = gr.JSON(label="Headlines and Sentiment")
        scrape_button.click(scrape_interface, outputs=headlines_output)

if __name__ == "__main__":
    demo.launch(share=True)