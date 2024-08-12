#  Market Reader: Sentiment Analysis on Financial News Using BERT

## Overview

This project provides a Flask API that leverages the BERT language model to perform sentiment analysis. Additionally, it features a web scraping component that collects headlines from Yahoo Finance and analyzes their sentiment. If interested in seeing a live website that does the features below, visit the model on HuggingFace [here](https://huggingface.co/spaces/arminap/sentiment-analysis). The deployed model on HuggingFace utilizes Gradio to deploy a front end application with our trained model.

## Features

- **Sentiment Analysis**
  - Utilizes the BERT model to analyze the sentiment of customized input text.
  - Supports three sentiment categories: neutral, negative, and positive.
  - Returns the sentiment of the input text to the user.

- **Yahoo Finance Web Scraping**
  - Scrapes headlines from Yahoo Finance using BeautifulSoup.
  - Analyzes the sentiment of each headline using the BERT model.
  - Returns a list of headlines with their corresponding sentiment in JSON format.
 
## Document Overview
- sentiment-analysis.py: the sentiment analysis model, including training and evaluation
- app.py: the main application file, responsible for running the sentiment analysis back end
- index.html: front end of the application
- dataset-cleaning.py: code for cleaning and preparing the pre-existing darasets
- plot.py: plotting the visualization of our model's results
- requirements.txt: project dependenices
- training_metrics.pkl: training results for loss and accuracy
- /datasets: raw and combined datasets
- /plots: configured visualization of our model's results
- /saved_model: config file for the model **(Note: the model is saved on HuggingFace due to GitHub size restrictions)**
- /saved_tokenizer: tokenizer files for our model
  
## Getting Started

### Prerequisites

Ensure you have the following installed to run the front end application
- Python 3.7+
- numpy < 2
- Flask
- torch
- transformers
- beautifulsoup4
- requests

### Installation
#### Only follow these steps if you wish to run the front end application locally.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-bert.git
   ```

2. Navigate to the project directory:
   ```bash
   cd sentiment-analysis-bert
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask API:
   ```bash
   python3 app.py
   ```

5. Open the index.html file using a live server. You can do this by opening the project repository in VSCode, right clicking on 'index.html' and clicking 'Open with Live Server'. 

## Model and Tokenizer

The BERT model and tokenizer are saved in the `saved_model` and `saved_tokenizer` directories, respectively. You can update these files by retraining the model and tokenizer with your own dataset.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request.

## Acknowledgments

This project uses the following libraries and resources:
- [Hugging Face Transformers](https://huggingface.co/transformers) for the BERT model and tokenizer
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for web scraping
- [Yahoo Finance](https://finance.yahoo.com/) for providing the headlines data

## Resources

The sentiment analysis models were trained using and combining the following datasets:

- [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/data)
- [Sentiment Analysis Labelled Financial News Data](https://www.kaggle.com/datasets/aravsood7/sentiment-analysis-labelled-financial-news-data/data)
- [Financial Sentiment Analysis](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis/data)
