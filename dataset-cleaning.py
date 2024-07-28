import pandas as pd

column_names1 = ['sentiment', 'text']
dataset1 = pd.read_csv('./datasets/finance_dataset1.csv', names=column_names1, encoding='latin1')
dataset2 = pd.read_csv('./datasets/finance_dataset2.csv', encoding='latin1')
dataset3 = pd.read_csv('./datasets/finance_dataset3.csv', encoding='latin1')

# clean up the data to have the same columns and values before combining
dataset2 = dataset2.rename(columns={'Full_text': 'text', 'Final Status': 'sentiment'})[['text', 'sentiment']]
dataset3 = dataset3.rename(columns={'Sentence': 'text', 'Sentiment': 'sentiment'})[['text', 'sentiment']]

# dataset2 has capitalized values in label and has an extra space in some values
dataset2['sentiment'] = dataset2['sentiment'].str.lower().str.strip()

# combine the datasets into one
dataset = pd.concat([dataset1[['text', 'sentiment']], dataset2, dataset3], ignore_index=True)
dataset.to_csv('./datasets/combined_dataset.csv', index=False)