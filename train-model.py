from datasets import load_dataset
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('stopwords')

# Load actual dataset from Hugging Face :cite[4]
dataset = load_dataset("dipesh/Intent-Classification-small")
df = pd.DataFrame(dataset['train'])

# Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['processed'] = df['text'].apply(preprocess)

# Train model
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=5000),
    MultinomialNB()
)

model.fit(df['processed'], df['intent'])

# Save model
joblib.dump(model, 'email_intent_model.joblib')

print("Model trained with dataset containing intents:")
print(df['intent'].value_counts())
