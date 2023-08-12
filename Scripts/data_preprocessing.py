import pandas as pd

# Load the dataset
data = pd.read_csv('../Datasets/Resume.csv')

# Explore the dataset
print(data.head())
print(data['category'].value_counts())

# Preprocessing: Tokenization, removing special characters, and lowercasing text
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

data['preprocessed_resume'] = data['resume_text'].apply(preprocess_text)
