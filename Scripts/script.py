import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def categorize_resumes(input_dir, model, vectorizer):
    categorized_data = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), 'r') as file:
                resume_text = file.read()
            preprocessed_text = preprocess_text(resume_text)
            vectorized_text = vectorizer.transform([preprocessed_text])
            category = model.predict(vectorized_text)[0]
            
            category_dir = os.path.join(input_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
            
            os.rename(os.path.join(input_dir, filename), os.path.join(category_dir, filename))
            categorized_data.append({'filename': filename, 'category': category})

    categorized_df = pd.DataFrame(categorized_data)
    categorized_df.to_csv('categorized_resumes.csv', index=False)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    model = RandomForestClassifier()  # Load your trained model here
    vectorizer = TfidfVectorizer(max_features=1000)  # Load your vectorizer here
    
    categorize_resumes(input_dir, model, vectorizer)
