from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split dataset
train_data, test_data, train_labels, test_labels = train_test_split(data['preprocessed_resume'], data['category'], test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# Train Random Forest classifier
model = RandomForestClassifier()
model.fit(train_vectors, train_labels)

# Evaluate
predictions = model.predict(test_vectors)
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)
