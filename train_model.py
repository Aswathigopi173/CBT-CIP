import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pickle

nltk.download('punkt')
nltk.download('wordnet')

# Load the cleaned dataset
df = pd.read_csv("cleaned_questions_answers.csv", header=0)

# Extract questions and answers
questions = df['questions'].values
answers = df['answers'].values

# Preprocessing: Tokenization and Lowercasing
questions = [q.lower() for q in questions]

# Encode labels (answers)
label_encoder = LabelEncoder()
encoded_answers = label_encoder.fit_transform(answers)
num_classes = len(label_encoder.classes_)

# Save the label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Vectorize the text data (Bag of Words)
vectorizer = CountVectorizer(tokenizer=word_tokenize, stop_words='english', max_features=5000)
X = vectorizer.fit_transform(questions).toarray()

# Save the vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Convert labels to one-hot encoding
y = tf.keras.utils.to_categorical(encoded_answers, num_classes=num_classes)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=8)

# Save the trained model
model.save("chatbot_model.h5")

print("Model, vectorizer, and label encoder saved successfully!")
