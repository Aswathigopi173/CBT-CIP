import tensorflow as tf
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Load the trained model, vectorizer, and label encoder
model = tf.keras.models.load_model("chatbot_model.h5")

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Chatbot inference
def chatbot_response(user_input):
    # Preprocess the user input
    user_input = user_input.lower()
    user_vector = vectorizer.transform([user_input]).toarray()
    
    # Predict the class
    predictions = model.predict(user_vector)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    # Decode the predicted class
    response = label_encoder.inverse_transform([predicted_class])[0]
    return response, confidence

# Start the chatbot
print("\nHello! I am your chatbot.\nIf you have any questions regarding programming basics, you can ask me!\nType 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    response, confidence = chatbot_response(user_input)
    print(f"Chatbot: {response} (Confidence: {confidence:.2f})")
