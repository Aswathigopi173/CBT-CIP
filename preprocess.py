import pandas as pd
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('questions_answers.csv', header=None)

# Check the number of columns and display first few rows
print(f"Number of columns: {df.shape[1]}")
print("Columns in DataFrame:", df.columns)
print("First 5 Rows:", df.head())

# Keep only the first two columns: 'questions' and 'answers'
df = df.iloc[:, :2]  # Select the first two columns

# Rename columns
df.columns = ['questions', 'answers']

# Preprocessing steps (strip any leading/trailing spaces)
df['questions'] = df['questions'].str.strip()
df['answers'] = df['answers'].str.strip()

# Save cleaned data to a new file
df.to_csv('cleaned_questions_answers.csv', index=False)
print("Preprocessing completed and saved to: cleaned_questions_answers.csv")
