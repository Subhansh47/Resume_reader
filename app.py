# Import necessary libraries
import streamlit as st
from PyPDF2 import PdfReader
import re
import json
import pickle

# Load pre-trained model and TF-IDF vectorizer
Model = pickle.load(open('Model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Load category mapping from JSON file
with open('category_mapping.json', 'r') as json_file:
    category_mapping = json.load(json_file)

# Function for text preprocessing
def preprocessing(text):
    text = re.sub('http\S+\s', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

# Function to predict category
def predict_category(x):
    x = preprocessing(x)
    x = tfidf.transform([x])
    pred = int(Model.predict(x))
    return category_mapping[f'{pred}']

# Streamlit app
def main():
    st.title("Your Streamlit App Title")

    # File Upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Read PDF file
        pdf_reader = PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)

        # st.write(f"Number of pages in the PDF: {num_pages}")

        # Extract text from the first page
        page = pdf_reader.pages[0]
        text = page.extract_text()

        # st.write("Text from the first page:")
        # st.write(text)

        # Predict category using your model
        category_prediction = predict_category(text)
        st.write(f"Predicted category: {category_prediction}")

if __name__ == "__main__":
    main()
