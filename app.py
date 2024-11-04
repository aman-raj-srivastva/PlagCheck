import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load the Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
data = pd.read_csv('dataset.csv')
original_texts = data[data['label'] == 0]['text'].tolist()  # Get original texts

# Function to compute semantic similarity and return plagiarism percentage
def detect(input_text):
    # Embed input text and original texts
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    original_embeddings = model.encode(original_texts, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(input_embedding, original_embeddings)[0]
    
    # Get the maximum similarity score
    max_score = torch.max(cosine_scores).item()
    
    # Calculate plagiarism percentage
    plagiarism_percentage = max_score * 100  # Convert similarity to percentage
    
    # Determine if the input text is plagiarized based on the adjusted threshold
    threshold = 0.7  # Adjust this value based on experiments
    if max_score > threshold:
        return f"Plagiarism Detected: {plagiarism_percentage:.2f}%"
    else:
        return f"No Plagiarism Detected. Similarity: {plagiarism_percentage:.2f}%"

# Streamlit UI
st.title("PlagCheck🔍: Plagiarism Detection App Using Sentence Transformers")
st.write("Enter the text you want to check for plagiarism:")

input_text = st.text_area("Input Text 👇", height=200)

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text to check.")
    else:
        detection_result = detect(input_text)
        st.write(f"Result: {detection_result}")
