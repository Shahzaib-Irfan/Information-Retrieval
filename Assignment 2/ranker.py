import os
import streamlit as st
from math import log
from nltk import word_tokenize
from nltk.corpus import stopwords

# Utility function for preprocessing text
def preprocess_text(text):
    tokens = word_tokenize(text)
    
    # Convert to lowercase and remove punctuation
    tokens = [word.lower() for word in tokens if word.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return filtered_tokens

# Function to calculate term frequency (TF)
def calculate_tf(word, document):
    return document.count(word) / len(document)

# Function to calculate inverse document frequency (IDF)
def calculate_idf(word, documents):
    doc_count = sum(doc.count(word) for doc in documents.values())
    if doc_count == 0:
        return 0
    return log(len(documents) / doc_count)

# Function to compute TF-IDF score
def calculate_tf_idf(query, documents):
    query_keywords = preprocess_text(query)
    scores = {}
    for doc_name, words in documents.items():
        tf_idf_score = 0
        for keyword in query_keywords:
            tf = calculate_tf(keyword, words)
            idf = calculate_idf(keyword, documents)
            tf_idf_score += tf * idf
        scores[doc_name] = tf_idf_score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

# Function to load documents from uploaded files
def load_uploaded_files(uploaded_files):
    documents = {}
    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode("utf-8")
        documents[uploaded_file.name] = preprocess_text(content)
    return documents

# Main Streamlit App
def main():
    st.title("Enhanced Document Ranking System")
    
    # Step 1: File Uploads
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload your text documents", accept_multiple_files=True, type="txt")
    
    if uploaded_files:
        documents = load_uploaded_files(uploaded_files)
        st.success(f"Loaded {len(documents)} documents.")

        # Step 2: Input user query
        query = st.text_input("Enter your search query:")
        if query:
            st.subheader("Choose Ranking Method")
            ranking_method = st.radio("Select ranking method", options=["Keyword Matching", "TF-IDF Scoring"])

            if ranking_method == "Keyword Matching":
                rankings = keyword_matching(query, documents)
            else:  # TF-IDF Scoring
                rankings = calculate_tf_idf(query, documents)

            # Step 3: Display rankings
            st.subheader("Ranked Documents:")
            if rankings:
                for rank, (doc_name, score) in enumerate(rankings, start=1):
                    if score == 0:
                        continue
                    snippet = " ".join(documents[doc_name][:50])  # Display first 50 words as snippet
                    st.markdown(f"**Rank {rank}: {doc_name}** (Score: {score:.4f})\n\n*Snippet:* {snippet}...\n")
            else:
                st.warning("No relevant documents found.")
    else:
        st.info("Please upload some text documents to begin.")

# Helper function for keyword matching (unchanged from previous version)
def keyword_matching(query, documents):
    query_keywords = preprocess_text(query)
    rankings = []
    for doc_name, words in documents.items():
        match_count = sum(words.count(keyword) for keyword in query_keywords if keyword in words)
        rankings.append((doc_name, match_count))
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings

if __name__ == "__main__":
    main()
