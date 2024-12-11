import streamlit as st
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import os
import math

class InformationRetrievalModels:
    def __init__(self, documents: List[str] = None):
        self.documents = documents or []
        self.queries = []
        self.relevance_judgments = {}

        self.network_structure = {
            'query': None,
            'document_features': {},
            'document_relevance': {}
        }
        
        # Probability distributions
        self.prior_probabilities = {
            'query_importance': 0.5,
            'term_significance': defaultdict(float)
        }
    
    def reset_data(self):
        """Reset all data and network structure"""
        self.documents = []
        self.queries = []
        self.relevance_judgments = {}
        self.network_structure = {
            'query': None,
            'document_features': {},
            'document_relevance': {}
        }
        self.prior_probabilities = {
            'query_importance': 0.5,
            'term_significance': defaultdict(float)
        }

    def _compute_term_statistics(self):
        """
        Precompute advanced term statistics for probabilistic inference
        """
        term_doc_frequencies = defaultdict(int)
        total_docs = len(self.documents)
        
        # Compute term frequencies across documents
        for _, content in self.documents:
            terms = set(content.lower().split())
            for term in terms:
                term_doc_frequencies[term] += 1
        
        # Compute term significance using inverse document frequency
        for term, freq in term_doc_frequencies.items():
            self.prior_probabilities['term_significance'][term] = math.log(total_docs / (freq + 1))
    
    def create_relevance_judgments(self, queries: List[str]) -> Dict[str, Dict[int, float]]:
        """
        Enhanced relevance judgment creation with more sophisticated scoring
        """
        # Precompute term statistics
        self._compute_term_statistics()
        
        self.queries = queries
        self.relevance_judgments = {}
        
        for query in queries:
            query_relevance = {}
            query_terms = set(query.lower().split())
            
            for doc_idx, (title, content) in enumerate(self.documents):
                doc_terms = set(content.lower().split())
                
                # Advanced term overlap computation
                overlap_score = sum(
                    self.prior_probabilities['term_significance'][term] 
                    for term in query_terms.intersection(doc_terms)
                )
                
                # Normalized and weighted relevance
                if overlap_score > 0:
                    # Consider term significance and normalized overlap
                    relevance_score = min(1, overlap_score / len(query_terms))
                    query_relevance[doc_idx] = relevance_score
                else:
                    query_relevance[doc_idx] = 0
            
            self.relevance_judgments[query] = query_relevance
        
        return self.relevance_judgments
    
    def interference_model(self, query: str) -> List[Tuple[int, float]]:
        """
        Compute document relevance using Interference Model
        
        Args:
            query (str): Query to assess
        
        Returns:
            Sorted list of (document_index, relevance_score)
        """
        if not self.documents or not query:
            return []
        
        # Compute relevance for each document
        relevance_scores = []
        for doc_idx, (title, content) in enumerate(self.documents):
            # Get pre-computed relevance judgment
            base_relevance = self.relevance_judgments.get(query, {}).get(doc_idx, 0)
            
            # Compute term frequency
            query_terms = set(query.lower().split())
            doc_terms = set(content.lower().split())
            
            # Term overlap score
            overlap_score = sum(
                self.prior_probabilities['term_significance'][term] 
                for term in query_terms.intersection(doc_terms)
            )
        
            # Normalized overlap probability
            overlap = overlap_score / len(query_terms) if query_terms else 0
            
            # Combined score
            relevance_score = base_relevance * (1 + overlap)
            relevance_scores.append((doc_idx, relevance_score))
        
        # Sort by relevance score
        return sorted(relevance_scores, key=lambda x: x[1], reverse=True)

    def compute_document_relevance(self, query: str, doc_index: int) -> float:
        """
        Advanced document relevance computation using Bayesian principles
        """
        # Prior relevance from existing judgments
        base_relevance = self.relevance_judgments.get(query, {}).get(doc_index, 0.5)
        
        # Compute advanced term overlap
        query_terms = set(query.lower().split())
        doc_terms = set(self.documents[doc_index][1].lower().split())
        
        # Weighted term overlap considering term significance
        overlap_score = sum(
            self.prior_probabilities['term_significance'][term] 
            for term in query_terms.intersection(doc_terms)
        )
        
        # Normalized overlap probability
        term_overlap_prob = overlap_score / len(query_terms) if query_terms else 0
        
        # Bayesian probability computation
        # P(Relevance | Query) = P(Query | Relevance) * P(Relevance) / P(Query)
        query_given_relevance = term_overlap_prob
        relevance_prior = base_relevance
        query_prior = self.prior_probabilities['query_importance']
        
        # Compute conditional probability with smoothing
        relevance_probability = (
            query_given_relevance * relevance_prior / (query_prior + 1e-10)
        )
        
        return relevance_probability
    
    def belief_network(self, query: str) -> List[Tuple[int, float]]:
        """
        Enhanced Belief Network document ranking
        """
        if not self.documents or not query:
            return []
        
        # Compute relevance for each document using advanced method
        document_relevances = [
            (doc_idx, self.compute_document_relevance(query, doc_idx))
            for doc_idx in range(len(self.documents))
        ]
        
        # Sort documents by relevance score
        return sorted(document_relevances, key=lambda x: x[1], reverse=True)

def main():
    st.title("Information Retrieval Probabilistic Models")
    
    # Initialize or get session state
    if 'ir_models' not in st.session_state:
        st.session_state.ir_models = InformationRetrievalModels()
    
    # Sidebar for document upload
    st.sidebar.header("Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Text Files", 
        type=['txt'], 
        accept_multiple_files=True
    )
    
    # Document processing
    if uploaded_files:
        # Clear previous documents if new files are uploaded
        st.session_state.ir_models.reset_data()
        
        # Read and store documents
        documents = []
        for file in uploaded_files:
            # Read file content
            title = file.name
            content = file.getvalue().decode('utf-8')
            documents.append((title, content))
        
        # Store documents
        st.session_state.ir_models.documents = documents
        
        # Display uploaded documents
        st.sidebar.success(f"Uploaded {len(documents)} documents")
        
        # Optional: Display document previews
        with st.sidebar.expander("Document Previews"):
            for i, doc in enumerate(documents, 1):
                st.text(f"Document {i} (First 100 chars):\n{doc[:100]}...")
    
    # Query input and model selection
    st.header("Relevance Analysis")
    query = st.text_input("Enter your query:")
    model_choice = st.selectbox(
        "Select Probabilistic Model", 
        ["Interference Model", "Belief Network"]
    )
    
    # Compute relevance judgments if documents exist
    if st.session_state.ir_models.documents:
        # Auto-generate initial relevance judgments
        st.session_state.ir_models.create_relevance_judgments([query])
    
    # Perform retrieval when query is entered
    if query and st.session_state.ir_models.documents:
        # Choose model
        if model_choice == "Interference Model":
            results = st.session_state.ir_models.interference_model(query)
        else:
            results = st.session_state.ir_models.belief_network(query)
        
        # Display results
        st.subheader("Ranked Documents")
        for rank, (doc_idx, score) in enumerate(results, 1):
            if score > 0:
                st.markdown(f"**Rank {rank}** (Score: {score:.4f}):")
                st.markdown(f"**FileName**: {st.session_state.ir_models.documents[doc_idx][0]}")
                st.text(st.session_state.ir_models.documents[doc_idx][1][:300] + "...")

if __name__ == "__main__":
    main()