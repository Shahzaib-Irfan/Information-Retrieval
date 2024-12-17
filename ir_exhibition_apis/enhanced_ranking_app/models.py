import numpy as np
import re
import pandas as pd
import io
from PyPDF2 import PdfReader
from django.db import models
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import logging

class DocumentLoader:
    """Handles document loading from various file formats."""
    
    @staticmethod
    def load_text_file(file_content: str) -> List[str]:
        """Load content from a text file."""
        try:
            return [line.strip() for line in file_content.split('\n') if line.strip()]
        except Exception as e:
            logging.error(f"Error loading text file: {str(e)}")
            raise

    @staticmethod
    def load_csv_file(file_content: str) -> List[str]:
        """Load content from a CSV file."""
        try:
            df = pd.read_csv(io.StringIO(file_content))
            if df.empty:
                raise ValueError("CSV file is empty")
            
            # Use first column by default
            text_column = df.columns[0]
            return df[text_column].dropna().tolist()
        except Exception as e:
            logging.error(f"Error loading CSV file: {str(e)}")
            raise

    @staticmethod
    def load_pdf_file(file_content: bytes) -> List[str]:
        """Load content from a PDF file."""
        try:
            pdf_reader = PdfReader(io.BytesIO(file_content))
            documents = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                documents.extend(paragraphs)
            return documents
        except Exception as e:
            logging.error(f"Error loading PDF file: {str(e)}")
            raise

class TextProcessor:
    """Handles text preprocessing and analysis."""
    
    @staticmethod
    def preprocess_text(text: str) -> List[str]:
        """Clean and tokenize text."""
        words = text.lower().split()
        words = [re.sub(r'[^a-zA-Z\s]', '', word) for word in words]
        return [word for word in words if word]

    @staticmethod
    def create_vocabulary(documents: List[str], stop_words: Set[str] = None) -> List[str]:
        """Create vocabulary from documents excluding stop words."""
        if stop_words is None:
            stop_words = set()
        vocabulary = set()
        for doc in documents:
            words = TextProcessor.preprocess_text(doc)
            vocabulary.update([word for word in words if word not in stop_words])
        return sorted(list(vocabulary))

    @staticmethod
    def create_binary_vector(text: str, vocabulary: List[str]) -> np.ndarray:
        """Create a binary vector for text based on vocabulary."""
        words = set(TextProcessor.preprocess_text(text))
        return np.array([1 if term in words else 0 for term in vocabulary])

    @staticmethod
    def calculate_jaccard_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Jaccard similarity between two binary vectors."""
        intersection = np.sum(vec1 & vec2)
        union = np.sum(vec1 | vec2)
        return intersection / union if union != 0 else 0

class AdvancedDocument(models.Model):
    """Django model to store documents for advanced search."""
    title = models.CharField(max_length=255)
    content = models.TextField()
    file_type = models.CharField(max_length=50)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    @classmethod
    def load_document(cls, file, file_type):
        """
        Load document based on file type and create model instance
        """
        try:
            if file_type == 'text/plain':
                content = file.read().decode('utf-8')
                documents = DocumentLoader.load_text_file(content)
            elif file_type == 'text/csv':
                content = file.read().decode('utf-8')
                documents = DocumentLoader.load_csv_file(content)
            elif file_type == 'application/pdf':
                content = file.read()
                documents = DocumentLoader.load_pdf_file(content)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # Create document instances for each loaded document
            doc_instances = []
            for doc_content in documents:
                doc_instance = cls.objects.create(
                    title=file.name,
                    content=doc_content,
                    file_type=file_type
                )
                doc_instances.append(doc_instance)

            return doc_instances
        except Exception as e:
            logging.error(f"Error loading document: {str(e)}")
            raise

    @classmethod
    def search_binary_term_matching(cls, query: str):
        """Perform binary term matching search."""
        documents = cls.objects.all()
        if not documents:
            return []

        # Create vocabulary and vectors
        documents_content = [doc.content for doc in documents]
        vocabulary = TextProcessor.create_vocabulary(documents_content + [query])
        
        doc_vectors = np.array([
            TextProcessor.create_binary_vector(doc.content, vocabulary)
            for doc in documents
        ])
        query_vector = TextProcessor.create_binary_vector(query, vocabulary)

        # Calculate similarities
        similarities = [
            TextProcessor.calculate_jaccard_similarity(doc_vec, query_vector)
            for doc_vec in doc_vectors
        ]

        # Return sorted results with similarity and document
        results = [
            {'document': doc, 'similarity': sim} 
            for doc, sim in zip(documents, similarities)
        ]
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

    @classmethod
    def search_non_overlapping_lists(cls, terms: List[str]):
        """Perform non-overlapping lists search."""
        results = {}
        for term in terms:
            term_lower = term.lower()
            matching_docs = cls.objects.filter(content__icontains=term_lower)
            if matching_docs:
                results[term] = matching_docs
        return results

    @classmethod
    def search_proximal_node(cls, entities: List[str], window_size: int = 50):
        """Perform proximal node search with configurable window size."""
        results = defaultdict(list)
        documents = cls.objects.all()

        for doc in documents:
            doc_lower = doc.content.lower()
            doc_length = len(doc_lower)
            
            # Find all occurrences of each entity
            entity_positions = {}
            for entity in entities:
                entity_lower = entity.lower()
                positions = []
                start = 0
                while True:
                    pos = doc_lower.find(entity_lower, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                if positions:
                    entity_positions[entity] = positions

            # Check for proximity between entities
            if len(entity_positions) >= 2:
                for entity1, pos1_list in entity_positions.items():
                    for entity2, pos2_list in entity_positions.items():
                        if entity1 >= entity2:
                            continue
                            
                        for pos1 in pos1_list:
                            for pos2 in pos2_list:
                                if abs(pos1 - pos2) <= window_size:
                                    start_pos = max(0, min(pos1, pos2) - 20)
                                    end_pos = min(doc_length, max(pos1, pos2) + 20)
                                    context = doc.content[start_pos:end_pos]
                                    
                                    result_entry = {
                                        'document': doc,
                                        'context': context,
                                        'entities_found': [entity1, entity2],
                                        'distance': abs(pos1 - pos2)
                                    }
                                    
                                    key = f"{entity1}-{entity2}"
                                    results[key].append(result_entry)

        return dict(results)