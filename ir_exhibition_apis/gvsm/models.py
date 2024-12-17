from django.db import models
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np

class GVSMDocument(models.Model):
    """Django model to store GVSM documents."""
    title = models.CharField(max_length=255)
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    @classmethod
    def lemmatize_tokens(cls, text):
        """Lemmatize tokens to handle word variations."""
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in text.lower().split()]

    @classmethod
    def get_synonyms(cls, word):
        """Retrieve synonyms for a given word."""
        synonyms = set()
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().lower())
        except LookupError:
            return set()
        return synonyms

    @classmethod
    def expand_query(cls, query):
        """Expand query with lemmatized tokens and synonyms."""
        lemmatizer = WordNetLemmatizer()
        expanded_tokens = []
        for word in query.split():
            base_word = lemmatizer.lemmatize(word)
            expanded_tokens.append(base_word)
            expanded_tokens.extend(list(cls.get_synonyms(base_word)))
        return ' '.join(set(expanded_tokens))

    @classmethod
    def build_vocabulary(cls, documents):
        """Build vocabulary using lemmatized tokens."""
        vocabulary = set()
        for doc in documents:
            vocabulary.update(cls.lemmatize_tokens(doc))
        return sorted(vocabulary)

    @classmethod
    def calculate_tf(cls, term, document_tokens):
        """Calculate Term Frequency."""
        term_count = document_tokens.count(term)
        total_terms = len(document_tokens)
        return term_count / total_terms if total_terms > 0 else 0

    @classmethod
    def calculate_idf(cls, term, all_document_tokens):
        """Calculate Inverse Document Frequency."""
        doc_count = sum(1 for doc_tokens in all_document_tokens if term in doc_tokens)
        total_docs = len(all_document_tokens)
        return math.log(total_docs / (1 + doc_count))

    @classmethod
    def vectorize_with_tfidf(cls, text, vocabulary, idf_values):
        """Convert text to TF-IDF weighted vector."""
        tokens = cls.lemmatize_tokens(text)
        tfidf_vector = [
            cls.calculate_tf(term, tokens) * idf_values.get(term, 0) 
            for term in vocabulary
        ]
        return tfidf_vector

    @classmethod
    def calculate_cosine_similarity(cls, query_vector, document_vectors):
        """Calculate cosine similarity between query and documents."""
        def cosine_similarity(vec1, vec2):
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_vec1 = sum(a ** 2 for a in vec1) ** 0.5
            norm_vec2 = sum(a ** 2 for a in vec2) ** 0.5
            return dot_product / (norm_vec1 * norm_vec2 + 1e-10)
        
        return [cosine_similarity(query_vector, doc_vector) for doc_vector in document_vectors]

    @classmethod
    def semantic_search(cls, query):
        """Perform semantic search on documents."""
        # Retrieve all documents
        documents = [doc.content for doc in cls.objects.all()]
        
        if not documents:
            return []

        # Process documents
        vocabulary = cls.build_vocabulary(documents)
        
        # Tokenize documents with lemmatization
        tokenized_documents = [cls.lemmatize_tokens(doc) for doc in documents]
        
        # Calculate IDF values
        idf_values = {
            term: cls.calculate_idf(term, tokenized_documents) 
            for term in vocabulary
        }
        
        # Vectorize documents with TF-IDF
        document_vectors = [
            cls.vectorize_with_tfidf(doc, vocabulary, idf_values) 
            for doc in documents
        ]

        # Expand query with synonyms and lemmatization
        expanded_query = cls.expand_query(query)
        
        # Vectorize expanded query
        query_vector = cls.vectorize_with_tfidf(
            expanded_query, 
            vocabulary, 
            idf_values
        )
        
        # Calculate similarities
        similarities = cls.calculate_cosine_similarity(
            query_vector, 
            document_vectors
        )
        
        # Rank and return results
        ranked_docs = sorted(
            zip(cls.objects.all(), similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked_docs