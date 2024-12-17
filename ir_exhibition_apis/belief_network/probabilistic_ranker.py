import math
from collections import defaultdict
from typing import List, Dict, Tuple

class InformationRetrievalModels:
    def __init__(self, documents: List[Tuple[str, str]] = None):
        """
        Initialize the Information Retrieval Models class.
        
        Args:
            documents (List[Tuple[str, str]], optional): Initial list of documents to process. 
            Each document is a tuple of (title, content). Defaults to None.
        """
        self.documents = documents or []
        
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
    
    def reset_data(self):
        """Reset all data structures to their initial state."""
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
        Precompute advanced term statistics for probabilistic inference.
        """
        term_doc_frequencies = defaultdict(int)
        total_docs = len(self.documents)
        
        for _, content in self.documents:
            terms = set(content.lower().split())
            for term in terms:
                term_doc_frequencies[term] += 1
        
        for term, freq in term_doc_frequencies.items():
            self.prior_probabilities['term_significance'][term] = math.log(total_docs / (freq + 1))
    
    def create_relevance_judgments(self, queries: List[str]) -> Dict[str, Dict[int, float]]:
        """
        Create advanced relevance judgments for given queries.
        """
        self._compute_term_statistics()
        
        self.queries = queries
        self.relevance_judgments = {}
        
        for query in queries:
            query_relevance = {}
            query_terms = set(query.lower().split())
            
            for doc_idx, (title, content) in enumerate(self.documents):
                doc_terms = set(content.lower().split())
                
                overlap_score = sum(
                    self.prior_probabilities['term_significance'][term] 
                    for term in query_terms.intersection(doc_terms)
                )
                
                if overlap_score > 0:
                    relevance_score = min(1, overlap_score / len(query_terms))
                    query_relevance[doc_idx] = relevance_score
                else:
                    query_relevance[doc_idx] = 0
            
            self.relevance_judgments[query] = query_relevance
        
        return self.relevance_judgments
    
    def interference_model(self, query: str) -> List[Tuple[int, float]]:
        """
        Compute document relevance using an Interference Model approach.
        """
        if not self.documents or not query:
            return []
        
        relevance_scores = []
        for doc_idx, (title, content) in enumerate(self.documents):
            base_relevance = self.relevance_judgments.get(query, {}).get(doc_idx, 0)
            
            query_terms = set(query.lower().split())
            doc_terms = set(content.lower().split())
            
            overlap_score = sum(
                self.prior_probabilities['term_significance'][term] 
                for term in query_terms.intersection(doc_terms)
            )
        
            overlap = overlap_score / len(query_terms) if query_terms else 0
            
            relevance_score = base_relevance * (1 + overlap)
            relevance_scores.append((doc_idx, relevance_score))
        
        return sorted(relevance_scores, key=lambda x: x[1], reverse=True)

    def compute_document_relevance(self, query: str, doc_index: int) -> float:
        """
        Compute document relevance using advanced Bayesian probabilistic principles.
        """
        base_relevance = self.relevance_judgments.get(query, {}).get(doc_index, 0.5)
        
        query_terms = set(query.lower().split())
        doc_terms = set(self.documents[doc_index][1].lower().split())
        
        overlap_score = sum(
            self.prior_probabilities['term_significance'][term] 
            for term in query_terms.intersection(doc_terms)
        )
        
        term_overlap_prob = overlap_score / len(query_terms) if query_terms else 0
        
        query_given_relevance = term_overlap_prob
        relevance_prior = base_relevance
        query_prior = self.prior_probabilities['query_importance']
        
        relevance_probability = (
            query_given_relevance * relevance_prior / (query_prior + 1e-10)
        )
        
        return relevance_probability
    
    def belief_network(self, query: str) -> List[Tuple[int, float]]:
        """
        Enhanced Belief Network for document ranking.
        """
        if not self.documents or not query:
            return []
        
        document_relevances = [
            (doc_idx, self.compute_document_relevance(query, doc_idx))
            for doc_idx in range(len(self.documents))
        ]
        
        return sorted(document_relevances, key=lambda x: x[1], reverse=True)