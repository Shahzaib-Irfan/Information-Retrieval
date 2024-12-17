from django.db import models
import os
import math
from collections import defaultdict, Counter
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

class Document(models.Model):
    title = models.CharField(max_length=255, unique=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    @classmethod
    def search_by_title(cls, query):
        return cls.objects.filter(title__icontains=query)

    @classmethod
    def get_synonyms(cls, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
        return list(synonyms)

    @classmethod
    def expand_query_with_synonyms(cls, query_tokens):
        expanded_query = set(query_tokens)
        for word in query_tokens:
            expanded_query.update(cls.get_synonyms(word))
        return list(expanded_query)

    @classmethod
    def extract_nouns_and_entities(cls, content):
        words = word_tokenize(content.lower())
        pos_tags = pos_tag(words)
        
        nouns = [word for word, pos in pos_tags if pos in ('NN', 'NNS','NNP', 'NNPS')]
        
        lemmatizer = WordNetLemmatizer()
        lemmatized_nouns = [lemmatizer.lemmatize(noun) for noun in nouns]
        
        return lemmatized_nouns

    @classmethod
    def calculate_tf(cls, doc_words):
        tf = {}
        total_words = len(doc_words)
        word_counts = Counter(doc_words)
        
        for word, count in word_counts.items():
            tf[word] = count / total_words
        return tf

    @classmethod
    def calculate_idf(cls, documents):
        idf = {}
        total_docs = len(documents)
        word_doc_counts = defaultdict(int)
        
        for doc_words in documents:
            unique_words = set(doc_words)
            for word in unique_words:
                word_doc_counts[word] += 1
        
        for word, doc_count in word_doc_counts.items():
            idf[word] = math.log(total_docs / (1 + doc_count))
        return idf

    @classmethod
    def calculate_tfidf(cls, doc_words, tf, idf):
        tfidf = {}
        for word in doc_words:
            if word in idf:
                tfidf[word] = tf[word] * idf[word]
        return tfidf

    @classmethod
    def phrase_match(cls, content, query):
        content_words = word_tokenize(content.lower())
        query_words = word_tokenize(query.lower())
        
        for i in range(len(content_words) - len(query_words) + 1):
            if content_words[i:i+len(query_words)] == query_words:
                return True
        return False

    @classmethod
    def search_by_content(cls, query):
        # Get all documents
        all_documents = cls.objects.all()
        
        matching_docs = set()
        
        # First, try exact phrase matching
        for doc in all_documents:
            if cls.phrase_match(doc.content, query):
                matching_docs.add(doc)
        
        # If no exact matches, try semantic search
        if not matching_docs:
            # Tokenize and tag the query
            query_tokens = word_tokenize(query.lower())
            query_pos_tags = pos_tag(query_tokens)
            
            # Improved noun extraction
            query_nouns = [
                word for word, pos in query_pos_tags 
                if pos in ('NN', 'NNS', 'NNP', 'NNPS')
                and len(word) > 1
                and word not in stopwords.words('english') 
                and word not in ['something', 'anything', 'everything']
            ]
            
            # If no meaningful nouns, try using most significant words
            if not query_nouns:
                query_nouns = [
                    word for word, pos in query_pos_tags 
                    if pos.startswith('N')
                    or pos.startswith('JJ')
                    or word.lower() not in stopwords.words('english')
                ]
            
            # Get expanded tokens for nouns
            expanded_noun_tokens = {}
            for noun in query_nouns:
                synonyms = cls.expand_query_with_synonyms([noun])
                if synonyms:
                    expanded_noun_tokens[noun] = synonyms
            
            # Build an index dynamically for search
            index = defaultdict(list)
            for doc in all_documents:
                doc_words = cls.extract_nouns_and_entities(doc.content)
                
                # Calculate TF-IDF
                doc_word_list = [doc_words]
                idf = cls.calculate_idf(doc_word_list)
                tf = cls.calculate_tf(doc_words)
                tfidf = cls.calculate_tfidf(doc_words, tf, idf)
                
                # Index important terms
                important_terms = {term for term, score in tfidf.items() if score > 0.01}
                for term in important_terms:
                    index[term].append(doc)
            
            # Find matching documents
            noun_matching_docs = None
            if expanded_noun_tokens:
                for noun, synonyms in expanded_noun_tokens.items():
                    current_docs = set()
                    
                    # Add documents matching the noun
                    current_docs.update(index.get(noun, []))
                    
                    # Add documents matching synonyms
                    for synonym in synonyms:
                        current_docs.update(index.get(synonym, []))
                    
                    # Intersect or union documents based on iteration
                    if noun_matching_docs is None:
                        noun_matching_docs = current_docs
                    else:
                        noun_matching_docs.intersection_update(current_docs)
            
            return noun_matching_docs or set()
        
        return matching_docs