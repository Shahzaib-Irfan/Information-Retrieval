from django.db import models
import os
from math import log
from nltk import word_tokenize
from nltk.corpus import stopwords

class DocumentRanker(models.Model):
    title = models.CharField(max_length=255, unique=True)
    content = models.TextField()
    preprocessed_content = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def preprocess_text(self):
        tokens = word_tokenize(self.content)
        
        # Convert to lowercase and remove punctuation
        tokens = [word.lower() for word in tokens if word.isalnum()]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        # Store preprocessed content as space-separated string
        self.preprocessed_content = ' '.join(filtered_tokens)
        return filtered_tokens

    def save(self, *args, **kwargs):
        # Preprocess content before saving
        self.preprocess_text()
        super().save(*args, **kwargs)

    @classmethod
    def calculate_tf(cls, word, document):
        return document.count(word) / len(document.split())

    @classmethod
    def calculate_idf(cls, word):
        # Count documents containing the word
        doc_count = cls.objects.filter(preprocessed_content__contains=word).count()
        total_docs = cls.objects.count()
        
        if doc_count == 0:
            return 0
        return log(total_docs / doc_count)

    @classmethod
    def keyword_matching(cls, query):
        query_keywords = cls._preprocess_query(query)
        
        # Rank documents based on keyword matches
        rankings = []
        for doc in cls.objects.all():
            doc_words = doc.preprocessed_content.split()
            match_count = sum(doc_words.count(keyword) for keyword in query_keywords if keyword in doc_words)
            rankings.append((doc, match_count))
        
        # Sort by match count in descending order
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    @classmethod
    def calculate_tf_idf(cls, query):
        query_keywords = cls._preprocess_query(query)
        
        # Calculate TF-IDF scores
        scores = {}
        for doc in cls.objects.all():
            doc_words = doc.preprocessed_content.split()
            tf_idf_score = 0
            for keyword in query_keywords:
                # Only calculate for keywords present in document
                if keyword in doc_words:
                    tf = cls.calculate_tf(keyword, doc.preprocessed_content)
                    idf = cls.calculate_idf(keyword)
                    tf_idf_score += tf * idf
            
            # Only add if score is non-zero
            if tf_idf_score > 0:
                scores[doc] = tf_idf_score
        
        # Sort scores in descending order
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores

    @classmethod
    def _preprocess_query(cls, query):
        tokens = word_tokenize(query)
        
        # Convert to lowercase and remove punctuation
        tokens = [word.lower() for word in tokens if word.isalnum()]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        return filtered_tokens

    def __str__(self):
        return self.title