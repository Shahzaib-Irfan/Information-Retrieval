from rest_framework import serializers
from .models import BeliefNetworkDocument

class BeliefNetworkDocumentSerializer(serializers.ModelSerializer):
    """
    Serializer for BeliefNetwork Documents with additional fields for ranking
    """
    relevance_score = serializers.FloatField(required=False)
    rank = serializers.IntegerField(required=False)
    
    class Meta:
        model = BeliefNetworkDocument
        fields = [
            'id', 
            'title', 
            'content', 
            'uploaded_at', 
            'relevance_score', 
            'rank'
        ]