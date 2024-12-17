from rest_framework import serializers
from .models import DocumentRanker

class DocumentRankerSerializer(serializers.ModelSerializer):
    preprocessed_snippet = serializers.SerializerMethodField()
    ranking_score = serializers.FloatField(required=False)

    class Meta:
        model = DocumentRanker
        fields = ['id', 'title', 'content', 'preprocessed_snippet', 'ranking_score', 'created_at']

    def get_preprocessed_snippet(self, obj):
        # Return first 50 words of preprocessed content
        words = obj.preprocessed_content.split()
        return ' '.join(words[:50]) + '...'