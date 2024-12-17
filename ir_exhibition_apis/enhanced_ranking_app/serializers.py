from rest_framework import serializers
from .models import AdvancedDocument

class AdvancedDocumentSerializer(serializers.ModelSerializer):
    similarity = serializers.FloatField(required=False)
    context = serializers.CharField(required=False)
    entities_found = serializers.ListField(child=serializers.CharField(), required=False)
    distance = serializers.IntegerField(required=False)

    class Meta:
        model = AdvancedDocument
        fields = ['id', 'title', 'content', 'file_type', 'uploaded_at', 
                  'similarity', 'context', 'entities_found', 'distance']