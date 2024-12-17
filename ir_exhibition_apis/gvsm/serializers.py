from rest_framework import serializers
from .models import GVSMDocument

class GVSMDocumentSerializer(serializers.ModelSerializer):
    similarity = serializers.FloatField(required=False)
    expanded_query = serializers.CharField(required=False)

    class Meta:
        model = GVSMDocument
        fields = ['id', 'title', 'content', 'uploaded_at', 'similarity', 'expanded_query']