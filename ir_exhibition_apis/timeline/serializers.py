from rest_framework import serializers
from .models import HistoricalPeriod, HistoricalContent

class HistoricalPeriodSerializer(serializers.ModelSerializer):
    class Meta:
        model = HistoricalPeriod
        fields = '__all__'

class HistoricalContentSerializer(serializers.ModelSerializer):
    period_name = serializers.CharField(source='period.name', read_only=True)

    class Meta:
        model = HistoricalContent
        fields = ['id', 'period_name', 'content']