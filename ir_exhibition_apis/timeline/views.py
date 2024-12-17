from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import HistoricalPeriod, HistoricalContent
from .serializers import HistoricalPeriodSerializer, HistoricalContentSerializer

class HistoricalPeriodViewSet(viewsets.ModelViewSet):
    queryset = HistoricalPeriod.objects.all()
    serializer_class = HistoricalPeriodSerializer

    @action(detail=False, methods=['GET'])
    def timeline_data(self, request):
        periods = self.get_queryset()
        serializer = self.get_serializer(periods, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['GET'])
    def period_content(self, request, pk=None):
        period = self.get_object()
        contents = HistoricalContent.objects.filter(period=period)
        serializer = HistoricalContentSerializer(contents, many=True)
        return Response(serializer.data)