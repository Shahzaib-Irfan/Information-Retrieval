from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Product
from .serializers import ProductSerializer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

    @action(detail=False, methods=['GET'])
    def filter_products(self, request):
        category = request.query_params.get('category')
        min_price = request.query_params.get('min_price')
        max_price = request.query_params.get('max_price')
        search_query = request.query_params.get('search')

        queryset = self.get_queryset()

        if category:
            queryset = queryset.filter(category=category)
        
        if min_price:
            queryset = queryset.filter(price__gte=min_price)
        
        if max_price:
            queryset = queryset.filter(price__lte=max_price)
        
        if search_query:
            queryset = queryset.filter(
                title__icontains=search_query
            )

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)