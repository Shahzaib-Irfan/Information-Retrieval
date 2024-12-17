from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from .models import DocumentRanker
from .serializers import DocumentRankerSerializer

class DocumentRankerViewSet(viewsets.ModelViewSet):
    queryset = DocumentRanker.objects.all()
    serializer_class = DocumentRankerSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def create(self, request, *args, **kwargs):
        """
        Handle file uploads for multiple documents
        """
        files = request.FILES.getlist('files')
        
        if not files:
            return Response(
                {"error": "No files uploaded"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_docs = []
        for file in files:
            try:
                # Read file content
                content = file.read().decode('utf-8').strip()
                
                # Create DocumentRanker instance
                doc = DocumentRanker.objects.create(
                    title=file.name,
                    content=content
                )
                uploaded_docs.append(doc)
            
            except Exception as e:
                return Response(
                    {"error": f"Error processing file {file.name}: {str(e)}"}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        # Serialize and return uploaded documents
        serializer = self.get_serializer(uploaded_docs, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['POST'], parser_classes=[MultiPartParser, FormParser, JSONParser])
    def keyword_matching(self, request):
        """
        Perform keyword matching search
        """
        query = request.data.get('query')
        
        if not query:
            return Response(
                {"error": "No search query provided"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Perform keyword matching
            rankings = DocumentRanker.keyword_matching(query)
            
            # Prepare response with serialized documents and scores
            results = []
            for doc, score in rankings:
                serialized_doc = self.get_serializer(doc).data
                serialized_doc['ranking_score'] = score
                results.append(serialized_doc)
            
            return Response(results)
        
        except Exception as e:
            return Response(
                {"error": f"Search failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['POST'], parser_classes=[MultiPartParser, FormParser, JSONParser])
    def tf_idf_ranking(self, request):
        """
        Perform TF-IDF ranking search
        """
        query = request.data.get('query')
        
        if not query:
            return Response(
                {"error": "No search query provided"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Perform TF-IDF ranking
            rankings = DocumentRanker.calculate_tf_idf(query)
            
            # Prepare response with serialized documents and scores
            results = []
            for doc, score in rankings:
                serialized_doc = self.get_serializer(doc).data
                serialized_doc['ranking_score'] = score
                results.append(serialized_doc)
            
            return Response(results)
        
        except Exception as e:
            return Response(
                {"error": f"Search failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )