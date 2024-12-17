from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from .models import GVSMDocument
from .serializers import GVSMDocumentSerializer

class GVSMDocumentViewSet(viewsets.ModelViewSet):
    queryset = GVSMDocument.objects.all()
    serializer_class = GVSMDocumentSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def create(self, request, *args, **kwargs):
        """
        Handle multiple file uploads for text files
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
                content = file.read().decode('utf-8')
                
                # Create document instance
                doc_instance = GVSMDocument.objects.create(
                    title=file.name,
                    content=content
                )
                uploaded_docs.append(doc_instance)
            
            except Exception as e:
                return Response(
                    {"error": f"Error processing file {file.name}: {str(e)}"}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        # Serialize and return uploaded documents
        serializer = self.get_serializer(uploaded_docs, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['POST'])
    def semantic_search(self, request):
        """
        Perform semantic search on documents
        """
        query = request.data.get('query')
        
        if not query:
            return Response(
                {"error": "No search query provided"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Perform semantic search
            results = GVSMDocument.semantic_search(query)
            
            # Serialize results
            serialized_results = []
            for doc, similarity in results:
                serializer = self.get_serializer(doc)
                result_data = serializer.data
                result_data['similarity'] = similarity
                result_data['expanded_query'] = GVSMDocument.expand_query(query)
                serialized_results.append(result_data)
            
            return Response(serialized_results)
        
        except Exception as e:
            return Response(
                {"error": f"Search failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )