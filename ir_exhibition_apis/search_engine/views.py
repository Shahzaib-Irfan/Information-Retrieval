from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from .models import Document
from .serializers import DocumentSerializer

class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    @action(detail=False, methods=['POST'])
    def upload_documents(self, request):
        files = request.FILES.getlist('files')
        uploaded_docs = []
        
        if not files:
            return Response(
                {"error": "No files uploaded"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        for file in files:
            try:
                # Read file content
                content = file.read().decode('utf-8').strip()
                
                # Create Document instance
                doc = Document.objects.create(
                    title=file.name,
                    content=content
                )
                uploaded_docs.append(doc.title)
            
            except Exception as e:
                return Response(
                    {"error": f"Error processing file {file.name}: {str(e)}"}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response({
            "message": f"Successfully uploaded {len(uploaded_docs)} documents", 
            "uploaded_documents": uploaded_docs
        }, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['GET'])
    def list_documents(self, request):
        documents = Document.objects.all()
        serializer = self.get_serializer(documents, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['GET'])
    def search_by_title(self, request):
        query = request.query_params.get('query', '')
        results = Document.search_by_title(query)
        serializer = self.get_serializer(results, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['POST'])
    def search_by_content(self, request):
        query = request.data.get('query', '')
        results = Document.search_by_content(query)
        serializer = self.get_serializer(results, many=True)
        return Response(serializer.data)