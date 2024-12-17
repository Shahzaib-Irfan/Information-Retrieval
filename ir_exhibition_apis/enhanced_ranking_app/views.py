from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from .models import AdvancedDocument
from .serializers import AdvancedDocumentSerializer

class AdvancedSearchViewSet(viewsets.ModelViewSet):
    queryset = AdvancedDocument.objects.all()
    serializer_class = AdvancedDocumentSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def create(self, request, *args, **kwargs):
        """
        Handle multiple file uploads for various document types
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
                # Load document based on file type
                doc_instances = AdvancedDocument.load_document(file, file.content_type)
                uploaded_docs.extend(doc_instances)
            
            except Exception as e:
                return Response(
                    {"error": f"Error processing file {file.name}: {str(e)}"}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        # Serialize and return uploaded documents
        serializer = self.get_serializer(uploaded_docs, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['POST'])
    def binary_term_matching(self, request):
        """
        Perform binary term matching search
        """
        query = request.data.get('query')
        
        if not query:
            return Response(
                {"error": "No search query provided"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Perform binary term matching
            results = AdvancedDocument.search_binary_term_matching(query)
            
            # Serialize results
            serializer = self.get_serializer(
                [result['document'] for result in results], 
                many=True
            )
            
            # Add similarity to serialized data
            for i, result in enumerate(results):
                serializer.data[i]['similarity'] = result['similarity']
            
            return Response(serializer.data)
        
        except Exception as e:
            return Response(
                {"error": f"Search failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['POST'])
    def non_overlapping_lists(self, request):
        """
        Perform non-overlapping lists search
        """
        terms = request.data.get('terms', [])
        
        if not terms:
            return Response(
                {"error": "No search terms provided"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Perform non-overlapping lists search
            results = AdvancedDocument.search_non_overlapping_lists(terms)
            
            # Prepare serialized results
            serialized_results = {}
            for term, docs in results.items():
                serializer = self.get_serializer(docs, many=True)
                serialized_results[term] = serializer.data
            
            return Response(serialized_results)
        
        except Exception as e:
            return Response(
                {"error": f"Search failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['POST'])
    def proximal_node_search(self, request):
        """
        Perform proximal node search
        """
        entities = request.data.get('entities', [])
        window_size = request.data.get('window_size', 50)
        
        if not entities:
            return Response(
                {"error": "No entities provided"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Perform proximal node search
            results = AdvancedDocument.search_proximal_node(entities, window_size)
            
            # Prepare serialized results
            serialized_results = {}
            for entity_pair, docs in results.items():
                serialized_docs = []
                for doc_data in docs:
                    serializer = self.get_serializer(doc_data['document'])
                    serialized_doc = serializer.data
                    serialized_doc.update({
                        'context': doc_data['context'],
                        'entities_found': doc_data['entities_found'],
                        'distance': doc_data['distance']
                    })
                    serialized_docs.append(serialized_doc)
                serialized_results[entity_pair] = serialized_docs
            
            return Response(serialized_results)
        
        except Exception as e:
            return Response(
                {"error": f"Search failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )