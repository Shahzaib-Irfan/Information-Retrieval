from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response

from .models import BeliefNetworkDocument
from .serializers import BeliefNetworkDocumentSerializer
from .probabilistic_ranker import InformationRetrievalModels

class BeliefNetworkViewSet(viewsets.ModelViewSet):
    """
    ViewSet for BeliefNetwork Document Search and Retrieval
    """
    queryset = BeliefNetworkDocument.objects.all()
    serializer_class = BeliefNetworkDocumentSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def create(self, request, *args, **kwargs):
        """
        Handle multiple document uploads
        """
        files = request.FILES.getlist('files')
        
        if not files:
            return Response(
                {"error": "No files uploaded"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Upload documents
            uploaded_docs = BeliefNetworkDocument.upload_documents(files)
            
            # Serialize and return uploaded documents
            serializer = self.get_serializer(uploaded_docs, many=True)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            return Response(
                {"error": f"Upload failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['POST'])
    def interference_model_search(self, request):
        """
        Perform search using Interference Model
        """
        query = request.data.get('query')
        
        if not query:
            return Response(
                {"error": "No search query provided"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Retrieve all documents
            documents = [
                (doc.title, doc.content) 
                for doc in BeliefNetworkDocument.objects.all()
            ]
            
            # Initialize IR Models
            ir_models = InformationRetrievalModels(documents)
            
            # Compute relevance judgments and perform interference model search
            ir_models.create_relevance_judgments([query])
            results = ir_models.interference_model(query)
            
            # Prepare serialized results
            serialized_results = []
            for doc_idx, score in results:
                if score > 0:
                    doc = BeliefNetworkDocument.objects.all()[doc_idx]
                    serializer = self.get_serializer(doc)
                    result_data = serializer.data
                    result_data['relevance_score'] = score
                    result_data['rank'] = results.index((doc_idx, score)) + 1
                    serialized_results.append(result_data)
            
            return Response(serialized_results)
        
        except Exception as e:
            return Response(
                {"error": f"Search failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['POST'])
    def belief_network_search(self, request):
        """
        Perform search using Belief Network Model
        """
        query = request.data.get('query')
        
        if not query:
            return Response(
                {"error": "No search query provided"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Retrieve all documents
            documents = [
                (doc.title, doc.content) 
                for doc in BeliefNetworkDocument.objects.all()
            ]
            
            # Initialize IR Models
            ir_models = InformationRetrievalModels(documents)
            
            # Compute relevance judgments and perform belief network search
            ir_models.create_relevance_judgments([query])
            results = ir_models.belief_network(query)
            
            # Prepare serialized results
            serialized_results = []
            for doc_idx, score in results:
                if score > 0:
                    doc = BeliefNetworkDocument.objects.all()[doc_idx]
                    serializer = self.get_serializer(doc)
                    result_data = serializer.data
                    result_data['relevance_score'] = score
                    result_data['rank'] = results.index((doc_idx, score)) + 1
                    serialized_results.append(result_data)
            
            return Response(serialized_results)
        
        except Exception as e:
            return Response(
                {"error": f"Search failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )