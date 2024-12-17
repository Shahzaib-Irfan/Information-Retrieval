from django.db import models
import uuid

class BeliefNetworkDocument(models.Model):
    """
    Model to store documents for Belief Network Information Retrieval
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title

    @classmethod
    def upload_documents(cls, files):
        """
        Class method to handle multiple document uploads
        """
        uploaded_docs = []
        for file in files:
            # Read file content
            content = file.read().decode('utf-8')
            
            # Create document instance
            doc = cls.objects.create(
                title=file.name,
                content=content
            )
            uploaded_docs.append(doc)
        
        return uploaded_docs