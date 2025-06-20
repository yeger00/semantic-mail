from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from ..models import Email


class BaseEmbedder(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_model_id(self) -> str:
        """Return a unique identifier for this model/provider combination"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this model"""
        pass
    
    @abstractmethod
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    def embed_emails(self, emails: List[Email]) -> List[Tuple[Email, Optional[List[float]]]]:
        """Generate embeddings for a list of emails"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the embedder is properly configured and can connect"""
        pass 