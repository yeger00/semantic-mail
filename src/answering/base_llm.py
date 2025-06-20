from abc import ABC, abstractmethod
from typing import List, Optional
from ..models import Email


class BaseLLM(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def get_model_id(self) -> str:
        """Return a unique identifier for this model/provider combination"""
        pass
    
    @abstractmethod
    def answer_question(self, question: str, emails: List[Email], max_tokens: int = 500) -> str:
        """Answer a question based on the provided emails"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the LLM is properly configured and can connect"""
        pass 