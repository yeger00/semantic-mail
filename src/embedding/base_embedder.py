from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from src.embedding.simplify_html import simplify_html
from ..models import Email
from rich.console import Console
console = Console()

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
    
    def embed_emails(self, emails: List[Email]) -> List[Tuple[Email, Optional[List[float]]]]:
        """Generate embeddings for a list of emails"""
        console.print(
            f"[bold blue]Generating embeddings for {len(emails)} emails...[/bold blue]"
        )
        texts = [simplify_html(email.content_for_embedding) for email in emails]
        embeddings = self.generate_embeddings_batch(texts)

        results = list(zip(emails, embeddings))

        successful = sum(1 for _, emb in results if emb is not None)
        console.print(
            f"[green]Successfully generated {successful}/{len(emails)} embeddings[/green]"
        )

        return results

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the embedder is properly configured and can connect"""
        pass 