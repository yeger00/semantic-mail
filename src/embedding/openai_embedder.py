from typing import List, Optional, Tuple
import openai
from tqdm import tqdm
from rich.console import Console

from .base_embedder import BaseEmbedder
from ..models import Email
from ..config import get_settings


console = Console()


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        if not self.settings.openai_api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env file")
        
        self.model_name = model_name or self.settings.openai_model
        self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
        self._embedding_dimension = self._get_model_dimension()
    
    def get_model_id(self) -> str:
        return f"openai_{self.model_name.replace('-', '_')}"
    
    def get_embedding_dimension(self) -> int:
        return self._embedding_dimension
    
    def _get_model_dimension(self) -> int:
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model_name, 1536)
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )
            
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                console.print(f"[red]Unexpected response format from OpenAI[/red]")
                return None
                
        except Exception as e:
            console.print(f"[red]Error generating embedding: {e}[/red]")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        embeddings = []
        batch_size = 100
        
        with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                        encoding_format="float"
                    )
                    
                    for data in response.data:
                        embeddings.append(data.embedding)
                    
                except Exception as e:
                    console.print(f"[red]Error in batch {i//batch_size}: {e}[/red]")
                    embeddings.extend([None] * len(batch))
                
                pbar.update(len(batch))
        
        return embeddings
    
    def embed_emails(self, emails: List[Email]) -> List[Tuple[Email, Optional[List[float]]]]:
        console.print(f"[bold blue]Generating embeddings for {len(emails)} emails using OpenAI {self.model_name}...[/bold blue]")
        
        texts = [email.content_for_embedding for email in emails]
        embeddings = self.generate_embeddings_batch(texts)
        
        results = list(zip(emails, embeddings))
        
        successful = sum(1 for _, emb in results if emb is not None)
        console.print(f"[green]Successfully generated {successful}/{len(emails)} embeddings[/green]")
        
        return results
    
    def test_connection(self) -> bool:
        try:
            console.print(f"[green]Testing OpenAI API connection...[/green]")
            console.print(f"[blue]Using model: {self.model_name} (dimension: {self._embedding_dimension})[/blue]")
            
            test_embedding = self.generate_embedding("Test connection")
            if test_embedding:
                console.print(f"[green]OpenAI embedding generation test successful[/green]")
                return True
            else:
                console.print("[red]OpenAI embedding generation test failed[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Failed to connect to OpenAI: {e}[/red]")
            return False 