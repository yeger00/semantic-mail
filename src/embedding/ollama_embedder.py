from typing import List, Optional
import ollama
from tqdm import tqdm
from rich.console import Console

from .base_embedder import BaseEmbedder
from ..models import Email
from ..config import get_settings


console = Console()


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.ollama_model
        self.client = ollama.Client(host=self.settings.ollama_host)
        self._embedding_dimension = None
        self._ensure_model_available()

    def get_model_id(self) -> str:
        return f"ollama_{self.model_name.replace(':', '_')}"

    def get_embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            test_embedding = self.generate_embedding("test")
            if test_embedding:
                self._embedding_dimension = len(test_embedding)
            else:
                raise ValueError("Could not determine embedding dimension")
        return self._embedding_dimension

    def _ensure_model_available(self):
        try:
            response = self.client.list()
            if isinstance(response, dict) and "models" in response:
                models = response["models"]
            else:
                models = response

            model_names = [
                model["name"] if isinstance(model, dict) else str(model)
                for model in models
            ]

            if self.model_name not in model_names:
                console.print(
                    f"[yellow]Model {self.model_name} not found. Pulling...[/yellow]"
                )
                self.client.pull(self.model_name)
                console.print(
                    f"[green]Model {self.model_name} pulled successfully[/green]"
                )
        except Exception as e:
            console.print(f"[red]Error checking/pulling model: {e}[/red]")
            raise

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        try:
            response = self.client.embed(model=self.model_name, input=text)

            if "embeddings" in response and len(response["embeddings"]) > 0:
                return response["embeddings"][0]
            else:
                console.print("[red]Unexpected response format from Ollama[/red]")
                return None

        except Exception as e:
            console.print(f"[red]Error generating embedding: {e}[/red]")
            return None

    def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        embeddings = []

        with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
            for text in texts:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                pbar.update(1)

        return embeddings

    def embed_emails(
        self, emails: List[Email]
    ) -> List[tuple[Email, Optional[List[float]]]]:
        console.print(
            f"[bold blue]Generating embeddings for {len(emails)} emails...[/bold blue]"
        )

        texts = [email.content_for_embedding for email in emails]
        embeddings = self.generate_embeddings_batch(texts)

        results = list(zip(emails, embeddings))

        successful = sum(1 for _, emb in results if emb is not None)
        console.print(
            f"[green]Successfully generated {successful}/{len(emails)} embeddings[/green]"
        )

        return results

    def test_connection(self) -> bool:
        try:
            response = self.client.list()
            console.print(
                f"[green]Connected to Ollama at {self.settings.ollama_host}[/green]"
            )

            if isinstance(response, dict) and "models" in response:
                models = response["models"]
            else:
                models = response

            model_names = [
                model["name"] if isinstance(model, dict) else str(model)
                for model in models
            ]
            console.print(f"[blue]Available models: {model_names}[/blue]")

            test_embedding = self.generate_embedding("Test connection")
            if test_embedding:
                console.print(
                    f"[green]Embedding generation test successful (dimension: {len(test_embedding)})[/green]"
                )
                return True
            else:
                console.print("[red]Embedding generation test failed[/red]")
                return False

        except Exception as e:
            console.print(f"[red]Failed to connect to Ollama: {e}[/red]")
            return False
