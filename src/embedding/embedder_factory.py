from typing import Optional, Tuple
from .base_embedder import BaseEmbedder
from .ollama_embedder import OllamaEmbedder
from .openai_embedder import OpenAIEmbedder
from ..config import get_settings
from rich.console import Console


console = Console()


OLLAMA_MODELS = {
    "nomic-embed-text": "Nomic Embed Text (768d) - Fast and efficient",
    "mxbai-embed-large": "Mixedbread AI Large (1024d) - Balanced performance",
    "all-minilm": "All-MiniLM (384d) - Lightweight",
    "llama2": "Llama 2 embeddings - Heavy but powerful"
}

OPENAI_MODELS = {
    "text-embedding-3-small": "OpenAI Small (1536d) - Cost-effective",
    "text-embedding-3-large": "OpenAI Large (3072d) - Best quality",
    "text-embedding-ada-002": "OpenAI Ada v2 (1536d) - Legacy"
}


def get_embedder(provider: Optional[str] = None, model: Optional[str] = None) -> BaseEmbedder:
    settings = get_settings()
    
    provider = provider or settings.embedding_provider
    
    if provider == "ollama":
        return OllamaEmbedder(model_name=model)
    elif provider == "openai":
        return OpenAIEmbedder(model_name=model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def get_smart_embedder(provider: Optional[str] = None, model: Optional[str] = None) -> Tuple[BaseEmbedder, str]:
    """
    Intelligently select an embedder based on available collections.
    Returns tuple of (embedder, collection_info_string)
    """
    from ..search.vector_store import EmailVectorStore
    
    # Find matching collections
    matches = EmailVectorStore.find_matching_collections(provider, model)
    
    if not matches:
        # No matching collections, create new one with defaults
        console.print("[yellow]No matching collections found. Creating new collection with default settings.[/yellow]")
        embedder = get_embedder(provider, model)
        return embedder, "new collection"
    
    # If specific model was requested, use it
    if model:
        # Sort by email count to get the most populated one
        matches.sort(key=lambda x: EmailVectorStore.get_collection_email_count(x[0]), reverse=True)
        collection_name, metadata = matches[0]
        model_id = metadata.get('model_id', '')
        
        # Extract the actual model name from model_id
        if model_id.startswith('ollama_'):
            actual_model = model_id.replace('ollama_', '').replace('_', ':')
            embedder = OllamaEmbedder(model_name=actual_model)
        elif model_id.startswith('openai_'):
            actual_model = model_id.replace('openai_', '').replace('_', '-')
            embedder = OpenAIEmbedder(model_name=actual_model)
        else:
            embedder = get_embedder(provider, model)
        
        email_count = EmailVectorStore.get_collection_email_count(collection_name)
        return embedder, f"{collection_name} ({email_count} emails)"
    
    # If only provider specified or nothing specified, pick the best match
    if len(matches) == 1:
        # Only one match, use it
        collection_name, metadata = matches[0]
        model_id = metadata.get('model_id', '')
        console.print(f"[green]Found matching collection: {collection_name}[/green]")
    else:
        # Multiple matches, pick the one with most emails
        matches_with_counts = [(name, meta, EmailVectorStore.get_collection_email_count(name)) 
                               for name, meta in matches]
        matches_with_counts.sort(key=lambda x: x[2], reverse=True)
        
        collection_name, metadata, count = matches_with_counts[0]
        model_id = metadata.get('model_id', '')
        
        console.print(f"[green]Found {len(matches)} matching collections. Using '{collection_name}' with {count} emails.[/green]")
        if len(matches) > 1:
            console.print("[dim]Use --model flag to specify a different model.[/dim]")
    
    # Create embedder based on the selected collection
    if model_id.startswith('ollama_'):
        actual_model = model_id.replace('ollama_', '').replace('_', ':')
        embedder = OllamaEmbedder(model_name=actual_model)
    elif model_id.startswith('openai_'):
        actual_model = model_id.replace('openai_', '').replace('_', '-')
        embedder = OpenAIEmbedder(model_name=actual_model)
    else:
        # Fallback
        embedder = get_embedder(provider, model)
    
    email_count = EmailVectorStore.get_collection_email_count(collection_name)
    return embedder, f"{collection_name} ({email_count} emails)"


def list_available_models():
    console.print("\n[bold cyan]Available Embedding Models:[/bold cyan]\n")
    
    console.print("[bold yellow]Ollama Models:[/bold yellow]")
    for model, desc in OLLAMA_MODELS.items():
        console.print(f"  • {model}: {desc}")
    
    console.print("\n[bold yellow]OpenAI Models:[/bold yellow]")
    for model, desc in OPENAI_MODELS.items():
        console.print(f"  • {model}: {desc}")
    
    console.print("\n[dim]Note: Ollama models need to be pulled first with 'ollama pull <model>'[/dim]")
    console.print("[dim]OpenAI models require an API key set in OPENAI_API_KEY[/dim]\n") 