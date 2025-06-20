from typing import Optional
from rich.console import Console
import ollama

from .base_llm import BaseLLM
from .ollama_llm import OllamaLLM
from .openai_llm import OpenAILLM
from ..config import get_settings


console = Console()


def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLM:
    """Get an LLM instance based on provider and model"""
    settings = get_settings()
    
    if provider is None:
        provider = settings.embedding_provider
    
    provider = provider.lower()
    
    if provider == 'ollama':
        return OllamaLLM(model_name=model)
    elif provider == 'openai':
        return OpenAILLM(model_name=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def list_available_llm_models():
    """List available LLM models for each provider"""
    console.print("[bold blue]Available LLM Models[/bold blue]\n")
    
    console.print("[cyan]Ollama Models (local):[/cyan]")
    
    try:
        settings = get_settings()
        client = ollama.Client(host=settings.ollama_host)
        response = client.list()
        
        models = []
        if hasattr(response, 'models'):
            models = response.models
        elif isinstance(response, dict) and "models" in response:
            models = response["models"]
        elif isinstance(response, list):
            models = response
        
        if models:
            console.print("  [green]Installed models:[/green]")
            for model in models:
                if hasattr(model, 'model'):
                    model_name = model.model
                    if hasattr(model, 'details') and hasattr(model.details, 'parameter_size'):
                        size = model.details.parameter_size
                        console.print(f"    • {model_name} ({size})")
                    else:
                        console.print(f"    • {model_name}")
                elif isinstance(model, dict):
                    model_name = model.get("name", model.get("model", str(model)))
                    if "details" in model and "parameter_size" in model["details"]:
                        size = model["details"]["parameter_size"]
                        console.print(f"    • {model_name} ({size})")
                    else:
                        console.print(f"    • {model_name}")
                else:
                    console.print(f"    • {str(model)}")
        else:
            console.print("  [yellow]No models installed[/yellow]")
            
        console.print("\n  [dim]Recommended models to install:[/dim]")
        console.print("    • deepseek-r1:8b - Best quality/size ratio")
        console.print("    • llama3.1 - More powerful than 3.2")
        console.print("    • mistral - Good balance of speed and quality")
        console.print("    • mixtral - Larger, more capable")
        console.print("    • gemma2 - Google's efficient model")
        
    except Exception as e:
        console.print(f"  [red]Could not connect to Ollama: {e}[/red]")
        console.print("  [dim]Common models:[/dim]")
        console.print("    • llama3.2 - Compact and fast")
        console.print("    • llama3.1 - More powerful")
        console.print("    • mistral - Good balance")
        console.print("    • deepseek-r1:8b - Best quality/size ratio")
    
    console.print("\n[cyan]OpenAI Models (cloud):[/cyan]")
    console.print("  • gpt-4o-mini - Fast and cost-effective")
    console.print("  • gpt-4o - Most capable")
    console.print("  • gpt-4-turbo - Previous generation, still powerful")
    console.print("  • gpt-3.5-turbo - Legacy, cheapest option") 