from typing import List, Optional
import ollama
from rich.console import Console
from datetime import datetime

from .base_llm import BaseLLM
from ..models import Email
from ..config import get_settings


console = Console()


class OllamaLLM(BaseLLM):
    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.client = ollama.Client(host=self.settings.ollama_host)
        
        if model_name is None:
            model_name = self._get_best_available_model()
        
        self.model_name = model_name
        self._ensure_model_available()
    
    def _get_best_available_model(self) -> str:
        """Select the best available model from what's installed"""
        try:
            response = self.client.list()
            
            models = []
            if hasattr(response, 'models'):
                models = response.models
            elif isinstance(response, dict) and "models" in response:
                models = response["models"]
            elif isinstance(response, list):
                models = response
            
            available_models = []
            for model in models:
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif isinstance(model, dict):
                    model_name = model.get("name", model.get("model"))
                    if model_name:
                        available_models.append(model_name)
                else:
                    available_models.append(str(model))
            
            preferred_models = [
                'deepseek-r1:8b',
                'llama3.1:latest',
                'llama3.1',
                'mixtral:latest',
                'mixtral',
                'mistral:latest',
                'mistral',
                'gemma2:latest',
                'gemma2',
                'qwen2.5:latest',
                'qwen2.5',
                'llama3.2:latest',
                'llama3.2'
            ]
            
            for model in preferred_models:
                if model in available_models:
                    console.print(f"[green]Selected available model: {model}[/green]")
                    return model
            
            if available_models:
                model = available_models[0]
                console.print(f"[yellow]Using first available model: {model}[/yellow]")
                return model
            
            return 'llama3.2'
            
        except Exception:
            return 'llama3.2'
    
    def get_model_id(self) -> str:
        return f"ollama_{self.model_name.replace(':', '_')}"
    
    def _ensure_model_available(self):
        try:
            response = self.client.list()
            
            models = []
            if hasattr(response, 'models'):
                models = response.models
            elif isinstance(response, dict) and "models" in response:
                models = response["models"]
            elif isinstance(response, list):
                models = response
            
            model_names = []
            for model in models:
                if hasattr(model, 'model'):
                    model_names.append(model.model)
                elif isinstance(model, dict):
                    model_name = model.get("name", model.get("model"))
                    if model_name:
                        model_names.append(model_name)
                else:
                    model_names.append(str(model))
            
            model_found = False
            for available_model in model_names:
                if available_model == self.model_name or available_model.startswith(f"{self.model_name}:"):
                    model_found = True
                    break
            
            if not model_found:
                console.print(f"[yellow]Model {self.model_name} not found. Pulling...[/yellow]")
                self.client.pull(self.model_name)
                console.print(f"[green]Model {self.model_name} pulled successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error checking/pulling model: {e}[/red]")
            raise
    
    def answer_question(self, question: str, emails: List[Email], max_tokens: int = 500) -> str:
        email_context = self._format_emails_for_context(emails)
        
        prompt = f"""You are a helpful assistant analyzing emails. Based on the following emails, please answer this question: {question}

Email context:
{email_context}

Please provide a clear and concise answer based only on the information in these emails. If the emails don't contain enough information to answer the question, say so. Today's date: {datetime.now().strftime("%Y-%m-%d")}."""
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            
            return response['response'].strip()
        except Exception as e:
            console.print(f"[red]Error generating answer: {e}[/red]")
            return f"Error: Could not generate answer - {str(e)}"
    
    def _format_emails_for_context(self, emails: List[Email]) -> str:
        context_parts = []
        for i, email in enumerate(emails[:5], 1):
            context_parts.append(f"\n--- Email {i} ---")
            context_parts.append(f"From: {email.sender}")
            context_parts.append(f"Date: {email.date}")
            context_parts.append(f"Subject: {email.subject}")
            if email.body:
                body_preview = email.body[:1000] + "..." if len(email.body) > 1000 else email.body
                context_parts.append(f"Body:\n{body_preview}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def test_connection(self) -> bool:
        try:
            response = self.client.list()
            console.print(f"[green]Connected to Ollama at {self.settings.ollama_host}[/green]")
            
            models = []
            if hasattr(response, 'models'):
                models = response.models
            elif isinstance(response, dict) and "models" in response:
                models = response["models"]
            elif isinstance(response, list):
                models = response
            
            available_models = []
            for model in models:
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif isinstance(model, dict):
                    model_name = model.get("name", model.get("model"))
                    if model_name:
                        available_models.append(model_name)
                else:
                    available_models.append(str(model))
            
            model_found = False
            for available_model in available_models:
                if available_model == self.model_name or available_model.startswith(f"{self.model_name}:"):
                    model_found = True
                    console.print(f"[green]Model {self.model_name} is available (found as {available_model})[/green]")
                    break
            
            if not model_found:
                console.print(f"[yellow]Warning: Model {self.model_name} not found[/yellow]")
                console.print(f"[yellow]Available models: {', '.join(available_models)}[/yellow]")
                return False
            
            return True
                
        except Exception as e:
            console.print(f"[red]Failed to connect to Ollama: {e}[/red]")
            return False 