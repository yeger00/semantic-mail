from typing import List, Optional
import openai
from rich.console import Console

from .base_llm import BaseLLM
from ..models import Email
from ..config import get_settings


console = Console()


class OpenAILLM(BaseLLM):
    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        if not self.settings.openai_api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env file")
        
        self.model_name = model_name or 'gpt-4o-mini'
        self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
    
    def get_model_id(self) -> str:
        return f"openai_{self.model_name.replace('-', '_')}"
    
    def answer_question(self, question: str, emails: List[Email], max_tokens: int = 500) -> str:
        email_context = self._format_emails_for_context(emails)
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on email content. Analyze the provided emails carefully and give clear, accurate answers. Only use information from the provided emails. If the emails don't contain enough information to fully answer the question, acknowledge what you can answer and note what's missing."
            },
            {
                "role": "user",
                "content": f"Based on these emails, please answer: {question}\n\nEmail context:\n{email_context}"
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
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
            console.print(f"[green]Testing OpenAI API connection...[/green]")
            console.print(f"[blue]Using model: {self.model_name}[/blue]")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            if response.choices and response.choices[0].message:
                console.print(f"[green]OpenAI LLM test successful[/green]")
                return True
            else:
                console.print("[red]OpenAI LLM test failed[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Failed to connect to OpenAI: {e}[/red]")
            return False 