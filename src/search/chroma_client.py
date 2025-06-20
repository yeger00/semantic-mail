import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import Optional


class ChromaClientManager:
    _instance: Optional['ChromaClientManager'] = None
    _client: Optional[chromadb.PersistentClient] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaClientManager, cls).__new__(cls)
        return cls._instance
    
    def get_client(self, persist_directory: Path) -> chromadb.PersistentClient:
        if self._client is None:
            persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        return self._client
    
    def reset(self):
        """Reset the client instance (useful for testing)"""
        self._client = None


def get_chroma_client(persist_directory: Path) -> chromadb.PersistentClient:
    """Get the singleton ChromaDB client"""
    manager = ChromaClientManager()
    return manager.get_client(persist_directory) 