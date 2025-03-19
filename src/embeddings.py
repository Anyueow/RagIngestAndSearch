from abc import ABC, abstractmethod
import ollama
from sentence_transformers import SentenceTransformer
from typing import List
from config import EmbeddingConfig

class EmbeddingModel(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig):
        self.model = config.model

    def get_embedding(self, text: str) -> List[float]:
        response = ollama.embeddings(model=self.model, prompt=text)
        return response["embedding"]

class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig):
        self.model = SentenceTransformer(config.model)

    def get_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

def create_embedding_model(config: EmbeddingConfig) -> EmbeddingModel:
    """Factory function to create the appropriate embedding model."""
    if config.model.startswith("nomic-embed-text"):
        return OllamaEmbeddingModel(config)
    elif config.model.startswith("sentence-transformers"):
        return SentenceTransformerEmbeddingModel(config)
    else:
        raise ValueError(f"Unsupported embedding model: {config.model}") 