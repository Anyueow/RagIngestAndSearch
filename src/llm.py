from abc import ABC, abstractmethod
import ollama
from typing import List, Dict, Any
from config import LLMConfig

class LLM(ABC):
    @abstractmethod
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        pass

class OllamaLLM(LLM):
    def __init__(self, config: LLMConfig):
        self.model = config.model
        self.temperature = config.temperature

    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        # Prepare context string
        context_str = "\n".join(
            [
                f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}) "
                f"with similarity {float(result.get('similarity', 0)):.2f}"
                for result in context
            ]
        )

        # Construct prompt with context
        prompt = f"""You are a helpful AI assistant. 
        Use the following context to answer the query as accurately as possible. If the context is 
        not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

        # Generate response using Ollama
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature}
        )

        return response["message"]["content"]

def create_llm(config: LLMConfig) -> LLM:
    """Factory function to create the appropriate LLM."""
    if config.model.startswith(("mistral", "llama2")):
        return OllamaLLM(config)
    else:
        raise ValueError(f"Unsupported LLM model: {config.model}") 