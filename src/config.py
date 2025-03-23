from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ChunkingConfig:
    chunk_size: int = 300
    overlap: int = 50
    preprocessing: List[str] = None

    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = []

@dataclass
class EmbeddingConfig:
    model: str = "nomic-embed-text"
    dimension: int = 768

@dataclass
class VectorDBConfig:
    type: str = "redis"  # or "chroma"
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    index_name: str = "embedding_index"
    doc_prefix: str = "doc:"
    vector_dim: int = 768
    distance_metric: str = "COSINE"

@dataclass
class LLMConfig:
    model: str = "mistral"
    temperature: float = 0.7
    max_tokens: int = 500

@dataclass
class ExperimentConfig:
    name: str
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    llm: LLMConfig

# Define experiment configurations
EXPERIMENT_CONFIGS = [
    ExperimentConfig(
        name="baseline",
        chunking=ChunkingConfig(),
        embedding=EmbeddingConfig(),
        vector_db=VectorDBConfig(),
        llm=LLMConfig()
    ),
    ExperimentConfig(
        name="large_chunks",
        chunking=ChunkingConfig(chunk_size=500, overlap=100),
        embedding=EmbeddingConfig(),
        vector_db=VectorDBConfig(),
        llm=LLMConfig()
    ),
    ExperimentConfig(
        name="llama2",
        chunking=ChunkingConfig(),
        embedding=EmbeddingConfig(),
        vector_db=VectorDBConfig(),
        llm=LLMConfig(model="llama2:7b")
    ),
    ExperimentConfig(
        name="chroma_db",
        chunking=ChunkingConfig(),
        embedding=EmbeddingConfig(),
        vector_db=VectorDBConfig(type="chroma"),
        llm=LLMConfig()
    )
] 