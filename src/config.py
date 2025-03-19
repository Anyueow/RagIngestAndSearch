from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ChunkingConfig:
    chunk_size: int
    overlap: int
    preprocessing: List[str]  # List of preprocessing steps to apply

@dataclass
class EmbeddingConfig:
    name: str
    model: str
    dimension: int

@dataclass
class LLMConfig:
    name: str
    model: str
    temperature: float = 0.7

@dataclass
class VectorDBConfig:
    name: str
    type: str  # 'redis', 'chroma', or other
    connection_params: dict

@dataclass
class ExperimentConfig:
    name: str
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    vector_db: VectorDBConfig

# Available configurations
CHUNKING_CONFIGS = [
    ChunkingConfig(chunk_size=200, overlap=0, preprocessing=[]),
    ChunkingConfig(chunk_size=500, overlap=50, preprocessing=[]),
    ChunkingConfig(chunk_size=1000, overlap=100, preprocessing=[]),
]

EMBEDDING_CONFIGS = [
    EmbeddingConfig(name="nomic-embed-text", model="nomic-embed-text", dimension=768),
    EmbeddingConfig(name="all-MiniLM-L6-v2", model="sentence-transformers/all-MiniLM-L6-v2", dimension=384),
    EmbeddingConfig(name="all-mpnet-base-v2", model="sentence-transformers/all-mpnet-base-v2", dimension=768),
]

LLM_CONFIGS = [
    LLMConfig(name="mistral", model="mistral:latest"),
    LLMConfig(name="llama2", model="llama2:7b"),
]

VECTOR_DB_CONFIGS = [
    VectorDBConfig(
        name="redis",
        type="redis",
        connection_params={"host": "localhost", "port": 6379, "db": 0}
    ),
    VectorDBConfig(
        name="chroma",
        type="chroma",
        connection_params={"persist_directory": "./chroma_db"}
    ),
]

# Example experiment configurations
EXPERIMENT_CONFIGS = [
    ExperimentConfig(
        name="baseline",
        chunking=CHUNKING_CONFIGS[0],
        embedding=EMBEDDING_CONFIGS[0],
        llm=LLM_CONFIGS[0],
        vector_db=VECTOR_DB_CONFIGS[0]
    ),
    # Add more experiment configurations as needed
] 