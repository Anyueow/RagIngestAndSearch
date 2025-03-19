from abc import ABC, abstractmethod
import numpy as np
import redis
import chromadb
from typing import List, Dict, Any
from config import VectorDBConfig

class VectorStore(ABC):
    @abstractmethod
    def store_embedding(self, key: str, embedding: List[float], metadata: Dict[str, Any]):
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear(self):
        pass

class RedisVectorStore(VectorStore):
    def __init__(self, config: VectorDBConfig):
        self.client = redis.Redis(**config.connection_params)
        self.index_name = "embedding_index"
        self.doc_prefix = "doc:"
        self.vector_dim = 768  # Default dimension, should be configurable

    def store_embedding(self, key: str, embedding: List[float], metadata: Dict[str, Any]):
        full_key = f"{self.doc_prefix}{key}"
        metadata["embedding"] = np.array(embedding, dtype=np.float32).tobytes()
        self.client.hset(full_key, mapping=metadata)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
        
        # Use Redis Search for vector similarity search
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("file", "page", "chunk", "vector_distance")
            .dialect(2)
        )
        
        results = self.client.ft(self.index_name).search(
            q, query_params={"vec": query_vector}
        )
        
        return [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

    def clear(self):
        self.client.flushdb()

class ChromaVectorStore(VectorStore):
    def __init__(self, config: VectorDBConfig):
        self.client = chromadb.PersistentClient(path=config.connection_params["persist_directory"])
        self.collection = self.client.create_collection("documents")

    def store_embedding(self, key: str, embedding: List[float], metadata: Dict[str, Any]):
        self.collection.add(
            embeddings=[embedding],
            documents=[metadata.get("chunk", "")],
            metadatas=[metadata],
            ids=[key]
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return [
            {
                "file": metadata["file"],
                "page": metadata["page"],
                "chunk": metadata["chunk"],
                "similarity": distance,
            }
            for metadata, distance in zip(results["metadatas"][0], results["distances"][0])
        ]

    def clear(self):
        self.client.delete_collection("documents")
        self.collection = self.client.create_collection("documents")

def create_vector_store(config: VectorDBConfig) -> VectorStore:
    """Factory function to create the appropriate vector store."""
    if config.type == "redis":
        return RedisVectorStore(config)
    elif config.type == "chroma":
        return ChromaVectorStore(config)
    else:
        raise ValueError(f"Unsupported vector store type: {config.type}") 