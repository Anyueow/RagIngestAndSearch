from abc import ABC, abstractmethod
import numpy as np
import redis
import chromadb
from typing import List, Dict, Any
from config import VectorDBConfig
from redis.commands.search.query import Query

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
        self.client = redis.Redis(host=config.host, port=config.port, db=config.db)
        self.index_name = config.index_name
        self.doc_prefix = config.doc_prefix
        self.vector_dim = config.vector_dim
        self.distance_metric = config.distance_metric

    def clear(self):
        """Clear the vector store."""
        try:
            self.client.execute_command(f"FT.DROPINDEX {self.index_name} DD")
        except redis.exceptions.ResponseError:
            pass
        self.client.flushdb()

    def create_index(self):
        """Create a new HNSW index."""
        try:
            self.client.execute_command(f"FT.DROPINDEX {self.index_name} DD")
        except redis.exceptions.ResponseError:
            pass

        self.client.execute_command(
            f"""
            FT.CREATE {self.index_name} ON HASH PREFIX 1 {self.doc_prefix}
            SCHEMA text TEXT
            embedding VECTOR HNSW 6 DIM {self.vector_dim} TYPE FLOAT32 DISTANCE_METRIC {self.distance_metric}
            """
        )

    def store_embedding(self, key: str, embedding: List[float], metadata: Dict[str, Any]):
        """Store an embedding with metadata."""
        self.client.hset(
            key,
            mapping={
                "text": metadata.get("chunk", ""),
                "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                **metadata
            }
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
        
        # Fixed query syntax
        q = (
            Query("*=>[KNN $k @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("text", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        results = self.client.ft(self.index_name).search(
            q,
            query_params={
                "vec": query_vector,
                "k": top_k
            }
        )

        return [
            {
                "chunk": doc.text,
                "file": doc.file,
                "page": doc.page,
                "similarity": float(doc.vector_distance)
            }
            for doc in results.docs
        ]

class ChromaVectorStore(VectorStore):
    def __init__(self, config: VectorDBConfig):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=config.index_name,
            metadata={"hnsw:space": config.distance_metric.lower()}
        )

    def clear(self):
        """Clear the vector store."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": self.collection.metadata["hnsw:space"]}
        )

    def create_index(self):
        """No-op for Chroma as it creates index automatically."""
        pass

    def store_embedding(self, key: str, embedding: List[float], metadata: Dict[str, Any]):
        """Store an embedding with metadata."""
        self.collection.add(
            embeddings=[embedding],
            documents=[metadata.get("chunk", "")],
            metadatas=[metadata],
            ids=[key]
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return [
            {
                "chunk": doc,
                "file": metadata.get("file", ""),
                "page": metadata.get("page", ""),
                "similarity": float(distance)
            }
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

def create_vector_store(config: VectorDBConfig) -> VectorStore:
    """Factory function to create the appropriate vector store."""
    if config.type == "redis":
        return RedisVectorStore(config)
    elif config.type == "chroma":
        return ChromaVectorStore(config)
    else:
        raise ValueError(f"Unsupported vector store type: {config.type}") 