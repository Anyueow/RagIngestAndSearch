## DS 4300 Example - from docs

import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
from typing import List, Dict, Any
from config import ChunkingConfig, EmbeddingConfig, VectorDBConfig, ExperimentConfig
from vector_store import create_vector_store
from embeddings import create_embedding_model
from metrics import MetricsTracker
import time
import json

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {chunk}")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir, chunking_config=None):
    """Process all PDF files in a given directory.
    
    Args:
        data_dir (str): Directory containing PDF files
        chunking_config (ChunkingConfig, optional): Configuration for text chunking
    """
    if chunking_config is None:
        chunking_config = ChunkingConfig(
            chunk_size=300,
            overlap=50,
            preprocessing=[]
        )

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunking_config.chunk_size, chunking_config.overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")


def main():
    clear_redis_store()
    create_hnsw_index()

    process_pdfs("./data/")
    print("\n---Done processing PDFs---\n")
    query_redis("What is the capital of France?")


if __name__ == "__main__":
    main()

class DocumentProcessor:
    def __init__(self, config: ExperimentConfig, metrics_tracker: MetricsTracker):
        self.config = config
        self.metrics_tracker = metrics_tracker
        self.vector_store = create_vector_store(config.vector_db)
        self.embedding_model = create_embedding_model(config.embedding)
        # Define important keywords for filtering
        self.keywords = {
            "database": ["database", "dbms", "sql", "query", "table", "schema", "index"],
            "relational": ["relational", "relation", "tuple", "attribute", "key", "foreign key"],
            "algebra": ["algebra", "operation", "join", "select", "project", "union", "intersection"],
            "normalization": ["normalization", "normal form", "dependency", "decomposition"],
            "transaction": ["transaction", "acid", "commit", "rollback", "concurrency"]
        }

    def check_keywords(self, text: str) -> Dict[str, int]:
        """Check for presence of important keywords in text."""
        text = text.lower()
        keyword_counts = {}
        for category, words in self.keywords.items():
            count = sum(1 for word in words if word in text)
            if count > 0:
                keyword_counts[category] = count
        return keyword_counts

    def clear_store(self):
        """Clear the vector store."""
        self.vector_store.clear()

    def create_index(self):
        """Create a new index in the vector store."""
        # This is currently only needed for Redis
        if hasattr(self.vector_store, 'create_index'):
            self.vector_store.create_index()

    def extract_text_from_pdf(self, pdf_path: str) -> List[tuple]:
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text_by_page.append((page_num, page.get_text()))
        return text_by_page

    def preprocess_text(self, text: str) -> str:
        """Apply text preprocessing steps."""
        for step in self.config.chunking.preprocessing:
            if step == "remove_whitespace":
                text = " ".join(text.split())
            elif step == "remove_punctuation":
                # Add punctuation removal logic here
                pass
        return text

    def split_text_into_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks based on configuration and check for keywords."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.config.chunking.chunk_size - self.config.chunking.overlap):
            chunk = " ".join(words[i : i + self.config.chunking.chunk_size])
            chunk = self.preprocess_text(chunk)
            
            # Check for keywords in the chunk
            keyword_counts = self.check_keywords(chunk)
            
            chunks.append({
                "text": chunk,
                "keywords": keyword_counts,
                "has_keywords": len(keyword_counts) > 0
            })
        return chunks

    def process_pdfs(self, data_dir: str):
        """Process all PDF files in a given directory."""
        total_chunks = 0
        chunks_with_keywords = 0
        embedding_time = 0
        indexing_time = 0

        for file_name in os.listdir(data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                text_by_page = self.extract_text_from_pdf(pdf_path)
                
                for page_num, text in text_by_page:
                    chunks = self.split_text_into_chunks(text)
                    total_chunks += len(chunks)
                    
                    for chunk_index, chunk_data in enumerate(chunks):
                        if chunk_data["has_keywords"]:
                            chunks_with_keywords += 1
                        
                        # Generate embedding
                        embedding_start = time.time()
                        embedding = self.embedding_model.get_embedding(chunk_data["text"])
                        embedding_time += time.time() - embedding_start

                        # Store in vector database
                        indexing_start = time.time()
                        metadata = {
                            "file": file_name,
                            "page": str(page_num),
                            "chunk": chunk_data["text"],
                            "keywords": json.dumps(chunk_data["keywords"]),
                            "has_keywords": chunk_data["has_keywords"]
                        }
                        key = f"{file_name}_page_{page_num}_chunk_{chunk_index}"
                        self.vector_store.store_embedding(key, embedding, metadata)
                        indexing_time += time.time() - indexing_start

        # Update metrics
        self.metrics_tracker.end_processing(
            num_chunks=total_chunks,
            embedding_time=embedding_time,
            indexing_time=indexing_time
        )
        
        # Print keyword statistics
        print(f"\nChunking Statistics:")
        print(f"Total chunks: {total_chunks}")
        print(f"Chunks with keywords: {chunks_with_keywords}")
        print(f"Percentage with keywords: {(chunks_with_keywords/total_chunks)*100:.2f}%")

def process_documents(config: ExperimentConfig, data_dir: str, metrics_tracker: MetricsTracker):
    """Main function to process documents with the given configuration."""
    processor = DocumentProcessor(config, metrics_tracker)
    processor.clear_store()
    processor.create_index()
    processor.process_pdfs(data_dir)
