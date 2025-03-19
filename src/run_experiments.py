import os
import time
from typing import List
import json
from config import EXPERIMENT_CONFIGS, ChunkingConfig, EmbeddingConfig, LLMConfig, VectorDBConfig
from metrics import MetricsTracker
from ingest import clear_redis_store, create_hnsw_index, process_pdfs
from search import search_embeddings, generate_rag_response

def run_experiment(config_name: str, data_dir: str, metrics_tracker: MetricsTracker):
    """Run a single experiment configuration."""
    # Find the experiment config
    config = next((c for c in EXPERIMENT_CONFIGS if c.name == config_name), None)
    if not config:
        raise ValueError(f"Experiment configuration '{config_name}' not found")

    # Start processing metrics
    num_documents = len([f for f in os.listdir(data_dir) if f.endswith('.pdf')])
    metrics_tracker.start_processing(config_name, num_documents)

    try:
        # Clear existing store and create new index
        clear_redis_store()
        create_hnsw_index()

        # Process documents with timing
        embedding_start = time.time()
        process_pdfs(data_dir, config.chunking)
        embedding_time = time.time() - embedding_start

        # Record processing metrics
        metrics_tracker.end_processing(
            num_chunks=0,  # TODO: Track actual number of chunks
            embedding_time=embedding_time,
            indexing_time=0  # TODO: Track actual indexing time
        )

        # Run test queries
        test_queries = [
            "What are the main topics covered in the course?",
            "How does vector similarity search work?",
            "What are the different types of embeddings?",
            # Add more test queries as needed
        ]

        for query in test_queries:
            # Time the search process
            query_start = time.time()
            results = search_embeddings(query)
            retrieval_time = time.time() - query_start

            # Time the LLM response
            llm_start = time.time()
            response = generate_rag_response(query, results)
            llm_time = time.time() - llm_start

            # Record search metrics
            similarities = [float(r['similarity']) for r in results]
            metrics_tracker.record_search(
                query_time=retrieval_time + llm_time,
                retrieval_time=retrieval_time,
                llm_time=llm_time,
                num_results=len(results),
                similarities=similarities
            )

            # Save results
            save_query_results(config_name, query, results, response)

    except Exception as e:
        print(f"Error running experiment {config_name}: {str(e)}")
        raise

def save_query_results(config_name: str, query: str, results: List[dict], response: str):
    """Save query results to a JSON file."""
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{config_name}_{query[:30]}.json")
    with open(output_file, 'w') as f:
        json.dump({
            "config": config_name,
            "query": query,
            "results": results,
            "response": response
        }, f, indent=2)

def main():
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()

    # Data directory
    data_dir = "data"

    # Run each experiment configuration
    for config in EXPERIMENT_CONFIGS:
        print(f"\nRunning experiment: {config.name}")
        try:
            run_experiment(config.name, data_dir, metrics_tracker)
            metrics_tracker.save_metrics()
        except Exception as e:
            print(f"Failed to run experiment {config.name}: {str(e)}")

if __name__ == "__main__":
    main() 