# Ollama RAG Ingest and Search

A RAG (Retrieval Augmented Generation) system that processes PDF documents and enables semantic search with natural language queries. This project includes experimental features to compare different configurations of the RAG pipeline.

## Prerequisites

1. **Ollama Installation**
   - Download and install Ollama from [Ollama.com](https://ollama.com/download)
   - After installation, start the Ollama service:
     ```bash
     ollama serve
     ```
   - Pull required models:
     ```bash
     ollama pull nomic-embed-text
     ollama pull mistral
     ollama pull llama2:7b
     ```

2. **Redis Stack**
   - Install Docker if you haven't already
   - Run Redis Stack container:
     ```bash
     docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
     ```
   - Note: The default Redis port is 6379, which is what the code uses

3. **Python Environment**
   - Create and activate a virtual environment (recommended):
     ```bash
     conda create -n ds4300_project python=3.11
     conda activate ds4300_project
     ```
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```

## Project Structure
- `src/ingest.py` - Processes PDF files and generates embeddings
- `src/search.py` - Provides search functionality
- `src/app.py` - Streamlit web interface
- `src/config.py` - Configuration management for experiments
- `src/metrics.py` - Metrics tracking for experiments
- `src/vector_store.py` - Vector database abstractions
- `src/embeddings.py` - Embedding model abstractions
- `src/llm.py` - LLM model abstractions
- `src/run_experiments.py` - Driver script for running experiments
- `data/` - Directory containing PDF files to process
- `metrics/` - Directory for storing experiment metrics
- `results/` - Directory for storing experiment results

## Usage

### Command Line Interface

1. **Ingest Documents**
   ```bash
   python src/ingest.py
   ```
   This will:
   - Clear any existing Redis store
   - Create a new HNSW index
   - Process all PDFs in the `data` directory
   - Generate embeddings and store them in Redis

2. **Search Documents**
   ```bash
   python src/search.py
   ```
   This will:
   - Start an interactive search interface
   - Allow you to ask questions about the documents
   - Use RAG to provide answers based on the most relevant document chunks

### Web Interface (Recommended)

Run the Streamlit app:
```bash
streamlit run src/app.py
```

The web interface provides:
- Easy document upload through a web browser
- Visual document processing status
- Interactive search interface
- Display of relevant document chunks
- AI-generated answers
- System status monitoring

### Running Experiments

The project includes experimental features to compare different configurations:

1. **Chunking Strategies**:
   - Different chunk sizes (200, 500, 1000 tokens)
   - Different overlap sizes (0, 50, 100 tokens)
   - Text preprocessing options

2. **Embedding Models**:
   - nomic-embed-text
   - all-MiniLM-L6-v2
   - all-mpnet-base-v2

3. **Vector Databases**:
   - Redis Vector DB
   - Chroma

4. **LLM Models**:
   - Mistral
   - Llama 2 7B

To run experiments:
```bash
python src/run_experiments.py
```

This will:
1. Process documents with different configurations
2. Run test queries
3. Record metrics (processing time, memory usage, etc.)
4. Save results in the `results/` directory
5. Save metrics in the `metrics/` directory

## Experiment Results

The experiment results include:
- Processing metrics (time, memory usage)
- Search metrics (query time, retrieval time, LLM time)
- Quality metrics (similarity scores)
- Generated responses

Results are saved in JSON format for easy analysis.

## Troubleshooting

1. **Ollama Issues**
   - If you get "address already in use" error, Ollama is already running
   - If you get "model not found" errors, make sure you've pulled the required models
   - Ensure Ollama service is running with `ollama serve`

2. **Redis Issues**
   - If Redis connection fails, ensure Redis Stack container is running
   - Check container status with `docker ps | grep redis-stack`
   - Restart container if needed: `docker restart redis-stack`

3. **PDF Processing Issues**
   - Ensure PDFs are in the `data` directory
   - Check PDF file permissions
   - Verify PDFs are not corrupted

4. **Streamlit Issues**
   - If the web interface doesn't load, ensure all dependencies are installed
   - Check if the port 8501 is available
   - Try running with `streamlit run src/app.py --server.port 8502` if port 8501 is in use

## Dependencies
- ollama
- redis
- numpy
- PyMuPDF (for PDF processing)
- sentence-transformers
- streamlit
- psutil (for metrics)
- chromadb
- pandas
- matplotlib
- seaborn
