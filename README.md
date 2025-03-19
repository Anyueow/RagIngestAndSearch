# Ollama RAG Ingest and Search

A RAG (Retrieval Augmented Generation) system that processes PDF documents and enables semantic search with natural language queries.

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
- `data/` - Directory containing PDF files to process

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
- sentence-transformers (optional, for alternative embeddings)
- streamlit (for web interface)
