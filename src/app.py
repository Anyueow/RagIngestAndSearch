import streamlit as st
import os
from ingest import clear_redis_store, create_hnsw_index, process_pdfs
from search import search_embeddings, generate_rag_response
import tempfile
import redis
import requests
import subprocess

st.set_page_config(page_title="RAG Document Search", layout="wide")

def check_redis_status():
    try:
        redis_client = redis.Redis(host="localhost", port=6379, db=0)
        redis_client.ping()
        return True
    except:
        return False

def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/version")
        return response.status_code == 200
    except:
        return False

def get_pdf_files(directory):
    """Get list of PDF files in the specified directory."""
    pdf_files = []
    for file in os.listdir(directory):
        if file.endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    return pdf_files

# Initialize session state for document processing status
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

st.title("📚 RAG Document Search System")

# Add status indicators at the top
col1, col2 = st.columns(2)
with col1:
    redis_status = check_redis_status()
    st.metric("Redis Status", "Connected" if redis_status else "Disconnected")
    if not redis_status:
        st.error("Please ensure Redis Stack is running: docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest")

with col2:
    ollama_status = check_ollama_status()
    st.metric("Ollama Status", "Running" if ollama_status else "Not Running")
    if not ollama_status:
        st.error("Please ensure Ollama is running: ollama serve")

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
    # Check for PDFs in data directory
    data_dir = "data"
    if os.path.exists(data_dir):
        pdf_files = get_pdf_files(data_dir)
        if pdf_files:
            st.write(f"Found {len(pdf_files)} PDF files in data directory")
            if not st.session_state.documents_processed:
                if st.button("Process Data Directory"):
                    if not redis_status or not ollama_status:
                        st.error("Please ensure both Redis and Ollama are running before processing documents")
                    else:
                        with st.spinner("Processing documents from data directory..."):
                            try:
                                # Clear existing store and create new index
                                clear_redis_store()
                                create_hnsw_index()
                                
                                # Process the files from data directory
                                process_pdfs(data_dir)
                                st.session_state.documents_processed = True
                                st.success("Documents processed successfully!")
                            except Exception as e:
                                st.error(f"Error processing documents: {str(e)}")
        else:
            st.warning("No PDF files found in data directory")
    
    st.divider()
    st.subheader("Manual Upload")
    
    # File uploader for manual uploads
    uploaded_files = st.file_uploader(
        "Upload Additional PDF Documents", 
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        # Create a temporary directory to store uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
            
            # Process button
            if st.button("Process Uploaded Documents"):
                if not redis_status or not ollama_status:
                    st.error("Please ensure both Redis and Ollama are running before processing documents")
                else:
                    with st.spinner("Processing uploaded documents..."):
                        try:
                            # Clear existing store and create new index
                            clear_redis_store()
                            create_hnsw_index()
                            
                            # Process the uploaded files
                            process_pdfs(temp_dir)
                            st.session_state.documents_processed = True
                            st.success("Documents processed successfully!")
                        except Exception as e:
                            st.error(f"Error processing documents: {str(e)}")

# Main content area for search
st.header("Search Documents")

# Search interface
search_query = st.text_input("Enter your question:", placeholder="Ask a question about your documents...")

if search_query:
    if not redis_status or not ollama_status:
        st.error("Please ensure both Redis and Ollama are running before searching")
    elif not st.session_state.documents_processed:
        st.error("Please process documents before searching")
    else:
        with st.spinner("Searching..."):
            try:
                # Search for relevant embeddings
                context_results = search_embeddings(search_query)
                
                if context_results:
                    # Display search results
                    st.subheader("Relevant Document Chunks:")
                    for result in context_results:
                        with st.expander(f"From {result['file']} (Page {result['page']})"):
                            st.write(result['chunk'])
                            st.write(f"Similarity: {float(result['similarity']):.2f}")
                    
                    # Generate and display RAG response
                    st.subheader("Answer:")
                    response = generate_rag_response(search_query, context_results)
                    st.write(response)
                else:
                    st.warning("No relevant information found in the documents.")
            except Exception as e:
                st.error(f"Error during search: {str(e)}")

# Add some helpful information
with st.expander("ℹ️ How to use this system"):
    st.markdown("""
    1. **Automatic Loading**: 
       - Place PDF documents in the `data` directory
       - Click "Process Data Directory" to process them
    2. **Manual Upload**: 
       - Use the "Upload Additional PDF Documents" option to add more files
       - Click "Process Uploaded Documents" to process them
    3. **Search**: 
       - Enter your question in the search box
       - View relevant document chunks and AI-generated answers
    """) 