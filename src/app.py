import streamlit as st
import os
from ingest import clear_redis_store, create_hnsw_index, process_pdfs
from search import search_embeddings, generate_rag_response
import tempfile

st.set_page_config(page_title="RAG Document Search", layout="wide")

st.title("📚 RAG Document Search System")

# Sidebar for document upload and processing
with st.sidebar:
    st.header("Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Documents", 
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
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        # Clear existing store and create new index
                        clear_redis_store()
                        create_hnsw_index()
                        
                        # Process the uploaded files
                        process_pdfs(temp_dir)
                        st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")

# Main content area for search
st.header("Search Documents")

# Search interface
search_query = st.text_input("Enter your question:", placeholder="Ask a question about your documents...")

if search_query:
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
    1. **Upload Documents**: Use the sidebar to upload PDF documents
    2. **Process Documents**: Click the 'Process Documents' button to generate embeddings
    3. **Search**: Enter your question in the search box
    4. **View Results**: 
       - See relevant document chunks
       - Get AI-generated answers based on the documents
    """)

# Add status indicators
col1, col2 = st.columns(2)
with col1:
    st.metric("Redis Status", "Connected" if os.system("redis-cli ping") == 0 else "Disconnected")
with col2:
    st.metric("Ollama Status", "Running" if os.system("curl http://localhost:11434/api/version") == 0 else "Not Running") 