import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List, Dict, Any
from config import EXPERIMENT_CONFIGS, ExperimentConfig
from metrics import MetricsTracker
from ingest import process_documents
from search import search_documents
import time

# Set page config
st.set_page_config(
    page_title="RAG Experiment Dashboard",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'metrics_tracker' not in st.session_state:
    st.session_state.metrics_tracker = MetricsTracker()
if 'current_experiment' not in st.session_state:
    st.session_state.current_experiment = None
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = {}

def load_experiment_results():
    """Load experiment results from JSON files."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return {}
    
    results = {}
    for file in os.listdir(results_dir):
        if file.endswith('.json'):
            with open(os.path.join(results_dir, file), 'r') as f:
                data = json.load(f)
                config_name = data['config']
                if config_name not in results:
                    results[config_name] = []
                results[config_name].append(data)
    return results

def plot_processing_metrics(metrics_df: pd.DataFrame):
    """Plot processing metrics."""
    if metrics_df.empty:
        st.info("No processing metrics available yet. Run an experiment to see the results.")
        return None

    # Filter for processing metrics only
    processing_df = metrics_df[metrics_df['query_num'].isna()]
    if processing_df.empty:
        st.info("No processing metrics available yet. Run an experiment to see the results.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot embedding time
    sns.barplot(data=processing_df, x='config', y='embedding_time', ax=ax1)
    ax1.set_title('Embedding Generation Time')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (seconds)')
    
    # Plot indexing time
    sns.barplot(data=processing_df, x='config', y='indexing_time', ax=ax2)
    ax2.set_title('Indexing Time')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    return fig

def plot_search_metrics(metrics_df: pd.DataFrame):
    """Plot search metrics."""
    if metrics_df.empty:
        st.info("No search metrics available yet. Run an experiment to see the results.")
        return None

    # Filter for search metrics only
    search_df = metrics_df[metrics_df['query_num'].notna()]
    if search_df.empty:
        st.info("No search metrics available yet. Run an experiment to see the results.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot total query time
    sns.barplot(data=search_df, x='config', y='query_time', ax=ax1)
    ax1.set_title('Total Query Time')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (seconds)')
    
    # Plot LLM time
    sns.barplot(data=search_df, x='config', y='llm_time', ax=ax2)
    ax2.set_title('LLM Response Time')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    return fig

def plot_similarity_scores(metrics_df: pd.DataFrame):
    """Plot similarity scores."""
    if metrics_df.empty:
        st.info("No similarity scores available yet. Run an experiment to see the results.")
        return None

    # Filter for search metrics only
    search_df = metrics_df[metrics_df['query_num'].notna()]
    if search_df.empty:
        st.info("No similarity scores available yet. Run an experiment to see the results.")
        return None

    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(data=search_df, x='config', y='avg_similarity')
    plt.title('Similarity Scores Distribution')
    plt.xlabel('Configuration')
    plt.ylabel('Similarity Score')
    return fig

def run_experiment(config: ExperimentConfig, data_dir: str):
    """Run a single experiment configuration."""
    st.write(f"Running experiment: {config.name}")
    
    # Process documents
    with st.spinner("Processing documents..."):
        process_documents(config, data_dir, st.session_state.metrics_tracker)
    
    # Run test queries
    test_queries = [
        "What are the main topics covered in the course?",
        "How does vector similarity search work?",
        "What are the different types of embeddings?",

    ]
    
    results = []
    for query in test_queries:
        with st.spinner(f"Running query: {query}"):
            search_results, response = search_documents(config, query, st.session_state.metrics_tracker)
            results.append({
                "query": query,
                "results": search_results,
                "response": response
            })
    
    return results

def main():
    st.title("🔬 RAG Experiment Dashboard")
    
    # Sidebar for experiment controls
    with st.sidebar:
        st.header("Experiment Controls")
        
        # Select experiment configuration
        config_name = st.selectbox(
            "Select Experiment Configuration",
            options=[config.name for config in EXPERIMENT_CONFIGS]
        )
        
        # Load selected config
        selected_config = next((c for c in EXPERIMENT_CONFIGS if c.name == config_name), None)
        
        if selected_config:
            st.write("Configuration Details:")
            st.json({
                "Chunking": {
                    "chunk_size": selected_config.chunking.chunk_size,
                    "overlap": selected_config.chunking.overlap,
                    "preprocessing": selected_config.chunking.preprocessing
                },
                "Embedding": {
                    "model": selected_config.embedding.model
                },
                "Vector DB": {
                    "type": selected_config.vector_db.type
                },
                "LLM": {
                    "model": selected_config.llm.model,
                    "temperature": selected_config.llm.temperature
                }
            })
        
        # Run experiment button
        if st.button("Run Experiment"):
            if selected_config:
                st.session_state.current_experiment = selected_config
                results = run_experiment(selected_config, "data")
                st.session_state.experiment_results[selected_config.name] = results
                st.success("Experiment completed!")
    
    # Main content area
    if st.session_state.current_experiment:
        st.header(f"Results for {st.session_state.current_experiment.name}")
        
        # Display metrics
        metrics_df = st.session_state.metrics_tracker.get_metrics_df()
        
        # Processing metrics
        st.subheader("Processing Metrics")
        processing_fig = plot_processing_metrics(metrics_df)
        if processing_fig:
            st.pyplot(processing_fig)
        
        # Search metrics
        st.subheader("Search Metrics")
        search_fig = plot_search_metrics(metrics_df)
        if search_fig:
            st.pyplot(search_fig)
        
        # Similarity scores
        st.subheader("Similarity Scores")
        similarity_fig = plot_similarity_scores(metrics_df)
        if similarity_fig:
            st.pyplot(similarity_fig)
        
        # Query results
        st.subheader("Query Results")
        results = st.session_state.experiment_results[st.session_state.current_experiment.name]
        
        for result in results:
            with st.expander(f"Query: {result['query']}"):
                st.write("Response:", result['response'])
                st.write("Relevant Chunks:")
                for chunk in result['results']:
                    st.write(f"- {chunk['chunk']} (Similarity: {chunk['similarity']:.2f})")
    
    # LLM Comparison
    st.header("LLM Model Comparison")
    
    # Load all experiment results
    all_results = load_experiment_results()
    
    if all_results:
        # Create comparison dataframe
        comparison_data = []
        for config_name, results in all_results.items():
            for result in results:
                comparison_data.append({
                    "config": config_name,
                    "query": result['query'],
                    "response": result['response']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison
        st.write("### Response Comparison")
        for query in comparison_df['query'].unique():
            st.write(f"**Query:** {query}")
            query_results = comparison_df[comparison_df['query'] == query]
            for _, row in query_results.iterrows():
                st.write(f"**{row['config']}:**")
                st.write(row['response'])
                st.write("---")
    else:
        st.info("Run some experiments to see LLM comparisons!")

if __name__ == "__main__":
    main() 