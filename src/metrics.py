import time
import psutil
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os

@dataclass
class ProcessingMetrics:
    start_time: float
    end_time: float
    memory_usage: float
    num_documents: int
    num_chunks: int
    embedding_time: float
    indexing_time: float

@dataclass
class SearchMetrics:
    query_time: float
    retrieval_time: float
    llm_time: float
    num_results: int
    average_similarity: float

@dataclass
class ExperimentMetrics:
    config_name: str
    processing: ProcessingMetrics
    search_metrics: List[SearchMetrics]

class MetricsTracker:
    def __init__(self, output_dir: str = "metrics"):
        self.output_dir = output_dir
        self.current_experiment = None
        self.current_metrics = None
        os.makedirs(output_dir, exist_ok=True)

    def start_processing(self, config_name: str, num_documents: int):
        self.current_experiment = config_name
        self.current_metrics = ExperimentMetrics(
            config_name=config_name,
            processing=ProcessingMetrics(
                start_time=time.time(),
                end_time=0,
                memory_usage=0,
                num_documents=num_documents,
                num_chunks=0,
                embedding_time=0,
                indexing_time=0
            ),
            search_metrics=[]
        )

    def end_processing(self, num_chunks: int, embedding_time: float, indexing_time: float):
        if self.current_metrics:
            self.current_metrics.processing.end_time = time.time()
            self.current_metrics.processing.memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.current_metrics.processing.num_chunks = num_chunks
            self.current_metrics.processing.embedding_time = embedding_time
            self.current_metrics.processing.indexing_time = indexing_time

    def record_search(self, query_time: float, retrieval_time: float, llm_time: float,
                     num_results: int, similarities: List[float]):
        if self.current_metrics:
            self.current_metrics.search_metrics.append(SearchMetrics(
                query_time=query_time,
                retrieval_time=retrieval_time,
                llm_time=llm_time,
                num_results=num_results,
                average_similarity=np.mean(similarities) if similarities else 0
            ))

    def save_metrics(self):
        if self.current_metrics:
            output_file = os.path.join(self.output_dir, f"{self.current_experiment}_metrics.json")
            with open(output_file, 'w') as f:
                json.dump(self._metrics_to_dict(self.current_metrics), f, indent=2)

    def _metrics_to_dict(self, metrics: ExperimentMetrics) -> Dict[str, Any]:
        return {
            "config_name": metrics.config_name,
            "processing": {
                "total_time": metrics.processing.end_time - metrics.processing.start_time,
                "memory_usage_mb": metrics.processing.memory_usage,
                "num_documents": metrics.processing.num_documents,
                "num_chunks": metrics.processing.num_chunks,
                "embedding_time": metrics.processing.embedding_time,
                "indexing_time": metrics.processing.indexing_time
            },
            "search_metrics": [
                {
                    "query_time": m.query_time,
                    "retrieval_time": m.retrieval_time,
                    "llm_time": m.llm_time,
                    "num_results": m.num_results,
                    "average_similarity": m.average_similarity
                }
                for m in metrics.search_metrics
            ]
        } 