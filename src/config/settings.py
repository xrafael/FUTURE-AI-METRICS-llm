"""Configuration settings for the universality assessment system."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM service."""
    host: str = "localhost:11434"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    temperature: float = 0.0
    timeout: float = 60.0
    num_ctx: int = 4096


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    max_chars: int = 2000
    batch_size: int = 50
    internal_batch_size: int = 16


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    pdf_chunk_size: int = 500
    pdf_overlap: int = 100
    readme_chunk_size: int = 400
    readme_overlap: int = 80


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 3
    max_chunks_warning: int = 2000


@dataclass
class AssessmentConfig:
    """Main configuration for assessment system."""
    ollama: OllamaConfig = None
    embedding: EmbeddingConfig = None
    chunking: ChunkingConfig = None
    retrieval: RetrievalConfig = None
    results_dir: Path = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.ollama is None:
            self.ollama = OllamaConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.results_dir is None:
            self.results_dir = Path(__file__).parent.parent.parent / "results"

