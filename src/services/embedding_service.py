"""Embedding and indexing service using FAISS."""
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from services.text_processor import Chunk
from config.settings import EmbeddingConfig


class EmbeddingService:
    """Service for generating embeddings and building FAISS indices."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding service.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.embedder = SentenceTransformer(
            config.model_name,
            device=config.device
        )
    
    def truncate_text(self, text: str) -> str:
        """
        Truncate text to a safe length for embedding models.
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text
        """
        if len(text) <= self.config.max_chars:
            return text
        
        # Truncate at word boundary to avoid cutting words
        truncated = text[:self.config.max_chars]
        last_space = truncated.rfind(' ')
        if last_space > self.config.max_chars * 0.8:  # Only use word boundary if it's not too far back
            return truncated[:last_space] + "..."
        return truncated + "..."
    
    def build_index(self, chunks: List[Chunk]) -> Tuple[faiss.Index, np.ndarray]:
        """
        Build FAISS index incrementally in batches.
        
        Args:
            chunks: List of Chunk objects to index
            
        Returns:
            Tuple of (FAISS index, embedding vectors)
        """
        if not chunks:
            raise ValueError("No chunks provided to build index")
        
        print(f"Building index for {len(chunks)} chunks...")
        
        # Determine embedding dimension from first chunk
        sample_text = chunks[0].text if chunks else "sample"
        sample_text = self.truncate_text(sample_text)
        sample_vector = self.embedder.encode(
            [sample_text],
            show_progress_bar=False,
            batch_size=1,
            convert_to_numpy=True,
        )
        embedding_dim = sample_vector.shape[1]
        
        # Initialize FAISS index
        index = faiss.IndexFlatL2(embedding_dim)
        
        # Process chunks in batches to avoid memory issues
        all_vectors = []
        
        for i in range(0, len(chunks), self.config.batch_size):
            batch_chunks = chunks[i:i + self.config.batch_size]
            # Truncate each text to safe length before encoding
            batch_texts = [self.truncate_text(chunk.text) for chunk in batch_chunks]
            
            print(f"  Processing batch {i//self.config.batch_size + 1}/{(len(chunks) + self.config.batch_size - 1)//self.config.batch_size} ({len(batch_chunks)} chunks)...")
            
            # Encode batch
            batch_vectors = self.embedder.encode(
                batch_texts,
                show_progress_bar=False,
                batch_size=min(self.config.internal_batch_size, len(batch_texts)),
                convert_to_numpy=True,
            )
            
            # Add to index incrementally
            index.add(batch_vectors.astype('float32'))
            all_vectors.append(batch_vectors)
        
        # Concatenate all vectors for return
        vectors = np.vstack(all_vectors) if len(all_vectors) > 1 else all_vectors[0]
        
        print(f"Index built successfully with {index.ntotal} vectors")
        return index, vectors
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query string into an embedding vector.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        query = self.truncate_text(query)
        qvec = self.embedder.encode(
            [query],
            show_progress_bar=False,
            batch_size=1,
            convert_to_numpy=True,
        )
        return qvec
    
    def cleanup(self):
        """Clean up model resources."""
        try:
            if hasattr(self.embedder, 'eval'):
                self.embedder.eval()
        except Exception:
            pass

