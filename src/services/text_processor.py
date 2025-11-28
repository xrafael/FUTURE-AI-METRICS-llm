"""Text processing services for chunking and structure extraction."""
import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    page: Any
    chunk_index: int
    line_start: int
    line_end: int
    source: str
    structure: Dict[str, Any]
    char_count: int
    word_count: int


class TextProcessor:
    """Service for processing and chunking text."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Initialize text processor.
        
        Args:
            chunk_size: Size of chunks in words
            overlap: Overlap between chunks in words
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def semantic_chunk_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk texts
        """
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap (last few sentences)
                overlap_sentences = current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def extract_document_structure(self, page_content: str, page_num: int) -> Dict[str, Any]:
        """
        Extract document structure information.
        
        Args:
            page_content: Content of the page
            page_num: Page number
            
        Returns:
            Dictionary with structure information
        """
        structure = {
            "sections": [],
            "has_table": False,
            "has_figure": False,
            "keywords": []
        }
        
        # Detect section headings (common patterns)
        heading_patterns = [
            r'^\d+\.\s+[A-Z][^\n]+',  # Numbered headings
            r'^[A-Z][A-Z\s]{3,}$',     # ALL CAPS headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:',  # Title Case headings
        ]
        
        lines = page_content.split('\n')
        for i, line in enumerate(lines[:20]):  # Check first 20 lines for structure
            line_stripped = line.strip()
            if any(re.match(pattern, line_stripped) for pattern in heading_patterns):
                if len(line_stripped) < 100:  # Likely a heading
                    structure["sections"].append({
                        "heading": line_stripped,
                        "line": i + 1
                    })
        
        # Detect tables (multiple consecutive lines with | or tabs)
        if re.search(r'\|.*\|', page_content) or re.search(r'\t.*\t', page_content):
            structure["has_table"] = True
        
        # Detect figure references
        if re.search(r'(?i)(figure|fig\.|table|tab\.)\s+\d+', page_content):
            structure["has_figure"] = True
        
        # Extract potential keywords (capitalized terms, technical terms)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', page_content)
        structure["keywords"] = list(set(words[:10]))  # Top 10 unique
        
        return structure
    
    def chunk_pdf_page(self, page: Dict[str, Any], chunk_size: int = None, overlap: int = None) -> List[Chunk]:
        """
        Chunk a PDF page with semantic context preservation.
        
        Args:
            page: Page object from PDF loader
            chunk_size: Size of chunks in words (uses instance default if None)
            overlap: Overlap between chunks in words (uses instance default if None)
            
        Returns:
            List of Chunk objects
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap
        
        page_content = page.page_content
        page_num = page.metadata.get("page", 0)
        
        # Extract document structure
        structure = self.extract_document_structure(page_content, page_num)
        
        # Use semantic chunking
        text_chunks = self.semantic_chunk_text(page_content)
        
        chunks = []
        for chunk_idx, chunk_text in enumerate(text_chunks):
            # Estimate line numbers (approximate)
            lines_before = page_content[:page_content.find(chunk_text)].count('\n')
            
            chunks.append(Chunk(
                text=chunk_text,
                page=page_num,
                chunk_index=chunk_idx,
                line_start=lines_before + 1,
                line_end=lines_before + chunk_text.count('\n') + 1,
                source="pdf",
                structure=structure,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split())
            ))
        
        return chunks
    
    def chunk_readme(self, readme_content: str, chunk_size: int = None, overlap: int = None) -> List[Chunk]:
        """
        Chunk README content with semantic context preservation.
        
        Args:
            readme_content: README content as string
            chunk_size: Size of chunks in words (uses instance default if None)
            overlap: Overlap between chunks in words (uses instance default if None)
            
        Returns:
            List of Chunk objects
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap
        
        # Extract README structure (sections, code blocks, etc.)
        structure = {
            "sections": [],
            "has_code": False,
            "has_links": False
        }
        
        # Detect markdown headings
        heading_pattern = r'^#{1,6}\s+(.+)$'
        lines = readme_content.split("\n")
        
        for i, line in enumerate(lines):
            heading_match = re.match(heading_pattern, line)
            if heading_match:
                structure["sections"].append({
                    "heading": heading_match.group(1),
                    "line": i + 1,
                    "level": len(line) - len(line.lstrip('#'))
                })
        
        # Detect code blocks
        if re.search(r'```', readme_content):
            structure["has_code"] = True
        
        # Detect links
        if re.search(r'\[.*?\]\(.*?\)', readme_content):
            structure["has_links"] = True
        
        # Use semantic chunking
        text_chunks = self.semantic_chunk_text(readme_content)
        
        chunks = []
        for chunk_idx, chunk_text in enumerate(text_chunks):
            lines_before = readme_content[:readme_content.find(chunk_text)].count('\n')
            chunks.append(Chunk(
                text=chunk_text,
                page="README",
                chunk_index=chunk_idx,
                line_start=lines_before + 1,
                line_end=lines_before + chunk_text.count('\n') + 1,
                source="readme",
                structure=structure,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split())
            ))
        
        return chunks

