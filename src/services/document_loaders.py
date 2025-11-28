"""Document loading services for PDF and README files."""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    """Service for loading and processing PDF documents."""
    
    REFERENCE_MARKERS = ["References", "Bibliography"]
    
    def __init__(self):
        """Initialize PDF loader."""
        pass
    
    @staticmethod
    def _remove_references(text: str) -> Tuple[str, int]:
        """
        Remove references section from text.
        
        Args:
            text: Text content to process
            
        Returns:
            Tuple of (cleaned_text, index_of_reference_marker)
        """
        text_lower = text.lower()
        cut_index = None
        
        for marker in PDFLoader.REFERENCE_MARKERS:
            idx = text_lower.find(marker.lower())
            if idx != -1:
                cut_index = idx if cut_index is None else min(cut_index, idx)
                break
        
        if cut_index is not None:
            return text[:cut_index], cut_index
        return text, -1
    
    def load(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Load PDF and filter out bibliography pages.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of page objects (bibliography pages excluded)
        """
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load_and_split()
        
        filtered_pages = []
        for page in pages:
            cleaned, idx = self._remove_references(page.page_content)
            if cleaned.strip():
                page.page_content = cleaned
                filtered_pages.append(page)
            if idx != -1:
                print(f"Reference section found at index {idx}")
                break
        
        return filtered_pages
    
    def extract_metadata(self, pdf_path: Path) -> Dict[str, Optional[str]]:
        """
        Extract title and authors from the first page of the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with 'title' and 'authors' keys
        """
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load_and_split()
        
        if not pages:
            return {"title": None, "authors": None}
        
        # Get first page content
        first_page = pages[0].page_content
        
        # Try to extract title (usually the first few lines, often in larger font or at the top)
        lines = first_page.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        title = None
        authors = None
        
        # Title is typically one of the first substantial lines (not empty, not too short)
        # Usually appears before authors
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            if len(line) > 10 and not line.lower().startswith(('abstract', 'keywords', 'introduction')):
                # Skip common header/footer patterns
                if not re.match(r'^\d+$', line) and not re.match(r'^[A-Z\s]{1,3}$', line):
                    if title is None:
                        title = line
                        # Authors typically come after title, often on next few lines
                        # Look for lines with common author patterns (commas, "and", etc.)
                        for j in range(i + 1, min(i + 5, len(lines))):
                            author_line = lines[j]
                            # Check if line looks like authors (contains commas, "and", or multiple capitalized words)
                            if (',' in author_line or ' and ' in author_line.lower() or 
                                (len(author_line.split()) > 1 and author_line[0].isupper())):
                                authors = author_line
                                break
                        break
        
        # Fallback: if we have a title but no authors, try to find authors in next few lines
        if title and not authors:
            title_idx = next((i for i, line in enumerate(lines) if line == title), None)
            if title_idx is not None:
                for j in range(title_idx + 1, min(title_idx + 5, len(lines))):
                    author_line = lines[j]
                    if (',' in author_line or ' and ' in author_line.lower() or 
                        len(author_line.split()) > 2):
                        authors = author_line
                        break
        
        return {"title": title, "authors": authors}


class ReadmeLoader:
    """Service for loading README files from GitHub."""
    
    def __init__(self, timeout: int = 10):
        """
        Initialize README loader.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
    
    def _convert_to_raw_url(self, readme_url: str) -> str:
        """
        Convert regular GitHub URL to raw URL.
        
        Args:
            readme_url: GitHub URL
            
        Returns:
            Raw GitHub URL
        """
        if "github.com" in readme_url and "raw.githubusercontent.com" not in readme_url:
            # Pattern: https://github.com/user/repo/blob/branch/path
            pattern = r'https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)'
            match = re.match(pattern, readme_url)
            if match:
                user, repo, branch, path = match.groups()
                return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
            else:
                # Try to extract from URL and assume main branch
                pattern = r'https://github\.com/([^/]+)/([^/]+)'
                match = re.match(pattern, readme_url)
                if match:
                    user, repo = match.groups()
                    return f"https://raw.githubusercontent.com/{user}/{repo}/main/README.md"
        
        return readme_url
    
    def load(self, readme_url: str) -> Optional[str]:
        """
        Fetch content from a GitHub README URL.
        
        Args:
            readme_url: GitHub URL
            
        Returns:
            README content as string, or None if failed
        """
        raw_url = self._convert_to_raw_url(readme_url)
        
        try:
            response = requests.get(raw_url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching README from {readme_url}: {e}")
            return None

