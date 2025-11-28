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

