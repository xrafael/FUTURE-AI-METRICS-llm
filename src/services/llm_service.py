"""LLM service for interacting with Ollama."""
import json
import os
import requests
from typing import Dict, Any, Optional

from langchain_ollama import ChatOllama
from config.settings import OllamaConfig


class LLMService:
    """Service for LLM interactions via Ollama."""
    
    def __init__(self, config: OllamaConfig):
        """
        Initialize LLM service.
        
        Args:
            config: Ollama configuration
        """
        self.config = config
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize ChatOllama with fallback options."""
        # Set environment variables
        os.environ.setdefault("OLLAMA_HOST", self.config.host)
        os.environ.setdefault("OLLAMA_BASE_URL", self.config.base_url)
        
        # Verify Ollama is accessible
        if not self._check_connection():
            raise ConnectionError(
                "Ollama connection refused. Please ensure Ollama is running.\n"
                "Start Ollama with: ollama serve\n"
                "Or check if it's running on a different port."
            )
        
        try:
            self.llm = ChatOllama(
                model=self.config.model,
                temperature=self.config.temperature,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                num_ctx=self.config.num_ctx,
            )
            print("ChatOllama initialized successfully")
        except Exception as e:
            print(f"Error initializing ChatOllama: {e}")
            print("Trying with minimal configuration...")
            try:
                self.llm = ChatOllama(
                    model=self.config.model,
                    temperature=self.config.temperature,
                )
                print("ChatOllama initialized with minimal configuration")
            except Exception as e2:
                print(f"ChatOllama initialization failed: {e2}")
                print("Will use direct API calls as fallback")
                self.llm = None
    
    def _check_connection(self) -> bool:
        """
        Verify Ollama is accessible.
        
        Returns:
            True if accessible, False otherwise
        """
        try:
            response = requests.get(
                f"http://{self.config.host}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"Error: Cannot connect to Ollama at http://{self.config.host}")
            print(f"Please ensure Ollama is running. Start it with: ollama serve")
            return False
    
    def _call_ollama_direct(self, prompt: str) -> str:
        """
        Direct API call to Ollama as fallback.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Response text
        """
        try:
            response = requests.post(
                f"http://{self.config.host}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            print(f"Direct Ollama API call also failed: {e}")
            raise
    
    def invoke(self, prompt: str) -> Dict[str, Any]:
        """
        Invoke LLM with a prompt.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Response dictionary with 'content' key
        """
        try:
            if self.llm is not None:
                result = self.llm.invoke(prompt)
                return result
            else:
                # Use direct API call if ChatOllama is not available
                response_text = self._call_ollama_direct(prompt)
                return {"content": response_text}
        except Exception as e:
            error_msg = str(e)
            if "50222" in error_msg or "ECONNREFUSED" in error_msg:
                print(f"Connection error detected: {e}")
                print("Attempting to use direct Ollama API call as fallback...")
                try:
                    response_text = self._call_ollama_direct(prompt)
                    print("Successfully used direct API call")
                    return {"content": response_text}
                except Exception as fallback_error:
                    print(f"Direct API fallback also failed: {fallback_error}")
                    return {
                        "content": json.dumps({
                            "addressed": "unknown",
                            "evidence": [],
                            "error": f"Connection error: {error_msg}"
                        })
                    }
            else:
                # For other errors, try direct API call as fallback
                print(f"Error with ChatOllama: {e}")
                print("Attempting direct API call...")
                try:
                    response_text = self._call_ollama_direct(prompt)
                    return {"content": response_text}
                except Exception as fallback_error:
                    print(f"Direct API fallback failed: {fallback_error}")
                    raise

