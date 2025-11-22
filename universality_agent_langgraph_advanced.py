# Disable multiprocessing to avoid semaphore leaks - MUST be set before any imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"

# Configure multiprocessing to use spawn method (better for cleanup)
#import multiprocessing
#if hasattr(multiprocessing, 'set_start_method'):
#    try:
#        multiprocessing.set_start_method('spawn', force=True)
#    except RuntimeError:
#        pass  # Already set

import json
import faiss
import numpy as np
import requests
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import atexit

from pathlib import Path
from langgraph.graph import StateGraph
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer, CrossEncoder

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# ---------- LLM: LOCAL via OLLAMA ----------

from langchain_ollama import ChatOllama

# Set environment variables to avoid connection issues
os.environ.setdefault("OLLAMA_HOST", "localhost:11434")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

# Check if Ollama is running before initializing
def check_ollama_connection():
    """Verify Ollama is accessible before initializing ChatOllama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error: Cannot connect to Ollama at http://localhost:11434")
        print(f"Please ensure Ollama is running. Start it with: ollama serve")
        print(f"Or check if Ollama is running on a different port.")
        raise ConnectionError(
            "Ollama connection refused. Please ensure Ollama is running.\n"
            "Start Ollama with: ollama serve\n"
            "Or check if it's running on a different port."
        ) from e

# Verify Ollama is accessible
check_ollama_connection()

# Fallback function to call Ollama API directly via requests
def call_ollama_direct(prompt: str, model: str = "llama3.1") -> str:
    """Direct API call to Ollama as fallback if ChatOllama fails"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
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

# Initialize ChatOllama with explicit base_url and connection settings
try:
    llm = ChatOllama(
        model="llama3.1", 
        temperature=0,
        base_url="http://localhost:11434",  # Explicit base URL
        timeout=60.0,  # Set explicit timeout
        num_ctx=4096,  # Context window size
    )
    print("ChatOllama initialized successfully")
except Exception as e:
    print(f"Error initializing ChatOllama: {e}")
    print("Trying with minimal configuration...")
    try:
        # Fallback to minimal configuration
        llm = ChatOllama(
            model="llama3.1", 
            temperature=0,
        )
        print("ChatOllama initialized with minimal configuration")
    except Exception as e2:
        print(f"ChatOllama initialization failed: {e2}")
        print("Will use direct API calls as fallback")
        llm = None  # Will use direct API calls

# Improved embedding model - larger and more capable for better semantic understanding
# Using all-mpnet-base-v2 which is better for semantic similarity tasks
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",#all-mpnet-base-v2", 
    device='cpu'
)
print("Using all-mpnet-base-v2 for embeddings (better semantic understanding)")

# Cross-encoder for reranking - provides better relevance scoring
#try:
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#print("Using cross-encoder for reranking")
#except:
#reranker = None
print("Cross-encoder not available, skipping reranking")

# Cleanup function to properly close resources
def cleanup_models():
    """Clean up model resources to prevent semaphore leaks"""
    global embedder, reranker
    try:
        if embedder is not None:
            # Clear any cached data
            if hasattr(embedder, 'eval'):
                embedder.eval()
    except:
        pass
    try:
        if reranker is not None:
            if hasattr(reranker, 'eval'):
                reranker.eval()
    except:
        pass

# Register cleanup function
atexit.register(cleanup_models)

metrics_json_path = str(Path(__file__).parent / "config/future_ai_metrics.json")


# ------------------------- LOAD JSON FILES -----------------------------
def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

# ------------------------- EXTRACT METRICS FROM JSON -----------------------------
import json

def load_evaluation_criteria(json_path: str):
    """
    Load evaluation criteria from JSON and convert them into evaluation questions
    usable by the LangGraph+Ollama pipeline.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    questions = []
    structured_items = []  # keeps the hierarchy to include in report

    for section in data["future_ai_metrics"]:
        section_name = section["name"]
        section_desc = section["description"]

        for criterion in section["evaluation_criteria"]:
            crit_name = criterion["name"]
            crit_desc = criterion["description"]

            for metric in criterion["metrics"]:
                metric_name = metric["name"]
                metric_desc = metric["description"]

                question_text = (
                    f"Metric: {metric_name}. "
                    f"Does the paper address the following? {metric_desc}"
                )

                questions.append(question_text)

                structured_items.append({
                    "section": section_name,
                    "criterion": crit_name,
                    "metric": metric_name,
                    "question": question_text,
                    "description": metric_desc,
                })

    return questions, structured_items


# ------------------------- LOAD PDF FILES -----------------------------
def is_bibliography_page(page_content: str) -> bool:
    """
    Detect if a page is a bibliography/references page.
    
    Args:
        page_content: The text content of the page
        
    Returns:
        True if the page appears to be a bibliography page
    """
    # Check first 500 characters for bibliography section headers
    header_text = page_content[:500].lower()
    
    # Common bibliography section headers
    bibliography_keywords = [
        r'\breferences\b',
        r'\bbibliography\b',
        r'\bcitations\b',
        r'\breference\s+list\b',
        r'\bworks\s+cited\b',
        r'\bliterature\s+cited\b'
    ]
    
    # Check for bibliography headers
    for pattern in bibliography_keywords:
        if re.search(pattern, header_text, re.IGNORECASE):
            return True
    
    # Check for common citation patterns that dominate bibliography pages
    # Pattern 1: Many lines starting with author names (Last, First or First Last)
    lines = page_content.split('\n')[:30]  # Check first 30 lines
    citation_pattern_count = 0
    non_empty_lines = [l for l in lines if l.strip()]
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Pattern: Author names followed by year (e.g., "Smith, J. (2020)" or "Smith J, 2020")
        if re.search(r'^[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)?,?\s+\(?\d{4}\)?', line_stripped):
            citation_pattern_count += 1
        # Pattern: Numbered citations (e.g., "[1] Author, Title")
        elif re.search(r'^\[\d+\]', line_stripped):
            citation_pattern_count += 1
        # Pattern: Author et al. patterns
        elif re.search(r'[A-Z][a-z]+\s+et\s+al\.', line_stripped, re.IGNORECASE):
            citation_pattern_count += 1
    
    # If more than 40% of non-empty lines look like citations, it's likely a bibliography page
    if len(non_empty_lines) > 5 and citation_pattern_count > len(non_empty_lines) * 0.4:
        return True
    
    return False

def load_pdf(path):
    """
    Load PDF and filter out bibliography pages.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        List of page objects (bibliography pages excluded)
    """
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    
    # Filter out bibliography pages
    filtered_pages = []
    bibliography_pages_count = 0
    
    for page in pages:
        if is_bibliography_page(page.page_content):
            bibliography_pages_count += 1
            continue  # Skip bibliography pages
        filtered_pages.append(page)
    
    if bibliography_pages_count > 0:
        print(f"Excluded {bibliography_pages_count} bibliography page(s) from processing")
    
    return filtered_pages

# ------------------------- LOAD GITHUB README -----------------------------
def load_github_readme(readme_url: str) -> str:
    """
    Fetch content from a GitHub README URL.
    Handles both regular GitHub URLs and raw GitHub URLs.
    
    Args:
        readme_url: GitHub URL (e.g., https://github.com/user/repo/blob/main/README.md)
                   or raw URL (e.g., https://raw.githubusercontent.com/user/repo/main/README.md)
    
    Returns:
        README content as string
    """
    # Convert regular GitHub URL to raw URL if needed
    if "github.com" in readme_url and "raw.githubusercontent.com" not in readme_url:
        # Pattern: https://github.com/user/repo/blob/branch/path
        pattern = r'https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)'
        match = re.match(pattern, readme_url)
        if match:
            user, repo, branch, path = match.groups()
            readme_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
        else:
            # Try to extract from URL and assume main branch
            pattern = r'https://github\.com/([^/]+)/([^/]+)'
            match = re.match(pattern, readme_url)
            if match:
                user, repo = match.groups()
                readme_url = f"https://raw.githubusercontent.com/{user}/{repo}/main/README.md"
    
    try:
        response = requests.get(readme_url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching README from {readme_url}: {e}")
        return ""

# ------------------------- IMPROVED CHUNKING STRATEGIES -----------------------------

def semantic_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into semantic chunks with overlap for better context preservation.
    Uses sentence boundaries when possible.
    """
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size and current_chunk:
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

def extract_document_structure(page_content: str, page_num: int) -> Dict[str, Any]:
    """
    Extract document structure information (sections, headings, etc.)
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

def chunk_with_semantic_context(page, chunk_size=500, overlap=100):
    """
    Improved chunking that preserves semantic context and document structure.
    
    Args:
        page: Page object from PDF loader
        chunk_size: Size of chunks in words (default: 500)
        overlap: Overlap between chunks in words (default: 100)
    """
    page_content = page.page_content
    page_num = page.metadata.get("page", 0)
    
    # Extract document structure
    structure = extract_document_structure(page_content, page_num)
    
    # Use semantic chunking instead of line-by-line
    text_chunks = semantic_chunk_text(page_content, chunk_size=chunk_size, overlap=overlap)
    
    chunks = []
    line_offset = 0
    
    for chunk_idx, chunk_text in enumerate(text_chunks):
        # Estimate line numbers (approximate)
        lines_before = page_content[:page_content.find(chunk_text)].count('\n')
        
        chunks.append({
            "text": chunk_text,
            "page": page_num,
            "chunk_index": chunk_idx,
            "line_start": lines_before + 1,
            "line_end": lines_before + chunk_text.count('\n') + 1,
            "source": "pdf",
            "structure": structure,
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split())
        })
    
    return chunks

# ------------------------- CHUNK README FILES -----------------------------
def chunk_readme(readme_content: str, chunk_size=400, overlap=80):
    """
    Improved README chunking with semantic context preservation.
    
    Args:
        readme_content: README content as string
        chunk_size: Size of chunks in words (default: 400)
        overlap: Overlap between chunks in words (default: 80)
    
    Returns:
        List of chunk dictionaries
    """
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
    text_chunks = semantic_chunk_text(readme_content, chunk_size=chunk_size, overlap=overlap)
    
    chunks = []
    for chunk_idx, chunk_text in enumerate(text_chunks):
        lines_before = readme_content[:readme_content.find(chunk_text)].count('\n')
        chunks.append({
            "text": chunk_text,
            "page": "README",
            "chunk_index": chunk_idx,
            "line_start": lines_before + 1,
            "line_end": lines_before + chunk_text.count('\n') + 1,
            "source": "readme",
            "structure": structure,
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split())
        })
    
    return chunks

# ------------------------- BUILD INDEX -----------------------------
def truncate_text_for_embedding(text: str, max_chars: int = 2000) -> str:
    """
    Truncate text to a safe length for embedding models.
    Most sentence transformers have a 512 token limit (~2000 characters).
    
    Args:
        text: Text to truncate
        max_chars: Maximum characters to keep (default: 2000)
    
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    # Truncate at word boundary to avoid cutting words
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:  # Only use word boundary if it's not too far back
        return truncated[:last_space] + "..."
    return truncated + "..."

def build_index(chunks):
    """
    Build FAISS index incrementally in batches to avoid memory issues and segmentation faults.
    Processes chunks in smaller batches to prevent overwhelming memory.
    """
    if not chunks:
        raise ValueError("No chunks provided to build index")
    
    print(f"Building index for {len(chunks)} chunks...")
    
    # Determine embedding dimension from first chunk
    sample_text = chunks[0]["text"] if chunks else "sample"
    # Truncate sample text to avoid issues with very long chunks
    sample_text = truncate_text_for_embedding(sample_text)
    sample_vector = embedder.encode(
        [sample_text],
        show_progress_bar=False,
        batch_size=1,
        convert_to_numpy=True,
    )
    embedding_dim = sample_vector.shape[1]
    
    # Initialize FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Process chunks in batches to avoid memory issues
    batch_size = 50  # Smaller batches to prevent memory overflow
    all_vectors = []
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        # Truncate each text to safe length before encoding
        batch_texts = [truncate_text_for_embedding(ch["text"]) for ch in batch_chunks]
        
        print(f"  Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch_chunks)} chunks)...")
        
        # Encode batch
        batch_vectors = embedder.encode(
            batch_texts,
            show_progress_bar=False,
            batch_size=min(16, len(batch_texts)),  # Smaller internal batch size
            convert_to_numpy=True,
        )
        
        # Add to index incrementally
        index.add(batch_vectors.astype('float32'))
        all_vectors.append(batch_vectors)
    
    # Concatenate all vectors for return
    vectors = np.vstack(all_vectors) if len(all_vectors) > 1 else all_vectors[0]
    
    print(f"Index built successfully with {index.ntotal} vectors")
    return index, vectors

def retrieve_node(state):
    query = state["current_question"]
    # Truncate query if needed (though queries should be short)
    query = truncate_text_for_embedding(query)
    # Explicitly disable multiprocessing in encode call
    qvec = embedder.encode(
        [query],
        show_progress_bar=False,
        batch_size=1,
        convert_to_numpy=True,
        #num_workers=0  # Disable multiprocessing
    )
    D, I = state["index"].search(qvec, k=3) #k=5
    state["retrieved_chunks"] = [state["chunks"][i] for i in I[0]]
    return state    



def evaluate_node(state):
    question = state["current_question"]
    context_parts = []
    for c in state["retrieved_chunks"]:
        source_label = c.get("source", "pdf")
        if source_label == "readme":
            context_parts.append(f"[README|l={c['line_start']}-{c['line_end']}]: {c['text']}")
        else:
            context_parts.append(f"[p={c['page']}|l={c['line_start']}-{c['line_end']}]: {c['text']}")
    context = "\n".join(context_parts)

    prompt = f"""
You are assessing whether a research paper and its associated GitHub README address specific questions.

QUESTION:
{question}

CONTEXT (segments with page/line identifiers from PDF, or line identifiers from README):
{context}

TASK:
1. Does the paper or README address the question? (yes/no)
2. Provide page and line numbers of evidence (or README line numbers).
3. Quote the relevant segments.
4. Indicate if evidence comes from the paper (PDF) or the README.


Return ONLY valid JSON with the following structure:
{{
    "addressed": "yes/no",
    "evidence": [
    {{
        "page": <page number or "README">,
        "line": <line number>,
        "quote": "<text>",
        "source": "pdf" or "readme"
    }}]
    }}
"""

    try:
        if llm is not None:
            result = llm.invoke(prompt)
            state["evaluation"] = result
        else:
            # Use direct API call if ChatOllama is not available
            response_text = call_ollama_direct(prompt)
            state["evaluation"] = {"content": response_text}
    except Exception as e:
        error_msg = str(e)
        if "50222" in error_msg or "ECONNREFUSED" in error_msg:
            print(f"Connection error detected (port 50222): {e}")
            print("Attempting to use direct Ollama API call as fallback...")
            try:
                # Use direct API call as fallback
                response_text = call_ollama_direct(prompt)
                state["evaluation"] = {"content": response_text}
                print("Successfully used direct API call")
            except Exception as fallback_error:
                print(f"Direct API fallback also failed: {fallback_error}")
                # Return a fallback response
                state["evaluation"] = {
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
                response_text = call_ollama_direct(prompt)
                state["evaluation"] = {"content": response_text}
            except Exception as fallback_error:
                print(f"Direct API fallback failed: {fallback_error}")
                raise
    
    return state


def append_report_node(state):
    state["report"].append({
        "question": state["current_question"],
        "analysis": state["evaluation"]
    })
    return state


def generate_pdf_report(report, output_path="assessment_report.pdf"):
    """
    Create a PDF assessment report from the LangGraph+Ollama evaluation results.
    """
    styles = getSampleStyleSheet()
    style_title = styles["Title"]
    style_h1 = styles["Heading1"]
    style_h2 = styles["Heading2"]
    style_h3 = styles["Heading3"]
    style_normal = styles["BodyText"]

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    story = []

    # Title
    story.append(Paragraph("Research Paper Compliance Assessment", style_title))
    story.append(Spacer(1, 0.5*cm))

    current_section = None
    current_criterion = None

    for item in report:
        # Section title
        if item["section"] != current_section:
            current_section = item["section"]
            story.append(Paragraph(item["section"][0].upper() + item["section"][1:].lower(), style_h1))
            story.append(Spacer(1, 0.3*cm))

        # Criterion subtitle
        if item["criterion"] != current_criterion:
            current_criterion = item["criterion"]
            story.append(Paragraph(item["criterion"][0].upper() + item["criterion"][1:].lower(), style_h2))
            story.append(Spacer(1, 0.2*cm))

        # Metric Name
        metric_name = item['metric'][0].upper() + item['metric'][1:].lower()
        story.append(Paragraph(f"Metric: <b>{metric_name}</b>", style_h3))
        story.append(Spacer(1, 0.2*cm))

        # Metric Description
        story.append(Paragraph(item["description"], style_normal))
        story.append(Spacer(1, 0.2*cm))

        # Parse LLM response - handle both string and object responses
        analysis_text = item["analysis"]
        if hasattr(analysis_text, 'content'):
            analysis_text = analysis_text.content
        elif hasattr(analysis_text, 'text'):
            analysis_text = analysis_text.text
        elif not isinstance(analysis_text, str):
            analysis_text = str(analysis_text)
        
        # Try to parse as JSON, fallback to raw text if parsing fails
        try:
            analysis = json.loads(analysis_text)
        except (json.JSONDecodeError, TypeError):
            # If JSON parsing fails, create a simple structure
            analysis = {
                "addressed": "unknown",
                "evidence": [],
                "raw_response": analysis_text
            }

        # Addressed?
        story.append(Paragraph(f"<b>Addressed:</b> {analysis.get('addressed', 'unknown')}", style_normal))
        story.append(Spacer(1, 0.2*cm))

        # Evidence
        if "evidence" in analysis and analysis["evidence"]:
            story.append(Paragraph("<b>Evidence:</b>", style_normal))
            for ev in analysis["evidence"]:
                source = ev.get('source', 'pdf')
                page_label = "README" if source == "readme" else f"Page {ev.get('page')}"
                txt = (
                    f"{page_label}, Line {ev.get('line')} ({source})<br/>"
                    f"<i>{ev.get('quote')}</i>"
                )
                story.append(Paragraph(txt, style_normal))
                story.append(Spacer(1, 0.2*cm))
        else:
            story.append(Paragraph("No evidence found.", style_normal))

        story.append(Spacer(1, 0.4*cm))

    doc.build(story)
    return output_path

workflow = StateGraph(
    state_schema=dict,
)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("append", append_report_node)
workflow.add_edge("retrieve", "evaluate")
workflow.add_edge("evaluate", "append")
workflow.set_entry_point("retrieve")

graph = workflow.compile()

def analyze_paper(pdf_path, questions, structured_items, readme_url=None, 
                  pdf_chunk_size=500, pdf_overlap=100, readme_chunk_size=400, readme_overlap=80):
    """
    Analyze a paper against a list of metrics.
    
    Args:
        pdf_path: Path to the PDF file
        questions: List of evaluation questions
        structured_items: List of structured metric items
        readme_url: Optional GitHub README URL to include in assessment
        pdf_chunk_size: Size of chunks for PDF pages (default: 500 words)
        pdf_overlap: Overlap between PDF chunks (default: 100 words)
        readme_chunk_size: Size of chunks for README (default: 400 words)
        readme_overlap: Overlap between README chunks (default: 80 words)
    """    
    pages = load_pdf(pdf_path)
    chunks = []
    
    # Chunk PDF pages with configurable sizes
    for page in pages:
        chunks.extend(chunk_with_semantic_context(page, chunk_size=pdf_chunk_size, overlap=pdf_overlap))
    
    # Load and chunk README if URL is provided
    if readme_url:
        print(f"Fetching README from: {readme_url}")
        readme_content = load_github_readme(readme_url)
        if readme_content:
            readme_chunks = chunk_readme(readme_content, chunk_size=readme_chunk_size, overlap=readme_overlap)
            chunks.extend(readme_chunks)
            print(f"Added {len(readme_chunks)} README chunks to assessment")
        else:
            print("Warning: Could not fetch README content")

    # Warn if too many chunks (could cause memory issues)
    if len(chunks) > 2000:
        print(f"Warning: {len(chunks)} chunks detected. This may cause memory issues.")
        print(f"Consider increasing chunk sizes (e.g., pdf_chunk_size=1000, readme_chunk_size=800)")
    else:
        print(f"Total chunks created: {len(chunks)}")

    index, vectors = build_index(chunks)

    report = []

    # Run LangGraph evaluation for each metric-based question
    for q, meta in zip(questions, structured_items):
        output = graph.invoke({
            "current_question": q,
            "chunks": chunks,
            "index": index,
            "report": report,
        })

        meta_out = {
            "section": meta["section"],
            "criterion": meta["criterion"],
            "metric": meta["metric"],
            "question": meta["question"],
            "description": meta["description"],
            "analysis": output["report"][-1]["analysis"],
        }

        report[-1] = meta_out  # replace raw output with structured metadata

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Generate PDF
    pdf_path = generate_pdf_report(report, "results/assessment_report.pdf")
    print("PDF saved at:", pdf_path)

    return report

if __name__ == "__main__":

    # Load metrics from JSON file
    questions, structured_items = load_evaluation_criteria(metrics_json_path)
    
    # Analyze paper
    pdf_path = "./papers/s41597-025-04707-4.pdf"
    readme_url = "https://github.com/LidiaGarrucho/MAMA-MIA/blob/main/README.md"  # Set to None if not using README
    
    report = analyze_paper(pdf_path, questions, structured_items, readme_url=readme_url)
    
    # Print summary
    print("Task completed. Report saved to assessment_report.pdf")
