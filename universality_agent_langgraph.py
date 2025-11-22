import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"


import json
import faiss
import numpy as np
import requests
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

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
llm = ChatOllama(model="llama3.1", temperature=0)

# Improved embedding model - larger and more capable for better semantic understanding
# Using all-mpnet-base-v2 which is better for semantic similarity tasks
try:
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("Using all-mpnet-base-v2 for embeddings (better semantic understanding)")
except:
    # Fallback to smaller model if the larger one isn't available
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Using all-MiniLM-L6-v2 for embeddings (fallback)")

# Cross-encoder for reranking - provides better relevance scoring
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Using cross-encoder for reranking")
except:
    reranker = None
    print("Cross-encoder not available, skipping reranking")

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
def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages

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

def chunk_with_semantic_context(page):
    """
    Improved chunking that preserves semantic context and document structure.
    """
    page_content = page.page_content
    page_num = page.metadata.get("page", 0)
    
    # Extract document structure
    structure = extract_document_structure(page_content, page_num)
    
    # Use semantic chunking instead of line-by-line
    text_chunks = semantic_chunk_text(page_content, chunk_size=500, overlap=100)
    
    chunks = []
    current_pos = 0
    
    for chunk_idx, chunk_text in enumerate(text_chunks):
        # Find the position of this chunk in the original text
        # Use a more robust method: find the chunk starting from current position
        chunk_start_pos = page_content.find(chunk_text, current_pos)
        if chunk_start_pos == -1:
            # Fallback: try finding from beginning
            chunk_start_pos = page_content.find(chunk_text)
        
        # Calculate line numbers
        lines_before = page_content[:chunk_start_pos].count('\n')
        lines_in_chunk = chunk_text.count('\n')
        
        chunks.append({
            "text": chunk_text,
            "page": page_num,
            "chunk_index": chunk_idx,
            "line_start": lines_before + 1,
            "line_end": lines_before + lines_in_chunk + 1,
            "source": "pdf",
            "structure": structure,
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split())
        })
        
        # Update position for next search (with some overlap tolerance)
        current_pos = chunk_start_pos + len(chunk_text) - 50  # Allow some overlap
    
    return chunks

# ------------------------- CHUNK README FILES -----------------------------
def chunk_readme(readme_content: str):
    """
    Improved README chunking with semantic context preservation.
    
    Args:
        readme_content: README content as string
    
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
    text_chunks = semantic_chunk_text(readme_content, chunk_size=400, overlap=80)
    
    chunks = []
    current_pos = 0
    
    for chunk_idx, chunk_text in enumerate(text_chunks):
        # Find position more robustly
        chunk_start_pos = readme_content.find(chunk_text, current_pos)
        if chunk_start_pos == -1:
            chunk_start_pos = readme_content.find(chunk_text)
        
        lines_before = readme_content[:chunk_start_pos].count('\n')
        lines_in_chunk = chunk_text.count('\n')
        
        chunks.append({
            "text": chunk_text,
            "page": "README",
            "chunk_index": chunk_idx,
            "line_start": lines_before + 1,
            "line_end": lines_before + lines_in_chunk + 1,
            "source": "readme",
            "structure": structure,
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split())
        })
        
        current_pos = chunk_start_pos + len(chunk_text) - 40  # Allow overlap
    
    return chunks

# ------------------------- BUILD INDEX -----------------------------
def build_index(chunks):
    """
    Build FAISS index with improved normalization for better similarity search.
    """
    texts = [ch["text"] for ch in chunks]
    vectors = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])  # Inner product for normalized vectors
    index.add(vectors.astype('float32'))
    return index, vectors

def expand_query(query: str, metric_context: Dict[str, str] = None) -> List[str]:
    """
    Expand query with related terms and concepts for better retrieval.
    """
    expanded_queries = [query]
    
    # Extract key terms from the query
    key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    key_terms.extend(re.findall(r'\b\w{4,}\b', query.lower()))  # Longer words
    
    # Add metric context if available
    if metric_context:
        if "description" in metric_context:
            expanded_queries.append(metric_context["description"])
        if "criterion" in metric_context:
            expanded_queries.append(metric_context["criterion"])
    
    # Create variations
    for term in key_terms[:3]:  # Top 3 terms
        if len(term) > 4:
            expanded_queries.append(term)
    
    return expanded_queries

def retrieve_node(state):
    """
    Improved retrieval with query expansion and multi-stage search.
    """
    query = state["current_question"]
    metric_context = state.get("metric_context", {})
    
    # Expand query for better coverage
    expanded_queries = expand_query(query, metric_context)
    
    # Multi-query retrieval - search with each expanded query
    all_candidates = []
    candidate_scores = defaultdict(float)
    
    for eq in expanded_queries:
        qvec = embedder.encode([eq], normalize_embeddings=True)
        D, I = state["index"].search(qvec.astype('float32'), k=10)  # Get more candidates
        
        # Aggregate scores across queries
        for score, idx in zip(D[0], I[0]):
            candidate_scores[idx] = max(candidate_scores[idx], float(score))
            if idx not in [c["original_index"] for c in all_candidates]:
                chunk = state["chunks"][idx].copy()
                chunk["original_index"] = idx
                chunk["retrieval_score"] = float(score)
                all_candidates.append(chunk)
    
    # Sort by aggregated scores
    all_candidates.sort(key=lambda x: x["retrieval_score"], reverse=True)
    
    # Rerank with cross-encoder if available
    if reranker and len(all_candidates) > 5:
        top_candidates = all_candidates[:20]  # Rerank top 20
        pairs = [[query, c["text"]] for c in top_candidates]
        rerank_scores = reranker.predict(pairs)
        
        # Update scores with reranking
        for i, score in enumerate(rerank_scores):
            top_candidates[i]["rerank_score"] = float(score)
        
        # Resort by rerank score
        top_candidates.sort(key=lambda x: x.get("rerank_score", x["retrieval_score"]), reverse=True)
        state["retrieved_chunks"] = top_candidates[:10]  # Top 10 after reranking
    else:
        state["retrieved_chunks"] = all_candidates[:10]  # Top 10 without reranking
    
    return state    



def evaluate_node(state):
    """
    Enhanced evaluation with chain-of-thought reasoning and deeper analysis.
    """
    question = state["current_question"]
    metric_context = state.get("metric_context", {})
    
    # Build rich context with structure information
    context_parts = []
    pdf_evidence = []
    readme_evidence = []
    
    for c in state["retrieved_chunks"]:
        source_label = c.get("source", "pdf")
        line_info = f"lines {c.get('line_start', '?')}-{c.get('line_end', '?')}"
        
        # Include structure information if available
        structure_info = ""
        if "structure" in c:
            struct = c["structure"]
            if struct.get("sections"):
                section_names = [s.get("heading", "")[:50] for s in struct["sections"][:2]]
                if section_names:
                    structure_info = f" [Sections: {', '.join(section_names)}]"
            if struct.get("has_table"):
                structure_info += " [Contains table]"
            if struct.get("has_figure"):
                structure_info += " [References figures]"
        
        chunk_info = f"[{source_label.upper()}|p={c.get('page', '?')}|{line_info}]{structure_info}: {c['text']}"
        context_parts.append(chunk_info)
        
        # Separate by source for cross-document analysis
        if source_label == "readme":
            readme_evidence.append(c)
        else:
            pdf_evidence.append(c)
    
    context = "\n".join(context_parts)
    
    # Enhanced prompt with chain-of-thought reasoning
    metric_name = metric_context.get("metric", "")
    criterion = metric_context.get("criterion", "")
    
    prompt = f"""You are an expert assessor evaluating whether a research paper and its GitHub README address specific metrics for medical AI systems.

METRIC CONTEXT:
- Metric: {metric_name}
- Criterion: {criterion}
- Question: {question}

CONTEXT FROM DOCUMENTS:
{context}

ANALYSIS TASK - Use chain-of-thought reasoning:

Step 1: UNDERSTANDING
- What is the metric asking about? Break down the key requirements.
- What type of information would satisfy this metric?

Step 2: EVIDENCE IDENTIFICATION
- Review each context segment carefully.
- Identify which segments directly or indirectly address the metric.
- Consider both explicit statements and implicit information.
- Look for related concepts, synonyms, or alternative phrasings.

Step 3: CROSS-DOCUMENT ANALYSIS
- Compare information from PDF ({len(pdf_evidence)} segments) and README ({len(readme_evidence)} segments).
- Do they complement each other? Are there contradictions?
- Which source provides more relevant information?

Step 4: ASSESSMENT
- Does the paper/README adequately address this metric? (yes/partial/no)
- If partial: what aspects are covered and what's missing?
- Provide specific evidence with exact quotes.

Step 5: CONFIDENCE
- How confident are you in this assessment? (high/medium/low)
- Why? What factors affect your confidence?

Return ONLY valid JSON with this structure:
{{
    "addressed": "yes" | "partial" | "no",
    "confidence": "high" | "medium" | "low",
    "reasoning": "<brief explanation of your assessment>",
    "evidence": [
        {{
            "page": <page number or "README">,
            "line_start": <start line>,
            "line_end": <end line>,
            "quote": "<exact quote from document>",
            "source": "pdf" | "readme",
            "relevance": "direct" | "indirect" | "related",
            "explanation": "<why this evidence is relevant>"
        }}
    ],
    "gaps": ["<what information is missing, if any>"],
    "cross_document_analysis": "<how PDF and README information relates>"
}}

IMPORTANT: Be thorough. Look for implicit information, related concepts, and technical details that may address the metric even if not explicitly stated."""

    result = llm.invoke(prompt)
    state["evaluation"] = result
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
        story.append(Paragraph(f"Metric: <b>{item['metric'][0].upper() + item["metric"][1:].lower()}</b>", style_h3))
        story.append(Spacer(1, 0.2*cm))

        # Metric Description
        story.append(Paragraph(item["description"], style_normal))
        story.append(Spacer(1, 0.2*cm))

        # Parse analysis (handle both string and object responses)
        try:
            if hasattr(item["analysis"], 'content'):
                analysis = json.loads(item["analysis"].content)
            else:
                analysis = json.loads(item["analysis"]) if isinstance(item["analysis"], str) else item["analysis"]
        except:
            analysis = {"addressed": "unknown", "evidence": []}

        # Addressed with confidence
        addressed = analysis.get('addressed', 'unknown')
        confidence = analysis.get('confidence', 'unknown')
        story.append(Paragraph(f"<b>Addressed:</b> {addressed} <i>(Confidence: {confidence})</i>", style_normal))
        story.append(Spacer(1, 0.2*cm))
        
        # Reasoning if available
        if "reasoning" in analysis:
            story.append(Paragraph(f"<b>Reasoning:</b> {analysis['reasoning']}", style_normal))
            story.append(Spacer(1, 0.2*cm))

        # Evidence with enhanced information
        if "evidence" in analysis and analysis["evidence"]:
            story.append(Paragraph("<b>Evidence:</b>", style_normal))
            for ev in analysis["evidence"]:
                source = ev.get('source', 'pdf')
                page_label = "README" if source == "readme" else f"Page {ev.get('page', ev.get('page', '?'))}"
                line_info = f"Lines {ev.get('line_start', ev.get('line', '?'))}"
                if ev.get('line_end'):
                    line_info = f"Lines {ev.get('line_start')}-{ev.get('line_end')}"
                
                relevance = ev.get('relevance', '')
                relevance_text = f" [{relevance}]" if relevance else ""
                
                txt = (
                    f"{page_label}, {line_info} ({source}){relevance_text}<br/>"
                    f"<i>{ev.get('quote', '')}</i>"
                )
                if ev.get('explanation'):
                    txt += f"<br/><b>Why relevant:</b> {ev.get('explanation')}"
                
                story.append(Paragraph(txt, style_normal))
                story.append(Spacer(1, 0.2*cm))
        else:
            story.append(Paragraph("No evidence found.", style_normal))
        
        # Gaps if identified
        if "gaps" in analysis and analysis["gaps"]:
            gaps_text = "; ".join(analysis["gaps"])
            story.append(Paragraph(f"<b>Information Gaps:</b> {gaps_text}", style_normal))
            story.append(Spacer(1, 0.2*cm))
        
        # Cross-document analysis if available
        if "cross_document_analysis" in analysis and analysis["cross_document_analysis"]:
            story.append(Paragraph(f"<b>Cross-Document Analysis:</b> {analysis['cross_document_analysis']}", style_normal))
            story.append(Spacer(1, 0.2*cm))

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

def analyze_paper(pdf_path, questions, structured_items, readme_url=None):
    """
    Analyze a paper against a list of metrics.
    
    Args:
        pdf_path: Path to the PDF file
        questions: List of evaluation questions
        structured_items: List of structured metric items
        readme_url: Optional GitHub README URL to include in assessment
    """    
    pages = load_pdf(pdf_path)
    chunks = []
    for page in pages:
        chunks.extend(chunk_with_semantic_context(page))  # Use improved chunking
    
    # Load and chunk README if URL is provided
    if readme_url:
        print(f"Fetching README from: {readme_url}")
        readme_content = load_github_readme(readme_url)
        if readme_content:
            readme_chunks = chunk_readme(readme_content)
            chunks.extend(readme_chunks)
            print(f"Added {len(readme_chunks)} README chunks to assessment")
        else:
            print("Warning: Could not fetch README content")

    index, vectors = build_index(chunks)

    report = []

    # Run LangGraph evaluation for each metric-based question
    for q, meta in zip(questions, structured_items):
        # Prepare metric context for better query expansion and evaluation
        metric_context = {
            "metric": meta["metric"],
            "description": meta["description"],
            "criterion": meta["criterion"],
            "section": meta["section"]
        }
        
        output = graph.invoke({
            "current_question": q,
            "chunks": chunks,
            "index": index,
            "report": report,
            "metric_context": metric_context,  # Pass context for better understanding
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
