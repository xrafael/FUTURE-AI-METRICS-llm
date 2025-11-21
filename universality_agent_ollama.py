import json
import re
import os
from pathlib import Path
from typing import List, Dict

# Choose your model provider here (OpenAI API example)
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

"""
*****************
Key Notes & Observations
*****************

Local LLM:

Uses Ollama LLaMA 3.1, avoiding API costs.
Good for reproducibility but depends on local GPU/CPU.

Evidence Retrieval:

Simple keyword search.
Could be improved via semantic search (vector embeddings).

Structured Evaluation:

JSON output ensures downstream analysis can parse results easily.
Includes fallback for non-JSON LLM responses.

Scalability:

Loops over metrics per criterion; may be slow on large PDFs or many metrics.
Could parallelize LLM calls per metric.

Customizability:

Supports multiple evaluation criteria and metrics.
Can switch LLM provider easily by replacing ChatOllama.
"""

# ---------- LLM: LOCAL via OLLAMA ----------
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1", temperature=0)

# ------------------------- LOAD DATA -----------------------------

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def load_pdf_text(pdf_path: str) -> List[Dict]:
    """
    Loads PDF and returns list of:
    { "page": page_num, "text": page_content }
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    out = []
    for i, page in enumerate(pages):
        out.append({"page": i+1, "text": page.page_content})
    return out



# -------------------- PDF SEARCH ENGINE --------------------------

def search_pdf(query: str, pages: List[Dict], top_k: int = 5):
    """
    Simple keyword-based search over the PDF pages.
    (You can replace with a vector index if needed.)
    """
    scored = []
    for p in pages:
        text = p["text"]
        score = text.lower().count(query.lower())
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top_k]]


# ------------------- LLM EVALUATION PROMPT -----------------------

EVAL_PROMPT = PromptTemplate(
    template="""
You are an expert evaluator of AI clinical universality.

Metric:
"{metric_name}"

Metric description:
"{metric_description}"

Relevant excerpts from the paper:
{evidence}

Task: Determine if the paper addresses this metric.
Be strict and classify as one of: "yes", "no", "partial".

Return ONLY valid JSON with fields:
- status
- interpretation
- evidence_used
""",
    input_variables=["metric_name", "metric_description", "evidence"]
)

def evaluate_metric(metric_name: str, metric_description: str, evidence: str) -> dict:
    prompt = EVAL_PROMPT.format(
            metric_name=metric_name,
            metric_description=metric_description,
            evidence=evidence
        )
    response = llm.invoke(prompt)
    
    # Extract JSON from response
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {
            "status": "error",
            "interpretation": "LLM returned non-JSON. Full text: " + response,
            "evidence_used": evidence
        }
    return parsed


# ----------------------- MAIN ANALYSIS ----------------------------

def run_evaluation(metrics_json_path, paper_pdf_path, idx=0):

    future_ai = load_json(metrics_json_path)
    pdf_pages = load_pdf_text(paper_pdf_path)

    idx_metric = future_ai["future_ai_metrics"][idx]

    results = {
        "name": idx_metric["name"],
        "description": idx_metric["description"],
        "evaluation": {}
    }

    for criterion in idx_metric["evaluation_criteria"]:
        c_name = criterion["name"]
        results["evaluation"][c_name] = {}
        for metric in criterion["metrics"]:
            m_name = metric["name"]
            m_desc = metric["description"]

            # --- retrieve evidence ---
            retrieved_pages = search_pdf(m_name, pdf_pages)
            if not retrieved_pages:
                # fallback: search using key terms of description
                keywords = m_desc.split(" ")
                if len(keywords) > 3:
                    retrieved_pages = search_pdf(keywords[0], pdf_pages)

            # Prepare evidence text
            evidence_text = "\n\n".join(
                [f"[Page {p['page']}]\n{p['text'][:1500]}" for p in retrieved_pages]
            )

            # Evaluate with LLM
            evaluation = evaluate_metric(m_name, m_desc, evidence_text)
            results["evaluation"][c_name][m_name] = evaluation

    return results


# ----------------------- RUN & SAVE OUTPUT ----------------------------

if __name__ == "__main__":
    metrics_json_path = "future_ai_metrics.json"
    paper_pdf_path = "./papers/s41597-025-04707-4.pdf"

    report = run_evaluation(metrics_json_path, paper_pdf_path, 0)

    with open("results/universality_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Done. Report saved to universality_report.json")
