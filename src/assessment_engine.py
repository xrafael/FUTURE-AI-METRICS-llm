"""Main assessment engine orchestrating the evaluation pipeline."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

from langgraph.graph import StateGraph

from config.settings import AssessmentConfig
from services.document_loaders import PDFLoader, ReadmeLoader
from services.text_processor import TextProcessor, Chunk
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.report_generator import ReportGenerator
from services.metrics_loader import MetricsLoader


class AssessmentEngine:
    """Main engine for conducting paper assessments."""
    
    def __init__(self, config: AssessmentConfig):
        """
        Initialize assessment engine.
        
        Args:
            config: Assessment configuration
        """
        self.config = config
        
        # Initialize services
        self.pdf_loader = PDFLoader()
        self.readme_loader = ReadmeLoader()
        self.text_processor_pdf = TextProcessor(
            chunk_size=config.chunking.pdf_chunk_size,
            overlap=config.chunking.pdf_overlap
        )
        self.text_processor_readme = TextProcessor(
            chunk_size=config.chunking.readme_chunk_size,
            overlap=config.chunking.readme_overlap
        )
        self.embedding_service = EmbeddingService(config.embedding)
        self.llm_service = LLMService(config.ollama)
        self.report_generator = ReportGenerator()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        self.graph = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow for assessment.
        
        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(state_schema=dict)
        
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("append", self._append_report_node)
        workflow.add_edge("retrieve", "evaluate")
        workflow.add_edge("evaluate", "append")
        workflow.set_entry_point("retrieve")
        
        return workflow
    
    def _retrieve_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a question.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Updated state with retrieved chunks
        """
        query = state["current_question"]
        qvec = self.embedding_service.encode_query(query)
        
        index = state["index"]
        chunks = state["chunks"]
        
        D, I = index.search(qvec, k=self.config.retrieval.top_k)
        state["retrieved_chunks"] = [chunks[i] for i in I[0]]
        return state
    
    def _evaluate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate question against retrieved chunks using LLM.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Updated state with evaluation result
        """
        question = state["current_question"]
        context_parts = []
        
        for chunk in state["retrieved_chunks"]:
            # Handle both dict and Chunk object formats
            if isinstance(chunk, dict):
                source_label = chunk.get("source", "pdf")
                line_start = chunk.get("line_start", 0)
                line_end = chunk.get("line_end", 0)
                text = chunk.get("text", "")
                page = chunk.get("page", "?")
            else:
                source_label = chunk.source
                line_start = chunk.line_start
                line_end = chunk.line_end
                text = chunk.text
                page = chunk.page
            
            if source_label == "readme":
                context_parts.append(
                    f"[README|l={line_start}-{line_end}]: {text}"
                )
            else:
                context_parts.append(
                    f"[p={page}|l={line_start}-{line_end}]: {text}"
                )
        
        context = "\n".join(context_parts)
        
        prompt = f"""
You are an expert research-paper reviewer.
Your task is to evaluate whether a research article and its associated GitHub README answer a specific question.  

Use ONLY the provided context.
If the context does not contain enough information, answer “no” or “partially”. Do NOT invent content.

QUESTION:
{question}

CONTEXT (segments with page/line identifiers from PDF, or line identifiers from README):
{context}

TASK:
1. Decide whether the paper or the README fully, partially, or not at all address the question. 
Allowed values: "yes", "no", "partially".
2. Provide all pieces of evidence that support your judgment:
Page number(s),
Line number(s),
Direct quote from the provided context (no paraphrasing),
Specify "pdf" or "readme" as the source.
3. Give personalized recommendations for how the authors could improve the paper/README to better address the question.


Return ONLY valid JSON with the following structure:
{{
    "addressed": "yes/no",
    "evidence": [
    {{
        "page": <page number>,
        "line": <line number>,
        "quote": "<exact quote from context>",
        "source": "pdf" or "readme",
        "recommendation": "<specific recommendation>"
    }}]
    }}
"""
        
        result = self.llm_service.invoke(prompt)
        state["evaluation"] = result
        return state
    
    def _append_report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append evaluation result to report.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Updated state with appended report
        """
        state["report"].append({
            "question": state["current_question"],
            "analysis": state["evaluation"]
        })
        return state
    
    def analyze_paper(
        self,
        pdf_path: Path,
        questions: List[str],
        structured_items: List[Dict[str, Any]],
        readme_url: Optional[str] = None,
        output_pdf_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze a paper against a list of metrics.
        
        Args:
            pdf_path: Path to the PDF file
            questions: List of evaluation questions
            structured_items: List of structured metric items
            readme_url: Optional GitHub README URL to include in assessment
            output_pdf_path: Optional path for the output PDF report. If not provided, uses default path.
            
        Returns:
            List of report items with evaluation results
        """
        # Load PDF
        pages = self.pdf_loader.load(pdf_path)
        chunks: List[Chunk] = []
        
        # Chunk PDF pages
        for page in pages:
            chunks.extend(self.text_processor_pdf.chunk_pdf_page(page))
        
        # Load and chunk README if URL is provided
        if readme_url:
            print(f"Fetching README from: {readme_url}")
            readme_content = self.readme_loader.load(readme_url)
            if readme_content:
                readme_chunks = self.text_processor_readme.chunk_readme(readme_content)
                chunks.extend(readme_chunks)
                print(f"Added {len(readme_chunks)} README chunks to assessment")
            else:
                print("Warning: Could not fetch README content")
        
        # Warn if too many chunks
        if len(chunks) > self.config.retrieval.max_chunks_warning:
            print(f"Warning: {len(chunks)} chunks detected. This may cause memory issues.")
            print(f"Consider increasing chunk sizes (e.g., pdf_chunk_size=1000, readme_chunk_size=800)")
        else:
            print(f"Total chunks created: {len(chunks)}")
        
        # Build index
        index, vectors = self.embedding_service.build_index(chunks)
        
        # Keep chunks as objects for retrieval, but convert to dicts for state
        chunks_dict = [
            {
                "text": chunk.text,
                "page": chunk.page,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "source": chunk.source
            }
            for chunk in chunks
        ]
        
        report = []
        
        # Run evaluation for each metric-based question
        for question, meta in zip(questions, structured_items):
            output = self.graph.invoke({
                "current_question": question,
                "chunks": chunks_dict,
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
        
        # Determine output PDF path
        if output_pdf_path is None:
            # Ensure results directory exists
            os.makedirs(self.config.results_dir, exist_ok=True)
            pdf_output_path = self.config.results_dir / "assessment_report_advanced_01.pdf"
        else:
            # Ensure parent directory exists for custom path
            os.makedirs(output_pdf_path.parent, exist_ok=True)
            pdf_output_path = output_pdf_path
        
        # Generate PDF
        pdf_path = self.report_generator.generate_pdf(report, pdf_output_path)
        print("PDF saved at:", pdf_path)
        
        return report
    
    def cleanup(self):
        """Clean up resources."""
        self.embedding_service.cleanup()

