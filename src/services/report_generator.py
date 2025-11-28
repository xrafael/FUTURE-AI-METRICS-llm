"""PDF report generation service."""
import json
from pathlib import Path
from typing import List, Dict, Any

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


class ReportGenerator:
    """Service for generating PDF assessment reports."""
    
    def __init__(self):
        """Initialize report generator."""
        self.styles = getSampleStyleSheet()
    
    def generate_pdf(self, report: List[Dict[str, Any]], output_path: Path) -> Path:
        """
        Create a PDF assessment report from evaluation results.
        
        Args:
            report: List of report items with evaluation results
            output_path: Path where PDF should be saved
            paper_metadata: Optional dictionary with 'title' and 'authors' keys
            
        Returns:
            Path to generated PDF
        """
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=2*cm,
            rightMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
        )
        
        story = []
        
        # Title
        story.append(Paragraph("FUTURE-AI Compliance Assessment", self.styles["Title"]))
        story.append(Spacer(1, 0.3*cm))
        
        # Paper title and authors
        title = "" #paper_metadata.get("title")
        authors = "" #paper_metadata.get("authors")
        
        if title:
            story.append(Paragraph(f"<b>Paper Title:</b> {title}", self.styles["Heading2"]))
            story.append(Spacer(1, 0.2*cm))
        
        if authors:
            story.append(Paragraph(f"<b>Authors:</b> {authors}", self.styles["BodyText"]))
            story.append(Spacer(1, 0.3*cm))
        
        story.append(Spacer(1, 0.5*cm))
        
        current_section = None
        current_criterion = None
        
        for item in report:
            # Section title
            if item["section"] != current_section:
                current_section = item["section"]
                section_title = item["section"][0].upper() + item["section"][1:].lower()
                story.append(Paragraph(section_title, self.styles["Heading1"]))
                story.append(Spacer(1, 0.3*cm))
            
            # Criterion subtitle
            if item["criterion"] != current_criterion:
                current_criterion = item["criterion"]
                criterion_title = item["criterion"][0].upper() + item["criterion"][1:].lower()
                story.append(Paragraph(criterion_title, self.styles["Heading2"]))
                story.append(Spacer(1, 0.2*cm))
            
            # Metric Name
            metric_name = item['metric'][0].upper() + item['metric'][1:].lower()
            story.append(Paragraph(f"Metric: <b>{metric_name}</b>", self.styles["Heading3"]))
            story.append(Spacer(1, 0.2*cm))
            
            # Metric Description
            story.append(Paragraph(item["description"], self.styles["BodyText"]))
            story.append(Spacer(1, 0.2*cm))
            
            # Parse LLM response
            analysis = self._parse_analysis(item["analysis"])
            
            # Addressed?
            story.append(Paragraph(
                f"<b>Addressed:</b> {analysis.get('addressed', 'unknown')}",
                self.styles["BodyText"]
            ))
            story.append(Spacer(1, 0.2*cm))
            
            # Evidence
            if "evidence" in analysis and analysis["evidence"]:
                story.append(Paragraph("<b>Evidence:</b>", self.styles["BodyText"]))
                for ev in analysis["evidence"]:
                    source = ev.get('source', 'pdf')
                    page_label = "README" if source == "readme" else f"Page {ev.get('page')}"
                    txt = (
                        f"{page_label}, Line {ev.get('line')} ({source})<br/>"
                        f"<i>{ev.get('quote')}</i>"
                    )
                    story.append(Paragraph(txt, self.styles["BodyText"]))
                    
                    # Add recommendation if present
                    if ev.get('recommendation'):
                        recommendation_txt = f"<b>Recommendation:</b> {ev.get('recommendation')}"
                        story.append(Paragraph(recommendation_txt, self.styles["BodyText"]))
                    
                    story.append(Spacer(1, 0.2*cm))
            else:
                story.append(Paragraph("No evidence found.", self.styles["BodyText"]))
            
            story.append(Spacer(1, 0.4*cm))
        
        doc.build(story)
        return output_path
    
    def _parse_analysis(self, analysis_text: Any) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.
        
        Args:
            analysis_text: Raw analysis text or object
            
        Returns:
            Parsed analysis dictionary
        """
        # Handle different response formats
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
        
        return analysis

