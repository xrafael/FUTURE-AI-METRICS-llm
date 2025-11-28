"""Main entry point for universality assessment."""
# Disable multiprocessing to avoid semaphore leaks - MUST be set before any imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"

import atexit
import argparse
from pathlib import Path
import sys

# Add src directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import AssessmentConfig
from assessment_engine import AssessmentEngine
from services.metrics_loader import MetricsLoader


def main(pdf_path: Path, readme_url: str, output_pdf_path: Path = None):
    """
    Main function to run universality assessment.
    
    Args:
        output_pdf_path: Optional path for the output PDF report. If not provided, uses default path.
    """
    # Initialize configuration
    config = AssessmentConfig()
    
    # Initialize assessment engine
    engine = AssessmentEngine(config)
    
    # Register cleanup function
    atexit.register(engine.cleanup)
    
    # Load evaluation criteria
    questions, structured_items = MetricsLoader().load_evaluation_criteria(Path(__file__).parent / "config" / "future_ai_metrics.json")
    
    # Analyze paper    
    report = engine.analyze_paper(
        pdf_path=pdf_path,
        questions=questions,
        structured_items=structured_items,
        readme_url=readme_url,
        output_pdf_path=output_pdf_path
    )
    
    # Print summary
    if output_pdf_path:
        print(f"Task completed. Report saved to {output_pdf_path}")
    else:
        print("Task completed. Report saved to assessment_report.pdf")


if __name__ == "__main__":
    pdf_path = Path("./papers/s41597-025-04707-4.pdf")
    readme_url = "https://github.com/LidiaGarrucho/MAMA-MIA/blob/main/README.md"  # Set to None if not using README
    output_pdf_path = Path("./results/assessment_report_advanced_06.pdf")
    main(pdf_path=pdf_path, readme_url=readme_url, output_pdf_path=output_pdf_path)

