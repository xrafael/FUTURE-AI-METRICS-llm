"""Service for loading evaluation criteria from JSON."""
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


class MetricsLoader:
    """Service for loading and processing evaluation criteria."""
    
    @staticmethod
    def load_evaluation_criteria(json_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load evaluation criteria from JSON and convert them into evaluation questions.
        
        Args:
            json_path: Path to JSON file with evaluation criteria
            
        Returns:
            Tuple of (questions list, structured_items list)
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

