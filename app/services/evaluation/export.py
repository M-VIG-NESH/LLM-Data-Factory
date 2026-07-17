"""
Export functionality for generated datasets.
"""

import json
import csv
import os
from typing import List, Dict
from datetime import datetime
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class DatasetExporter:
    """Export datasets to various formats."""
    
    @staticmethod
    def export_to_jsonl(
        data: List[Dict],
        output_path: str,
        format_type: str = "llama"
    ) -> str:
        """
        Export to JSONL format for LLM fine-tuning.
        
        Args:
            data: List of dataset entries
            output_path: Output file path
            format_type: Format (llama, gemini, openai)
            
        Returns:
            Path to exported file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                if format_type == "llama":
                    # Llama format: instruction-based
                    formatted = {
                        "instruction": entry.get("question", ""),
                        "input": entry.get("context", ""),
                        "output": entry.get("answer", entry.get("summary", ""))
                    }
                
                elif format_type == "gemini":
                    # Gemini format: text_input/output
                    formatted = {
                        "text_input": f"Context: {entry.get('context', '')}\n\nQuestion: {entry.get('question', '')}",
                        "output": entry.get("answer", entry.get("summary", ""))
                    }
                
                elif format_type == "openai":
                    # OpenAI format: messages
                    formatted = {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": entry.get("question", "")},
                            {"role": "assistant", "content": entry.get("answer", entry.get("summary", ""))}
                        ]
                    }
                
                else:
                    # Default: raw format
                    formatted = entry
                
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(data)} entries to JSONL: {output_path}")
        return output_path
    
    @staticmethod
    def export_to_csv(
        data: List[Dict],
        output_path: str
    ) -> str:
        """
        Export to CSV format for analysis.
        
        Args:
            data: List of dataset entries
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not data:
            logger.warning("No data to export")
            return output_path
        
        # Get all unique keys
        fieldnames = set()
        for entry in data:
            fieldnames.update(entry.keys())
        fieldnames = sorted(fieldnames)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Exported {len(data)} entries to CSV: {output_path}")
        return output_path
    
    @staticmethod
    def export_to_json(
        data: List[Dict],
        output_path: str
    ) -> str:
        """
        Export to JSON format.
        
        Args:
            data: List of dataset entries
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(data)} entries to JSON: {output_path}")
        return output_path
    
    @staticmethod
    def generate_export_filename(
        job_id: str,
        format: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Generate standardized export filename.
        
        Args:
            job_id: Job identifier
            format: File format (jsonl, csv, json)
            include_timestamp: Whether to include timestamp
            
        Returns:
            Filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        
        if timestamp:
            filename = f"dataset_{job_id}_{timestamp}.{format}"
        else:
            filename = f"dataset_{job_id}.{format}"
        
        return os.path.join(settings.export_dir, filename)


# Convenience function
def export_dataset(
    data: List[Dict],
    format: str = "jsonl",
    job_id: str = "default",
    llm_format: str = "llama"
) -> str:
    """
    Export dataset to specified format.
    
    Args:
        data: Dataset entries
        format: Export format (jsonl, csv, json)
        job_id: Job identifier
        llm_format: LLM-specific format (for JSONL)
        
    Returns:
        Path to exported file
    """
    exporter = DatasetExporter()
    output_path = exporter.generate_export_filename(job_id, format)
    
    if format == "jsonl":
        return exporter.export_to_jsonl(data, output_path, llm_format)
    elif format == "csv":
        return exporter.export_to_csv(data, output_path)
    elif format == "json":
        return exporter.export_to_json(data, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
