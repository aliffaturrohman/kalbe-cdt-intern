# src/metadata_manager.py
"""Metadata management system"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

from .config import config
from .logger import AuditLogger

logger = AuditLogger()

class MetadataManager:
    """Manages metadata retrieval and search"""
    
    def __init__(self, metadata_dir: Optional[Path] = None):
        # Menggunakan config.METADATA_DIR sebagai default
        self.metadata_dir = metadata_dir or config.METADATA_DIR
        self._metadata_cache = None
        
    def load_all_metadata(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load semua metadata dari folder"""
        if self._metadata_cache is not None and not force_reload:
            return self._metadata_cache
        
        metadata = {}
        
        # Pastikan direktori ada
        if not self.metadata_dir.exists():
            logger.log("METADATA_LOAD", {
                "status": "ERROR",
                "message": f"Directory not found: {self.metadata_dir}"
            }, level="ERROR")
            return metadata

        json_files = list(self.metadata_dir.glob("*.json"))
        
        if not json_files:
            logger.log("METADATA_LOAD", {
                "status": "WARNING",
                "message": f"No JSON files in {self.metadata_dir}"
            }, level="WARNING")
            return metadata
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    table_name = json_file.stem
                    data = json.load(f)
                    metadata[table_name] = data
                    
                # Log setiap file yang berhasil di-load
                logger.log("METADATA_LOAD_FILE", {
                    "file": json_file.name,
                    "table": table_name,
                    "status": "SUCCESS"
                })
                
            except Exception as e:
                logger.log("METADATA_ERROR", {
                    "file": json_file.name,
                    "error": str(e),
                    "status": "ERROR"
                }, level="ERROR")
        
        self._metadata_cache = metadata
        
        logger.log("METADATA_LOAD_COMPLETE", {
            "total_tables": len(metadata),
            "tables": list(metadata.keys())
        })
        
        return metadata
    
    def get_table_metadata(self, table_name: str) -> Optional[Dict]:
        """Get metadata untuk tabel spesifik"""
        metadata = self.load_all_metadata()
        return metadata.get(table_name)
    
    def find_relevant_tables(self, user_query: str, top_k: int = 3) -> List[Dict]:
        """Cari tabel yang relevan dengan query user"""
        metadata = self.load_all_metadata()
        if not metadata:
            return []
        
        query_terms = set(user_query.lower().split())
        
        scores = []
        for table_name, meta in metadata.items():
            score = 0
            
            # Check description
            desc = meta.get("description", "").lower()
            if any(term in desc for term in query_terms):
                score += 3
            
            # Check column names
            columns = meta.get("columns", {})
            for col_name in columns.keys():
                col_lower = col_name.lower()
                for term in query_terms:
                    if term in col_lower:
                        score += 2
                    elif len(term) > 3 and col_lower.startswith(term[:3]):
                        score += 1
            
            # Check example data
            examples = meta.get("example_rows", [])
            for row in examples:
                for value in row.values():
                    if isinstance(value, str):
                        value_lower = value.lower()
                        for term in query_terms:
                            if term in value_lower:
                                score += 0.5
            
            if score > 0:
                # --- PERBAIKAN: Menghapus slicing pada description ---
                scores.append({
                    "table_name": table_name,
                    "metadata": meta,
                    "relevance_score": score,
                    "description": meta.get("description", "") # Full description
                })
        
        # Sort by score highest first
        scores.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        top_results = scores[:top_k]
        
        logger.log("METADATA_RETRIEVAL", {
            "user_query": user_query, # Full query
            "total_candidates": len(scores),
            "top_results_names": [s["table_name"] for s in top_results],
            "scores": [s["relevance_score"] for s in top_results]
        })
        
        return top_results
    
    def build_schema_prompt(self, table_info: Dict) -> str:
        """Build schema description untuk prompt SQL"""
        meta = table_info["metadata"]
        columns = meta.get("columns", {})
        
        schema_lines = []
        for col_name, col_info in columns.items():
            # Handle jika col_info hanya string deskripsi
            if isinstance(col_info, str):
                col_info = {
                    "type": "string",
                    "description": col_info
                }

            col_type = col_info.get("type", "unknown")
            col_desc = col_info.get("description", "no description")
            schema_lines.append(f"  - {col_name} ({col_type}): {col_desc}")
        
        schema_text = "\n".join(schema_lines)
        
        # Mengembalikan prompt schema lengkap
        return f"""Table: {table_info['table_name']}
Description: {meta.get('description', 'No description')}
Columns:
{schema_text}
Access Column: {meta.get('access_column', 'None')}
Example Data: {json.dumps(meta.get('example_rows', [])[:2], ensure_ascii=False)}"""