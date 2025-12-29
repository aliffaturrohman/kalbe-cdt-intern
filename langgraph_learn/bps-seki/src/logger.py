# src/logger.py
"""Comprehensive logging system"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .config import config

class AuditLogger:
    """Sistem logging komprehensif untuk semua aktivitas"""
    
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or config.LOG_DIR
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure log directory exists
        self.log_dir.mkdir(exist_ok=True)
        
    def log(self, event_type: str, data: Dict, level: str = "INFO"):
        """Log event ke file dan console"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "level": level,
            "event_type": event_type,
            **data
        }
        
        # Console output dengan warna
        colors = {
            "INFO": "\033[94m",      # Blue
            "WARNING": "\033[93m",   # Yellow
            "ERROR": "\033[91m",     # Red
            "SUCCESS": "\033[92m"    # Green
        }
        color = colors.get(level, "\033[0m")
        reset = "\033[0m"
        
        # Format console message (hanya ambil message pendek untuk console)
        message = data.get("message", "")
        print(f"{color}[{level}] {event_type}: {message}{reset}")
        
        # Save to file (JSONL format) - DISINI DATA LENGKAP DISIMPAN
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        return log_entry
    
    def log_user_input(self, user_input: str, context: Dict):
        """Log input dari user"""
        return self.log("USER_INPUT", {
            "user_input": user_input,
            "user_context": context,
            "message": f"User query: {user_input[:100]}..." # Console tetap pendek agar rapi
        })
    
    def log_llm_call(self, model: str, prompt: str, response: str):
        """Log panggilan ke LLM (FULL CONTENT)"""
        return self.log("LLM_CALL", {
            "model": model,
            # --- PERBAIKAN: Mengambil string penuh (tanpa slicing [:200]) ---
            "prompt_content": prompt,       
            "response_content": response,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "message": f"LLM call to {model}"
        })
    
    def log_sql_generation(self, sql: str, metadata: Dict):
        """Log SQL generation"""
        return self.log("SQL_GENERATION", {
            "generated_sql": sql, # Full SQL disimpan di sini
            "table_used": metadata.get("table_name", "unknown"),
            "message": f"Generated SQL: {sql[:50]}..." # Console preview pendek
        })
    
    def log_table_selection(self, user_query: str, selected_table: str, 
                            confidence: float, reason: str):
        """Log table selection"""
        return self.log("TABLE_SELECTION", {
            "user_query": user_query,
            "selected_table": selected_table,
            "confidence": confidence,
            "reason": reason,
            "message": f"Selected table: {selected_table} (confidence: {confidence:.2f})"
        })
    
    def log_forecast_result(self, forecast_data: Dict):
        """Log hasil forecasting"""
        return self.log("FORECAST_RESULT", {
            "data_points": forecast_data.get("data_points", 0),
            "periods": forecast_data.get("periods", 0),
            "method": forecast_data.get("method", "unknown"),
            "message": f"Forecast completed with {forecast_data.get('method', 'unknown')} method"
        })
    
    def log_workflow_step(self, node: str, state: Dict):
        """Log langkah workflow"""
        return self.log("WORKFLOW_STEP", {
            "node": node,
            "state_keys": list(state.keys()),
            "selected_table": state.get("selected_table"),
            "intent": state.get("intent"),
            "message": f"Workflow node: {node}"
        })
    
    def get_recent_logs(self, limit: int = 10) -> list:
        """Get recent log entries"""
        log_files = list(self.log_dir.glob("*.jsonl"))
        if not log_files:
            return []
        
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-limit:]
                return [json.loads(line) for line in lines]
        except:
            return []