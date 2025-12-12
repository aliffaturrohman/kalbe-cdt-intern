# src/sql_executor.py
"""SQL execution module"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time

from .config import config
from .logger import AuditLogger

logger = AuditLogger()

class SQLExecutor:
    """Eksekusi SQL query ke database"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DB_PATH
        
    def execute(self, sql: str, params: Tuple = None) -> Dict[str, Any]:
        """Eksekusi SQL query dan return hasil"""
        start_time = time.time()
        
        try:
            # Cek file database
            if not self.db_path.exists():
                return {
                    "success": False,
                    "error": f"Database not found: {self.db_path}",
                    "data": None,
                    "row_count": 0,
                    "execution_time": 0
                }
            
            # Execute query
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                
                # Get column names
                columns = [description[0] for description in cursor.description] if cursor.description else []
                
                # Fetch all results
                rows = cursor.fetchall()
                
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=columns) if columns else pd.DataFrame()
                
                execution_time = time.time() - start_time
                
                result = {
                    "success": True,
                    "data": df,
                    "row_count": len(df),
                    "columns": columns,
                    "sql": sql,
                    "execution_time": execution_time
                }
                
                logger.log("SQL_EXECUTION_SUCCESS", {
                    "sql_preview": sql[:200],
                    "row_count": len(df),
                    "columns": columns[:5] if columns else [],
                    "execution_time": execution_time
                }, level="SUCCESS")
                
                return result
                
            finally:
                conn.close()
                
        except sqlite3.Error as e:
            execution_time = time.time() - start_time
            error_msg = f"SQL execution error: {str(e)}"
            
            logger.log("SQL_EXECUTION_ERROR", {
                "error": str(e),
                "sql": sql[:200],
                "execution_time": execution_time
            }, level="ERROR")
            
            return {
                "success": False,
                "error": error_msg,
                "data": None,
                "row_count": 0,
                "execution_time": execution_time
            }
    
    def test_connection(self) -> bool:
        """Test koneksi ke database"""
        try:
            if not self.db_path.exists():
                logger.log("DB_CONNECTION", {
                    "status": "ERROR",
                    "message": f"Database file not found: {self.db_path}"
                }, level="ERROR")
                return False
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
            tables = cursor.fetchall()
            conn.close()
            
            logger.log("DB_CONNECTION", {
                "status": "SUCCESS",
                "table_count": len(tables) if tables else 0,
                "message": f"Connected to database: {self.db_path}"
            }, level="SUCCESS")
            
            return True
            
        except Exception as e:
            logger.log("DB_CONNECTION", {
                "status": "ERROR",
                "error": str(e),
                "message": f"Failed to connect to database: {self.db_path}"
            }, level="ERROR")
            return False
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get informasi tentang tabel"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            sample_data = cursor.fetchall()
            
            conn.close()
            
            return {
                "success": True,
                "table_name": table_name,
                "columns": columns,
                "sample_data": sample_data,
                "column_count": len(columns)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "table_name": table_name
            }