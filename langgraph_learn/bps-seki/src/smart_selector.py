# src/smart_selector.py
"""Smart table selector dengan LLM"""

import json
import re
from typing import Dict, List, Any, Optional

from .config import config
from .logger import AuditLogger
from .llm_client import llm_client

logger = AuditLogger()

class SmartTableSelector:
    """LLM-powered intelligent table selector"""
    
    def __init__(self):
        self.selection_cache = {}
    
    def extract_years_from_query(self, query: str) -> List[int]:
        """Extract years from user query"""
        # Menangkap tahun 20xx
        years = re.findall(r'\b(20\d{2})\b', query)
        return [int(year) for year in years]
    
    def select_best_table(self, user_query: str, candidate_tables: List[Dict], 
                         user_context: Dict) -> Dict[str, Any]:
        """
        Use LLM to automatically select the best table from candidates.
        Returns selected table and selection reason.
        """
        if not candidate_tables:
            return {"selected": None, "reason": "No candidate tables", "confidence": 0}
        
        if len(candidate_tables) == 1:
            return {
                "selected": candidate_tables[0],
                "reason": "Only one candidate table available",
                "confidence": 1.0
            }
        
        # Extract years from query
        years = self.extract_years_from_query(user_query)
        
        # Build context for LLM
        tables_context = []
        for i, table in enumerate(candidate_tables, 1):
            meta = table["metadata"]
            # Ambil 5 kolom pertama sebagai preview
            columns = list(meta.get("columns", {}).keys())[:8] 
            
            # Check if table has year column
            has_year_col = any("tahun" in col.lower() or "year" in col.lower() for col in columns)
            
            tables_context.append(f"""
            Table Index: {i}
            Name: {table['table_name']}
            Description: {meta.get('description', 'No description')[:150]}
            Columns: {', '.join(columns)}...
            Has year column: {'Yes' if has_year_col else 'No'}
            Access column: {meta.get('access_column', 'None')}
            Relevance Score: {table.get('relevance_score', 0):.2f}
            """)
        
        tables_text = "\n".join(tables_context)
        
        # Build prompt for LLM
        prompt = f"""
        You are an AI assistant that helps select the most appropriate database table for a user's query.
        
        USER QUERY: "{user_query}"
        
        USER CONTEXT:
        - Region: {user_context.get('region', 'Not specified')}
        - Leveldata: {user_context.get('leveldata', 'Not specified')}
        - Years mentioned: {years if years else 'None'}
        
        CANDIDATE TABLES:
        {tables_text}
        
        INSTRUCTIONS:
        1. Analyze the user's query and identify the intent.
        2. Select the table that BEST matches the query based on columns, description, and context.
        3. If specific years are mentioned, prioritize tables containing year information.
        4. Return ONLY a JSON object with this EXACT format:
        {{
            "selected_table_index": <number 1-{len(candidate_tables)}>,
            "confidence": <float 0.1 to 1.0>,
            "reason": "<short explanation why this table was selected>"
        }}
        
        Do NOT include markdown formatting (like ```json). Just the raw JSON string.
        """
        
        try:
            # Call LLM
            response = llm_client.call_user_llm(prompt)
            
            if not response["success"]:
                raise ValueError(f"LLM call failed: {response.get('error')}")
            
            content = response["content"].strip()
            
            # Bersihkan markdown formatting jika ada
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON response
            # Gunakan regex untuk mencari kurung kurawal terluar
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                
                idx = result.get("selected_table_index")
                
                # Validate selection index
                if isinstance(idx, int) and 1 <= idx <= len(candidate_tables):
                    selected_idx = idx - 1
                    selected_table = candidate_tables[selected_idx]
                    
                    logger.log("TABLE_SELECTION_LLM", {
                        "user_query": user_query,
                        "selected_table": selected_table["table_name"],
                        "confidence": result.get("confidence", 0.5),
                        "reason": result.get("reason", "No reason"),
                        "years_detected": years
                    })
                    
                    return {
                        "selected": selected_table,
                        "confidence": result.get("confidence", 0.5),
                        "reason": result.get("reason", "No reason"),
                        "years_detected": years
                    }
            
            # Fallback to highest relevance score if parsing failed
            raise ValueError("Invalid JSON structure or index out of bounds")
            
        except Exception as e:
            # Fallback mechanism
            best_table = max(candidate_tables, key=lambda x: x.get("relevance_score", 0))
            
            logger.log("TABLE_SELECTION_FALLBACK", {
                "error": str(e),
                "user_query": user_query,
                "selected_table": best_table["table_name"],
                "reason": "LLM selection failed/error, using relevance score"
            }, level="WARNING")
            
            return {
                "selected": best_table,
                "confidence": 0.3,
                "reason": f"Fallback due to error: {str(e)[:50]}",
                "years_detected": years
            }
    
    def build_smart_sql_prompt(self, user_query: str, table_info: Dict, 
                              user_context: Dict) -> str:
        """Build smart SQL generation prompt dengan context lengkap"""
        table_name = table_info["table_name"]
        metadata = table_info["metadata"]
        
        # Build columns info
        columns_info = []
        for col_name, col_meta in metadata.get("columns", {}).items():
            if isinstance(col_meta, str):
                col_meta = {"type": "string", "description": col_meta}
            col_type = col_meta.get("type", "unknown")
            col_desc = col_meta.get("description", "no description")
            columns_info.append(f"  - {col_name} ({col_type}): {col_desc}")
        
        columns_text = "\n".join(columns_info)
        
        # Extract years logic
        years = self.extract_years_from_query(user_query)
        years_filter = ""
        if years:
            year_cols = [col for col in metadata.get("columns", {}).keys() 
                        if "tahun" in col.lower() or "year" in col.lower()]
            if year_cols:
                year_col = year_cols[0]
                years_filter = f"User mentioned years: {years}. Add filter WHERE {year_col} IN ({', '.join(map(str, years))}) if relevant."
        
        # Safe get default limit
        default_limit = getattr(config, "DEFAULT_LIMIT", 5)
        
        # --- PERBAIKAN LOGIKA ACCESS COLUMN (CRITICAL FIX) ---
        raw_access_col = metadata.get('access_column')
        
        # Cek apakah access_col valid (bukan None, bukan "None", bukan kosong)
        is_valid_access_col = (
            raw_access_col is not None 
            and str(raw_access_col).strip().lower() not in ['none', 'null', '']
        )
        
        region_rule = ""
        if is_valid_access_col and user_context.get('region'):
            region_rule = f"2. The user is restricted to region '{user_context.get('region')}'. You MUST add: WHERE {raw_access_col} LIKE '{user_context.get('region')}%'"
        else:
            # Eksplisit melarang filter region jika kolomnya tidak ada
            region_rule = "2. Do NOT add any region filter (Access Column is not applicable for this table)."

        # Build prompt
        prompt = f"""
        You are an expert SQLite SQL generator.
        
        USER QUERY: "{user_query}"
        
        TABLE INFORMATION:
        Table: {table_name}
        Description: {metadata.get('description', 'No description')}
        
        COLUMNS:
        {columns_text}
        
        IMPORTANT RULES:
        1. Generate ONLY SELECT statements.
        {region_rule}
        3. {years_filter}
        4. Add LIMIT {default_limit} if the user does not specify a quantity.
        5. Return ONLY the SQL code. No markdown, no explanations.
        6. CRITICAL: Generate ONLY ONE SINGLE SQL STATEMENT.
        
        SQLITE SPECIFIC RULES (CRITICAL):
        - Do NOT use '::' for casting (e.g., '::numeric'). Use CAST(col AS TYPE).
        - Do NOT use STRING_AGG. Use GROUP_CONCAT(col, ', ') instead.
        - Do NOT use specific PostgreSQL/MySQL functions. Stick to standard SQLite.
        - Date/Year columns are often stored as Strings or Integers, handle accordingly.
        - Case insensitive matching: use UPPER(col) LIKE '%VALUE%' if needed.
        
        EXAMPLE DATA (for context):
        {json.dumps(metadata.get('example_rows', [])[:2], ensure_ascii=False)}
        
        SQL QUERY:
        """
        
        return prompt
    
    