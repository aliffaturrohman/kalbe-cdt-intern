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
                "reason": "Only one candidate table",
                "confidence": 1.0
            }
        
        # Extract years from query
        years = self.extract_years_from_query(user_query)
        
        # Build context for LLM
        tables_context = []
        for i, table in enumerate(candidate_tables, 1):
            meta = table["metadata"]
            columns = list(meta.get("columns", {}).keys())[:5]
            
            # Check if table has year column
            has_year_col = any("tahun" in col.lower() or "year" in col.lower() for col in columns)
            
            tables_context.append(f"""
            {i}. {table['table_name']}
               - Description: {meta.get('description', 'No description')[:80]}...
               - Columns: {', '.join(columns)}
               - Has year column: {'Yes' if has_year_col else 'No'}
               - Access column: {meta.get('access_column', 'None')}
               - Relevance score: {table['relevance_score']:.2f}
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
        1. Analyze the user's query and what data they need
        2. Consider if specific years are mentioned
        3. Select the table that BEST matches the query based on:
           - Column names matching query keywords
           - Presence of year columns if years are mentioned
           - Access column matching user's region
           - Table description relevance
        
        4. Return ONLY JSON with this format:
        {{
            "selected_table_index": <number 1-{len(candidate_tables)}>,
            "selected_table_name": "<table_name>",
            "confidence": <0.1 to 1.0>,
            "reason": "<brief explanation>",
            "years_detected": {years}
        }}
        
        Do not include any other text.
        """
        
        try:
            # Call LLM
            response = llm_client.call_user_llm(prompt)
            
            if not response["success"]:
                raise ValueError(f"LLM call failed: {response.get('error')}")
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response["content"], re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate selection
                if 1 <= result["selected_table_index"] <= len(candidate_tables):
                    selected_idx = result["selected_table_index"] - 1
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
                        "years_detected": years,
                        "llm_response": result
                    }
            
            # Fallback to highest relevance score
            best_table = max(candidate_tables, key=lambda x: x["relevance_score"])
            
            logger.log("TABLE_SELECTION_FALLBACK", {
                "user_query": user_query,
                "selected_table": best_table["table_name"],
                "confidence": 0.3,
                "reason": "LLM selection failed, using relevance score"
            })
            
            return {
                "selected": best_table,
                "confidence": 0.3,
                "reason": "Fallback to highest relevance score",
                "years_detected": years
            }
            
        except Exception as e:
            # Error handling
            best_table = max(candidate_tables, key=lambda x: x["relevance_score"])
            
            logger.log("TABLE_SELECTION_ERROR", {
                "error": str(e),
                "user_query": user_query,
                "selected_table": best_table["table_name"],
                "confidence": 0.1,
                "reason": f"Error in LLM selection: {str(e)[:100]}"
            })
            
            return {
                "selected": best_table,
                "confidence": 0.1,
                "reason": f"Error: {str(e)[:100]}",
                "error": str(e)
            }
    
    def build_smart_sql_prompt(self, user_query: str, table_info: Dict, 
                              user_context: Dict) -> str:
        """Build smart SQL generation prompt dengan context lengkap"""
        table_name = table_info["table_name"]
        metadata = table_info["metadata"]
        
        # Build columns info
        columns_info = []
        for col_name, col_meta in metadata.get("columns", {}).items():

            # Normalize metadata if it's a string
            if isinstance(col_meta, str):
                col_meta = {
                    "type": "string",
                    "description": col_meta
                }

            col_type = col_meta.get("type", "unknown")
            col_desc = col_meta.get("description", "no description")

            columns_info.append(f"  - {col_name} ({col_type}): {col_desc}")
        
        columns_text = "\n".join(columns_info)
        
        # Extract years
        years = self.extract_years_from_query(user_query)
        years_filter = ""
        if years:
            # Find year column
            year_cols = [col for col in metadata.get("columns", {}).keys() 
                        if "tahun" in col.lower() or "year" in col.lower()]
            if year_cols:
                year_col = year_cols[0]
                years_filter = f"\nUser mentioned years: {years}. Add WHERE {year_col} IN ({', '.join(map(str, years))}) if appropriate."
        
        # Build prompt
        prompt = f"""
        You are an expert SQLite SQL generator.
        
        USER QUERY: "{user_query}"
        
        TABLE INFORMATION:
        Table: {table_name}
        Description: {metadata.get('description', 'No description')}
        
        COLUMNS:
        {columns_text}
        
        USER CONTEXT:
        - Region: {user_context.get('region')}
        - Leveldata: {user_context.get('leveldata')}
        - Access Column: {metadata.get('access_column', 'None')}
        
        IMPORTANT RULES:
        1. Generate ONLY SELECT statements
        2. If access_column exists, MUST filter by user's region
        3. {years_filter}
        4. Add LIMIT {config.DEFAULT_LIMIT} if not specified
        5. Return ONLY SQL code, no explanations
        
        EXAMPLE DATA (if helpful):
        {json.dumps(metadata.get('example_rows', [])[:2], ensure_ascii=False)}
        
        SQL QUERY:
        """
        
        return prompt