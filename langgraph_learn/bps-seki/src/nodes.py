# src/nodes.py
"""All LangGraph node definitions"""

import json
import re
from typing import Dict, Any

from .state import AgentState
from .config import config
from .logger import AuditLogger
from .metadata_manager import MetadataManager
from .sql_validator import SQLValidator
from .sql_executor import SQLExecutor
from .llm_client import llm_client
from .smart_selector import SmartTableSelector
from .forecast_agent import EnhancedForecastAgent, SimpleForecastAgent  # Tambah import

logger = AuditLogger()
metadata_manager = MetadataManager()
sql_executor = SQLExecutor()
smart_selector = SmartTableSelector()
enhanced_forecast_agent = EnhancedForecastAgent(llm_client)
simple_forecast_agent = SimpleForecastAgent()  # Tambah instance untuk basic version

# Basic Nodes
def router_node(state: AgentState) -> AgentState:
    """Node 1: Router - Intent detection"""
    logger.log("NODE_ENTER", {"node": "router", "input": state["user_input"]})
    
    user_input = state["user_input"].lower()
    
    forecast_keywords = ["prediksi", "forecast", "ramal", "estimasi", "proyeksi"]
    sql_keywords = ["tampilkan", "lihat", "berapa", "total", "jumlah", "data", "select"]
    
    if any(keyword in user_input for keyword in forecast_keywords):
        state["intent"] = "forecast"
    elif any(keyword in user_input for keyword in sql_keywords):
        state["intent"] = "sql"
    else:
        state["intent"] = "clarify"
    
    state["next_node"] = "metadata_retriever"
    
    logger.log("ROUTER_DECISION", {
        "intent": state["intent"],
        "user_input": state["user_input"]
    })
    
    return state

def enhanced_metadata_retriever_node(state: AgentState) -> AgentState:
    """Enhanced: Auto-table selection dengan LLM"""
    logger.log("NODE_ENTER", {"node": "enhanced_metadata_retriever"})
    
    # Find relevant tables
    relevant_tables = metadata_manager.find_relevant_tables(state["user_input"], top_k=5)
    state["relevant_tables"] = relevant_tables
    
    if not relevant_tables:
        state["needs_clarification"] = True
        state["clarification_question"] = (
            "Maaf, saya tidak menemukan data yang relevan. "
            "Bisa Anda gunakan kata kunci yang lebih spesifik?"
        )
        state["next_node"] = "clarify_agent"
        return state
    
    # Auto-table selection dengan LLM
    selection_result = smart_selector.select_best_table(
        user_query=state["user_input"],
        candidate_tables=relevant_tables,
        user_context=state.get("user_context", {})
    )
    
    selected_table = selection_result.get("selected")
    
    if selected_table and selection_result.get("confidence", 0) > 0.3:
        state["selected_table"] = selected_table["table_name"]
        state["table_metadata"] = selected_table["metadata"]
        state["selection_confidence"] = selection_result["confidence"]
        state["selection_reason"] = selection_result.get("reason", "")
        state["next_node"] = "planner"
        
        logger.log("AUTO_TABLE_SELECTED", {
            "user_query": state["user_input"],
            "selected_table": selected_table["table_name"],
            "confidence": selection_result["confidence"],
            "reason": selection_result.get("reason", "")
        }, level="SUCCESS")
    else:
        # Fallback
        best_table = max(relevant_tables, key=lambda x: x.get("relevance_score", 0))
        state["selected_table"] = best_table["table_name"]
        state["table_metadata"] = best_table["metadata"]
        state["next_node"] = "planner"
    
    return state

def planner_node(state: AgentState) -> AgentState:
    """Node 3: Planner - Tentukan langkah berikutnya"""
    logger.log("NODE_ENTER", {"node": "planner"})
    
    intent = state.get("intent", "sql")
    
    if intent == "forecast":
        state["next_node"] = "forecast_agent"
    elif intent == "sql":
        state["next_node"] = "sql_agent"
    else:
        state["next_node"] = "clarify_agent"
    
    logger.log("PLANNER_DECISION", {
        "intent": intent,
        "next_node": state["next_node"]
    })
    
    return state

def enhanced_sql_agent_node(state: AgentState) -> AgentState:
    """Enhanced SQL Agent dengan smart generation"""
    logger.log("NODE_ENTER", {"node": "enhanced_sql_agent"})
    
    if not state.get("selected_table"):
        state["error"] = "No table selected"
        state["next_node"] = "error_handler"
        return state
    
    table_info = next(
        (t for t in state.get("relevant_tables", []) if t["table_name"] == state["selected_table"]),
        None
    )
    
    if not table_info:
        state["error"] = f"Table {state['selected_table']} not found"
        state["next_node"] = "error_handler"
        return state
    
    # Build smart prompt
    prompt = smart_selector.build_smart_sql_prompt(
        user_query=state["user_input"],
        table_info=table_info,
        user_context=state.get("user_context", {})
    )
    
    # Generate SQL dengan LLM
    response = llm_client.call_sql_llm(prompt)
    
    if not response["success"]:
        state["error"] = f"SQL generation failed: {response.get('error')}"
        state["next_node"] = "error_handler"
        return state
    
    raw_sql = response["content"].strip().strip("`")
    
    # Validate SQL
    validation = SQLValidator.validate_sql(raw_sql)
    if not validation["is_valid"]:
        state["error"] = f"SQL validation failed: {validation['reason']}"
        state["next_node"] = "error_handler"
        return state
    
    # Inject region filter
    access_column = table_info["metadata"].get("access_column")
    if access_column and state.get("user_context", {}).get("region"):
        raw_sql = SQLValidator.inject_region_filter(
            raw_sql, 
            access_column, 
            state["user_context"]["region"]
        )
    
    # Add LIMIT jika tidak ada
    raw_sql = SQLValidator.add_limit_if_missing(raw_sql)
    
    state["raw_sql"] = raw_sql
    state["validated_sql"] = raw_sql
    state["next_node"] = "sql_executor"
    
    logger.log("ENHANCED_SQL_GENERATED", {
        "table": state["selected_table"],
        "sql_preview": raw_sql[:200]
    }, level="SUCCESS")
    
    return state

def sql_executor_node(state: AgentState) -> AgentState:
    """Execute SQL query"""
    logger.log("NODE_ENTER", {"node": "sql_executor"})
    
    if not state.get("validated_sql"):
        state["error"] = "No SQL query to execute"
        state["next_node"] = "error_handler"
        return state
    
    result = sql_executor.execute(state["validated_sql"])
    
    if result["success"]:
        state["execution_result"] = result
        state["next_node"] = "response_formatter"
    else:
        state["error"] = result["error"]
        state["next_node"] = "error_handler"
    
    return state

def metadata_retriever_node_basic(state: AgentState) -> AgentState:
    """
    BASIC VERSION: Metadata retriever tanpa auto-selection.
    User perlu memilih tabel jika ada multiple candidates.
    """
    logger.log("NODE_ENTER", {"node": "metadata_retriever_basic"})
    
    # Cari tabel relevan (hanya top 3 untuk simplicity)
    relevant_tables = metadata_manager.find_relevant_tables(
        state["user_input"], 
        top_k=3  # Basic: hanya ambil 3 terbaik
    )
    
    state["relevant_tables"] = relevant_tables
    
    if not relevant_tables:
        # Tidak ada tabel yang ditemukan
        state["needs_clarification"] = True
        state["clarification_question"] = (
            "Maaf, saya tidak menemukan data yang sesuai dengan permintaan Anda. "
            "Bisa Anda jelaskan dengan kata kunci yang berbeda?"
        )
        state["next_node"] = "clarify_agent"
        
        logger.log("METADATA_BASIC_NO_TABLES", {
            "user_query": state["user_input"],
            "message": "No relevant tables found"
        }, level="WARNING")
        
        return state
    
    elif len(relevant_tables) == 1:
        # BASIC: Jika hanya 1 tabel, langsung pilih
        selected_table = relevant_tables[0]
        state["selected_table"] = selected_table["table_name"]
        state["table_metadata"] = selected_table["metadata"]
        state["next_node"] = "planner"
        
        logger.log("METADATA_BASIC_AUTO_SELECT", {
            "user_query": state["user_input"],
            "selected_table": selected_table["table_name"],
            "reason": "Only one relevant table found"
        })
        
    else:
        # BASIC: Jika multiple tables, minta user pilih
        table_options = "\n".join([
            f"{i+1}. {table['table_name']} (relevansi: {table.get('relevance_score', 0):.1f})"
            for i, table in enumerate(relevant_tables)
        ])
        
        state["needs_clarification"] = True
        state["clarification_question"] = (
            f"Saya menemukan {len(relevant_tables)} tabel yang mungkin relevan:\n\n"
            f"{table_options}\n\n"
            f"Tabel mana yang Anda maksud? (sebutkan nomor 1-{len(relevant_tables)})"
        )
        state["next_node"] = "clarify_agent"
        
        logger.log("METADATA_BASIC_NEED_SELECTION", {
            "user_query": state["user_input"],
            "tables_found": len(relevant_tables),
            "table_names": [t["table_name"] for t in relevant_tables]
        })
    
    return state

def sql_agent_node_basic(state: AgentState) -> AgentState:
    """
    BASIC VERSION: SQL generation sederhana tanpa smart features.
    Tidak ada tahun detection, hanya prompt dasar.
    """
    logger.log("NODE_ENTER", {"node": "sql_agent_basic"})
    
    if not state.get("selected_table"):
        state["error"] = "Tidak ada tabel yang dipilih"
        state["next_node"] = "error_handler"
        return state
    
    # Dapatkan metadata tabel
    table_info = next(
        (t for t in state.get("relevant_tables", []) if t["table_name"] == state["selected_table"]),
        None
    )
    
    if not table_info:
        state["error"] = f"Tabel {state['selected_table']} tidak ditemukan"
        state["next_node"] = "error_handler"
        return state
    
    # BASIC: Prompt sederhana tanpa tahun detection
    schema_text = metadata_manager.build_schema_prompt(table_info)
    
    prompt = f"""
    Buatkan query SQL untuk pertanyaan berikut:
    
    Pertanyaan user: {state['user_input']}
    
    Informasi tabel:
    {schema_text}
    
    Aturan:
    1. Hanya gunakan SELECT statement
    2. Filter berdasarkan region jika ada access column
    3. Batasi hasil dengan LIMIT {config.DEFAULT_LIMIT} jika tidak disebutkan
    4. Return HANYA kode SQL, tanpa penjelasan
    
    SQL Query:
    """
    
    # Generate SQL dengan LLM
    response = llm_client.call_sql_llm(prompt)
    
    if not response["success"]:
        state["error"] = f"SQL generation failed: {response.get('error')}"
        state["next_node"] = "error_handler"
        return state
    
    raw_sql = response["content"].strip().strip("`")
    
    # Validasi dasar
    validation = SQLValidator.validate_sql(raw_sql)
    if not validation["is_valid"]:
        state["error"] = f"SQL validation failed: {validation['reason']}"
        state["next_node"] = "error_handler"
        return state
    
    # BASIC: Inject region filter (sama seperti enhanced)
    access_column = table_info["metadata"].get("access_column")
    if access_column and state.get("user_context", {}).get("region"):
        raw_sql = SQLValidator.inject_region_filter(
            raw_sql, 
            access_column, 
            state["user_context"]["region"]
        )
    
    # Tambahkan LIMIT jika tidak ada
    raw_sql = SQLValidator.add_limit_if_missing(raw_sql)
    
    state["raw_sql"] = raw_sql
    state["validated_sql"] = raw_sql
    state["next_node"] = "sql_executor"
    
    logger.log("SQL_BASIC_GENERATED", {
        "table": state["selected_table"],
        "sql_preview": raw_sql[:200],
        "prompt_length": len(prompt)
    }, level="SUCCESS")
    
    return state

def forecast_agent_node_basic(state: AgentState) -> AgentState:
    """
    BASIC VERSION: Forecasting sederhana tanpa enhanced features.
    Hanya linear regression, tidak ada auto-detection.
    """
    logger.log("NODE_ENTER", {"node": "forecast_agent_basic"})
    
    if not state.get("selected_table"):
        state["error"] = "Tidak ada tabel yang dipilih untuk forecasting"
        state["next_node"] = "error_handler"
        return state
    
    table_name = state["selected_table"]
    table_meta = state.get("table_metadata", {})
    
    # BASIC: Deteksi kolom sederhana (tanpa LLM)
    columns = table_meta.get("columns", {})
    
    if not columns:
        state["error"] = "Tabel metadata tidak memiliki informasi kolom"
        state["next_node"] = "error_handler"
        return state
    
    # Cari kolom tanggal (heuristik sederhana)
    date_candidates = []
    for col_name in columns.keys():
        col_lower = col_name.lower()
        if any(keyword in col_lower for keyword in ["tahun", "year", "bulan", "month", "tanggal", "date"]):
            date_candidates.append(col_name)
    
    # Cari kolom nilai
    value_candidates = []
    for col_name in columns.keys():
        col_lower = col_name.lower()
        if any(keyword in col_lower for keyword in ["nilai", "value", "jumlah", "total", "qty", "quantity", "volume"]):
            value_candidates.append(col_name)
    
    # Fallback jika tidak ditemukan
    if not date_candidates:
        date_candidates = list(columns.keys())
    
    if not value_candidates and len(columns) > 1:
        # Hindari menggunakan kolom yang sama untuk tanggal dan nilai
        value_candidates = [col for col in list(columns.keys()) if col != date_candidates[0]]
    
    date_col = date_candidates[0] if date_candidates else list(columns.keys())[0]
    value_col = value_candidates[0] if value_candidates else list(columns.keys())[1] if len(columns) > 1 else list(columns.keys())[0]
    
    # Build SQL query sederhana
    sql = f"SELECT {date_col}, {value_col} FROM {table_name}"
    
    # Tambahkan filter region jika ada
    access_column = table_meta.get("access_column")
    if access_column and state.get("user_context", {}).get("region"):
        sql += f" WHERE {access_column} LIKE '{state['user_context']['region']}%'"
    
    sql += f" ORDER BY {date_col}"
    
    # Execute query
    result = sql_executor.execute(sql)
    
    if not result["success"]:
        state["error"] = result.get("error", "Unknown SQL error")
        state["next_node"] = "error_handler"
        return state
    
    if "data" not in result:
        state["error"] = "SQL tidak mengembalikan data"
        state["next_node"] = "error_handler"
        return state
    
    df = result["data"]
    
    if len(df) < config.MIN_DATA_POINTS:
        state["error"] = f"Data tidak cukup untuk forecasting. Minimal {config.MIN_DATA_POINTS} baris, dapat {len(df)}"
        state["next_node"] = "error_handler"
        return state
    
    try:
        # BASIC: Gunakan simple forecast agent
        forecast_result = simple_forecast_agent.linear_forecast(
            df=df,
            date_column=date_col,
            value_column=value_col,
            periods=3
        )
        
        if not forecast_result["success"]:
            state["error"] = forecast_result.get("error", "Forecast failed")
            state["next_node"] = "error_handler"
            return forecast_result
        
        state["forecast_result"] = forecast_result
        state["next_node"] = "response_formatter"
        
        logger.log("FORECAST_BASIC_SUCCESS", {
            "table": table_name,
            "data_points": len(df),
            "method": "linear_regression_basic"
        }, level="SUCCESS")
        
    except Exception as e:
        state["error"] = f"Forecast basic failed: {str(e)}"
        state["next_node"] = "error_handler"
        logger.log("FORECAST_BASIC_ERROR", {
            "error": str(e),
            "table": table_name
        }, level="ERROR")
    
    return state

def clarify_agent_node(state: AgentState) -> AgentState:
    """Handle user clarification"""
    logger.log("NODE_ENTER", {"node": "clarify_agent"})
    
    if state.get("clarification_response"):
        # Process user response
        response = state["clarification_response"].strip()
        
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(state.get("relevant_tables", [])):
                selected = state["relevant_tables"][idx]
                state["selected_table"] = selected["table_name"]
                state["table_metadata"] = selected["metadata"]
                state["needs_clarification"] = False
                state["clarification_question"] = None
                state["clarification_response"] = None
                state["next_node"] = "planner"
                return state
    
    # Ask for clarification
    if state.get("needs_clarification") and state.get("clarification_question"):
        state["final_answer"] = state["clarification_question"]
        state["next_node"] = "end"
    else:
        # General clarification
        prompt = f"User query: {state['user_input']}. This is unclear. Ask for clarification in Bahasa Indonesia."
        response = llm_client.call_user_llm(prompt)  # Ganti dari call_user_llm ke call_general_llm
        
        if response["success"]:
            state["final_answer"] = response['content']
        else:
            state["error"] = "Failed to generate clarification"
        
        state["next_node"] = "end"
    
    return state

def response_formatter_node(state: AgentState) -> AgentState:
    """Format final response untuk user"""
    logger.log("NODE_ENTER", {"node": "response_formatter"})
    
    try:
        if state.get("execution_result"):
            # Format SQL result
            result = state["execution_result"]
            df = result["data"]
            
            if df.empty:
                response = "✅ **HASIL QUERY**\n\n"
                response += f"Tabel: {state.get('selected_table', 'Unknown')}\n"
                response += "Query berhasil dieksekusi tetapi tidak ada data yang ditemukan."
            else:
                response = f"✅ **HASIL QUERY**\n\n"
                response += f"Tabel: {state.get('selected_table', 'Unknown')}\n"
                response += f"Jumlah baris: {len(df)}\n\n"
                response += f"**Data (5 baris pertama):**\n"
                response += df.head().to_string(index=False)
                
                # Tambahkan query SQL jika ada
                if state.get("validated_sql"):
                    response += f"\n\n**Query SQL:**\n```sql\n{state.get('validated_sql', 'N/A')}\n```"
        
        elif state.get("forecast_result"):
            # Format forecast result
            forecast_data = state["forecast_result"]["forecast"]
            
            response = f"📈 **HASIL FORECASTING**\n\n"
            response += f"Tabel: {forecast_data['table_name']}\n"
            response += f"Data points: {forecast_data['data_points']}\n"
            response += f"Metode: {forecast_data['method']}\n\n"
            response += f"**Prediksi {forecast_data['forecast_periods']} periode:**\n"
            
            for i, pred in enumerate(forecast_data["predictions"]):
                response += f"{i+1}. {pred.get('period', f'Periode {i+1}')}: {pred.get('prediction', 0):.2f}\n"
            
            # Tambahkan metadata jika ada
            if forecast_data.get("metadata"):
                meta = forecast_data["metadata"]
                response += f"\n**Detail:**\n"
                response += f"- Kolom tanggal: {meta.get('date_column', 'N/A')}\n"
                response += f"- Kolom nilai: {meta.get('value_column', 'N/A')}\n"
        
        else:
            response = "Tidak ada hasil yang dapat ditampilkan."
        
        state["final_answer"] = response
        state["next_node"] = "end"
        
    except Exception as e:
        state["error"] = f"Response formatting error: {str(e)}"
        state["next_node"] = "error_handler"
    
    return state

def error_handler_node(state: AgentState) -> AgentState:
    """Handle errors gracefully"""
    logger.log("NODE_ENTER", {"node": "error_handler"})
    
    error_msg = state.get("error", "Unknown error")
    
    response = f"⚠️ **ERROR**\n\n{error_msg}\n\n"
    response += f"**Query:** {state.get('user_input', 'N/A')}\n"
    response += f"**Table:** {state.get('selected_table', 'N/A')}\n\n"
    response += "Silakan coba lagi dengan query yang lebih spesifik."
    
    state["final_answer"] = response
    state["next_node"] = "end"
    
    logger.log("ERROR_HANDLED", {
        "error": error_msg,
        "user_input": state.get("user_input")
    }, level="ERROR")
    
    return state

def end_node(state: AgentState) -> AgentState:
    """Node akhir"""
    logger.log("NODE_ENTER", {"node": "end"})
    logger.log("WORKFLOW_COMPLETE", {
        "user_input": state.get("user_input", "No input"),
        "has_final_answer": state.get("final_answer") is not None,
        "has_error": state.get("error") is not None,
        "selected_table": state.get("selected_table")
    })
    return state