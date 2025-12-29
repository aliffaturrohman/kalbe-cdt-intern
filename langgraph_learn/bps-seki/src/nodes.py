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
from .forecast_agent import EnhancedForecastAgent, SimpleForecastAgent
from .tools import web_search_tool  # <--- IMPORT BARU

# Inisialisasi Service
logger = AuditLogger()
metadata_manager = MetadataManager()
sql_executor = SQLExecutor()
smart_selector = SmartTableSelector()
enhanced_forecast_agent = EnhancedForecastAgent(llm_client)
simple_forecast_agent = SimpleForecastAgent()

# --- Basic Nodes ---

def router_node(state: AgentState) -> AgentState:
    """Node 1: Router - Intent detection"""
    logger.log("NODE_ENTER", {"node": "router", "input": state["user_input"]})
    
    user_input = state["user_input"].lower()
    
    forecast_keywords = ["prediksi", "forecast", "ramal", "estimasi", "proyeksi", "tren", "masa depan"]
    sql_keywords = ["tampilkan", "lihat", "berapa", "total", "jumlah", "data", "select", "daftar", "statistik"]
    
    if any(keyword in user_input for keyword in forecast_keywords):
        state["intent"] = "forecast"
    elif any(keyword in user_input for keyword in sql_keywords):
        state["intent"] = "sql"
    else:
        # Jika tidak mengandung keyword data eksplisit, masuk ke clarify/general chat
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
    
    # Jika intent awal adalah clarify/general question, skip pencarian tabel database
    # agar langsung ditangani oleh Web Search di clarify_agent_node
    if state.get("intent") == "clarify":
        state["next_node"] = "clarify_agent"
        return state

    # Find relevant tables
    relevant_tables = metadata_manager.find_relevant_tables(state["user_input"], top_k=5)
    state["relevant_tables"] = relevant_tables
    
    if not relevant_tables:
        # Jika tidak ketemu tabel, jangan langsung error/minta klarifikasi tabel.
        # Tapi lempar ke clarify_agent untuk dicoba cari di WEB.
        state["needs_clarification"] = False 
        state["next_node"] = "clarify_agent"
        
        logger.log("METADATA_NOT_FOUND", {
            "action": "Fallback to Web Search (Clarify Agent)"
        })
        return state
    
    # Auto-table selection dengan LLM
    selection_result = smart_selector.select_best_table(
        user_query=state["user_input"],
        candidate_tables=relevant_tables,
        user_context=state.get("user_context", {})
    )
    
    selected_table = selection_result.get("selected")
    
    # Ambang batas confidence 0.3
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
        # Fallback: Ambil yang relevance score-nya paling tinggi
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
        # Coba construct table info manual jika tidak ada di relevant_tables (fallback)
        meta = metadata_manager.get_table_metadata(state["selected_table"])
        if meta:
             table_info = {"table_name": state["selected_table"], "metadata": meta}
        else:
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
    
    # Bersihkan markdown syntax (```sql ... ```)
    raw_sql = response["content"].strip()
    raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    
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
        "sql_preview": raw_sql
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
    """
    logger.log("NODE_ENTER", {"node": "metadata_retriever_basic"})
    
    if state.get("intent") == "clarify":
        state["next_node"] = "clarify_agent"
        return state
    
    # Cari tabel relevan (hanya top 3 untuk simplicity)
    relevant_tables = metadata_manager.find_relevant_tables(
        state["user_input"], 
        top_k=3
    )
    
    state["relevant_tables"] = relevant_tables
    
    if not relevant_tables:
        # Fallback ke Web Search jika tidak ada tabel
        state["needs_clarification"] = False
        state["next_node"] = "clarify_agent"
        
        logger.log("METADATA_BASIC_NO_TABLES", {
            "message": "Fallback to Web Search"
        }, level="WARNING")
        
        return state
    
    elif len(relevant_tables) == 1:
        # BASIC: Jika hanya 1 tabel, langsung pilih
        selected_table = relevant_tables[0]
        state["selected_table"] = selected_table["table_name"]
        state["table_metadata"] = selected_table["metadata"]
        state["next_node"] = "planner"
        
        logger.log("METADATA_BASIC_AUTO_SELECT", {
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
            "tables_found": len(relevant_tables),
            "table_names": [t["table_name"] for t in relevant_tables]
        })
    
    return state

def sql_agent_node_basic(state: AgentState) -> AgentState:
    """BASIC VERSION: SQL generation sederhana"""
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
        # Fallback load manual
        meta = metadata_manager.get_table_metadata(state["selected_table"])
        if meta:
             table_info = {"table_name": state["selected_table"], "metadata": meta}
        else:
            state["error"] = f"Tabel {state['selected_table']} tidak ditemukan"
            state["next_node"] = "error_handler"
            return state
    
    schema_text = metadata_manager.build_schema_prompt(table_info)
    default_limit = getattr(config, 'DEFAULT_LIMIT', 5)

    prompt = f"""
    Buatkan query SQL untuk pertanyaan berikut:
    
    Pertanyaan user: {state['user_input']}
    
    Informasi tabel:
    {schema_text}
    
    Aturan:
    1. Hanya gunakan SELECT statement
    2. Filter berdasarkan region jika ada access column
    3. Batasi hasil dengan LIMIT {default_limit} jika tidak disebutkan
    4. Return HANYA kode SQL, tanpa penjelasan atau markdown formatting.
    
    SQL Query:
    """
    
    response = llm_client.call_sql_llm(prompt)
    
    if not response["success"]:
        state["error"] = f"SQL generation failed: {response.get('error')}"
        state["next_node"] = "error_handler"
        return state
    
    raw_sql = response["content"].strip()
    raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
    
    validation = SQLValidator.validate_sql(raw_sql)
    if not validation["is_valid"]:
        state["error"] = f"SQL validation failed: {validation['reason']}"
        state["next_node"] = "error_handler"
        return state
    
    access_column = table_info["metadata"].get("access_column")
    if access_column and state.get("user_context", {}).get("region"):
        raw_sql = SQLValidator.inject_region_filter(
            raw_sql, 
            access_column, 
            state["user_context"]["region"]
        )
    
    raw_sql = SQLValidator.add_limit_if_missing(raw_sql)
    
    state["raw_sql"] = raw_sql
    state["validated_sql"] = raw_sql
    state["next_node"] = "sql_executor"
    
    logger.log("SQL_BASIC_GENERATED", {
        "table": state["selected_table"],
        "sql_preview": raw_sql
    }, level="SUCCESS")
    
    return state

def forecast_agent_node_basic(state: AgentState) -> AgentState:
    """BASIC VERSION: Forecasting sederhana"""
    logger.log("NODE_ENTER", {"node": "forecast_agent_basic"})
    
    if not state.get("selected_table"):
        state["error"] = "Tidak ada tabel yang dipilih untuk forecasting"
        state["next_node"] = "error_handler"
        return state
    
    table_name = state["selected_table"]
    table_meta = state.get("table_metadata", {})
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
    
    # Fallback
    if not date_candidates:
        date_candidates = list(columns.keys())
    
    if not value_candidates and len(columns) > 1:
        value_candidates = [col for col in list(columns.keys()) if col != date_candidates[0]]
    
    date_col = date_candidates[0] if date_candidates else list(columns.keys())[0]
    value_col = value_candidates[0] if value_candidates else list(columns.keys())[1] if len(columns) > 1 else list(columns.keys())[0]
    
    sql = f"SELECT {date_col}, {value_col} FROM {table_name}"
    
    access_column = table_meta.get("access_column")
    if access_column and state.get("user_context", {}).get("region"):
        sql += f" WHERE {access_column} LIKE '{state['user_context']['region']}%'"
    
    sql += f" ORDER BY {date_col}"
    
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
    min_data = getattr(config, 'MIN_DATA_POINTS', 3)

    if len(df) < min_data:
        state["error"] = f"Data tidak cukup untuk forecasting. Minimal {min_data} baris, dapat {len(df)}"
        state["next_node"] = "error_handler"
        return state
    
    try:
        forecast_result = simple_forecast_agent.linear_forecast(
            df=df,
            date_column=date_col,
            value_column=value_col,
            periods=3
        )
        
        if not forecast_result["success"]:
            state["error"] = forecast_result.get("error", "Forecast failed")
            state["next_node"] = "error_handler"
            return state
        
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
    """
    Node Clarify / General Chat dengan Tavily Web Search.
    Menangani 2 kondisi:
    1. Disambiguasi Tabel (user perlu memilih tabel).
    2. General Question (data tidak ada di DB, cari di web).
    """
    logger.log("NODE_ENTER", {"node": "clarify_agent"})
    
    # KONDISI 1: User merespon klarifikasi tabel sebelumnya
    if state.get("clarification_response"):
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
    
    # KONDISI 2: System butuh user memilih tabel (belum dijawab)
    if state.get("needs_clarification") and state.get("clarification_question"):
        state["final_answer"] = state["clarification_question"]
        state["next_node"] = "end"
        
    # KONDISI 3: General Question / Web Search
    else:
        # Jika masuk sini, artinya:
        # a. Intent awal adalah 'clarify' (bukan SQL/Forecast)
        # b. Metadata retriever tidak menemukan tabel (empty result)
        
        user_query = state['user_input']
        logger.log("WEB_SEARCH_INIT", {"query": user_query})
        
        # 1. Cari Informasi di Web via Tavily
        web_results = web_search_tool.search(user_query)
        
        # 2. Minta LLM menjawab berdasarkan hasil web
        prompt = f"""
        Anda adalah asisten AI yang membantu menjawab pertanyaan pengguna.
        
        PERTANYAAN PENGGUNA:
        "{user_query}"
        
        HASIL PENCARIAN WEB (TAVILY):
        {web_results}
        
        INSTRUKSI:
        1. Jawab pertanyaan pengguna dalam Bahasa Indonesia yang baik dan benar.
        2. Gunakan informasi dari hasil pencarian web di atas sebagai referensi utama.
        3. Jika hasil pencarian web tidak relevan, gunakan pengetahuan umum Anda, namun beritahu pengguna bahwa data spesifik tidak ditemukan.
        4. Jangan menyebutkan "berdasarkan hasil pencarian tavily", gunakan gaya bahasa natural seperti "Berdasarkan informasi terkini..."
        
        JAWABAN:
        """
        
        response = llm_client.call_user_llm(prompt)
        
        if response["success"]:
            state["final_answer"] = response['content']
            logger.log("WEB_SEARCH_SUCCESS", {"query": user_query})
        else:
            state["error"] = "Gagal menghasilkan jawaban dari Web Search."
        
        state["next_node"] = "end"
    
    return state

def response_formatter_node(state: AgentState) -> AgentState:
    """Format final response untuk user dengan bantuan LLM"""
    logger.log("NODE_ENTER", {"node": "response_formatter"})
    
    try:
        # --- KASUS 1: Hasil dari SQL Executor ---
        if state.get("execution_result"):
            result = state["execution_result"]
            df = result.get("data")
            
            # Jika data kosong
            if df is None or df.empty:
                response = "✅ **HASIL QUERY**\n\n"
                response += f"Tabel: {state.get('selected_table', 'Unknown')}\n"
                response += "Query berhasil dieksekusi tetapi tidak ada data yang ditemukan sesuai kriteria filter Anda."
                
                # Jika ada SQL, tampilkan untuk debug
                if state.get("validated_sql"):
                    response += f"\n\n**Query SQL:**\n```sql\n{state.get('validated_sql')}\n```"
                
                state["final_answer"] = response
            
            else:
                # Jika ada data, minta LLM untuk menjelaskan
                logger.log("FORMATTING_WITH_LLM", {"rows": len(df)})
                
                # Konversi data ke string CSV/Markdown untuk prompt
                data_preview = df.head(10).to_markdown(index=False)
                
                user_query = state['user_input']
                table_name = state.get('selected_table', 'Unknown')
                
                prompt = f"""
                Anda adalah Data Analyst expert. Tugas Anda adalah menjelaskan data hasil query database kepada pengguna.
                
                PERTANYAAN PENGGUNA:
                "{user_query}"
                
                SUMBER DATA (Tabel: {table_name}):
                {data_preview}
                
                INSTRUKSI:
                1. Jawab pertanyaan pengguna berdasarkan data di atas.
                2. Berikan analisis singkat atau highlight (misal: tren, nilai tertinggi/terendah).
                3. WAJIB: Sertakan ulang tabel data di atas dalam format Markdown agar pengguna bisa melihat detailnya.
                4. Gunakan bahasa Indonesia yang profesional dan mudah dimengerti.
                
                JAWABAN:
                """
                
                # Panggil LLM
                response = llm_client.call_user_llm(prompt)
                
                if response["success"]:
                    state["final_answer"] = response['content']
                else:
                    # Fallback jika LLM gagal format
                    state["final_answer"] = f"Berikut data yang ditemukan:\n\n{data_preview}\n\n(Gagal membuat narasi penjelasan)"

        # --- KASUS 2: Hasil dari Forecast Agent ---
        elif state.get("forecast_result"):
            forecast_data = state["forecast_result"]["forecast"]
            
            # Siapkan data prediksi untuk LLM
            preds = forecast_data.get("predictions", [])
            pred_text = "\n".join([f"- {p['period']}: {p['prediction']:.2f}" for p in preds])
            
            prompt = f"""
            Anda adalah Data Analyst. Jelaskan hasil prediksi forecasting berikut kepada pengguna.
            
            DATA FORECASTING (Metode: {forecast_data.get('method')}):
            {pred_text}
            
            Info Tambahan:
            - Tabel: {forecast_data.get('table_name')}
            - Data points history: {forecast_data.get('data_points')}
            
            INSTRUKSI:
            1. Jelaskan tren prediksi (naik/turun/stabil).
            2. Sebutkan angka prediksi untuk periode terakhir.
            3. Buatkan tabel markdown ringkas dari hasil prediksi tersebut.
            """
            
            response = llm_client.call_user_llm(prompt)
            if response["success"]:
                state["final_answer"] = response['content']
            else:
                # Fallback manual formatting
                resp_text = f"📈 **HASIL FORECASTING**\n\n"
                resp_text += f"Metode: {forecast_data['method']}\n\n"
                for p in preds:
                    resp_text += f"- {p['period']}: {p['prediction']:.2f}\n"
                state["final_answer"] = resp_text

        # --- KASUS 3: Tidak ada hasil (Error atau Clarify) ---
        else:
            # Biasanya sudah dihandle di node lain, tapi buat jaga-jaga
            if not state.get("final_answer"):
                state["final_answer"] = "Maaf, tidak ada data yang dapat ditampilkan saat ini."
        
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