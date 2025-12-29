import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, Any

# Import dari SRC (Core Logic)
from src.config import config
from src.workflow import build_enhanced_workflow
from src.metadata_manager import MetadataManager
from src.tools import web_search_tool

# ================== CONFIGURATION ==================
st.set_page_config(
    page_title="BPS Agentic AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: bold; color: #0078D4; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .agent-step { padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 5px; border-left: 4px solid #0078D4; }
    .success-box { padding: 10px; background-color: #d1e7dd; border-radius: 5px; color: #0f5132; }
    .error-box { padding: 10px; background-color: #f8d7da; border-radius: 5px; color: #842029; }
</style>
""", unsafe_allow_html=True)

# ================== INITIALIZATION ==================

@st.cache_resource
def initialize_graph():
    """Build LangGraph Workflow sekali saja"""
    return build_enhanced_workflow()

@st.cache_resource
def get_db_stats():
    """Load statistik database untuk sidebar"""
    mm = MetadataManager()
    tables = mm.load_all_metadata()
    return {
        "table_count": len(tables),
        "table_names": list(tables.keys())
    }

# ================== MAIN UI ==================

def main():
    st.markdown('<div class="main-header">🤖 BPS Agentic AI System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Agent System: SQL Generator + Forecasting + Web Search</div>', unsafe_allow_html=True)

    # 1. Sidebar Info
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Cek Database
        if config.DB_PATH.exists():
            st.success(f"Database Connected: `{config.DB_NAME}`")
            try:
                stats = get_db_stats()
                with st.expander(f"📚 {stats['table_count']} Tables Indexed"):
                    st.write(stats['table_names'])
            except Exception as e:
                st.error(f"Metadata Error: {e}")
        else:
            st.error(f"❌ Database Not Found: {config.DB_PATH}")
            st.warning("Pastikan file .db ada di root folder.")

        # Cek Tavily
        if web_search_tool.is_active:
            st.success("🌐 Tavily Search: Active")
        else:
            st.warning("⚠️ Tavily Search: Inactive (Check API Key)")

        st.markdown("---")
        st.info(f"**Model:** {config.USER_MODEL}\n\n**Context:**\n- {config.USER_CONTEXT['region']}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.rerun()

    # 2. Chat Logic
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Init Graph
    try:
        graph = initialize_graph()
    except Exception as e:
        st.error(f"Failed to initialize workflow: {e}")
        return

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Jika itu pesan assistant dan punya 'details' (log agent), tampilkan di expander
            if msg.get("details"):
                with st.expander("🕵️ Agent Thought Process"):
                    for step in msg["details"]:
                        st.markdown(f"- {step}")
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Tanyakan data inflasi, ekspor, atau prediksi masa depan..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run Agent System
        with st.chat_message("assistant"):
            logs_container = st.empty()
            response_container = st.empty()
            
            agent_logs = []
            final_response = ""
            
            # Prepare Input State
            initial_state = {
                "user_input": prompt,
                "user_context": config.USER_CONTEXT,
                "messages": []
            }

            try:
                # 
                # Menjalankan graph secara streaming (per node update)
                with st.status("🤖 AI Agents working...", expanded=True) as status:
                    
                    # Kita stream output dari setiap node
                    for event in graph.stream(initial_state):
                        for key, value in event.items():
                            node_name = key
                            state_snapshot = value
                            
                            # Logging UI berdasarkan Node yang aktif
                            if node_name == "router":
                                intent = state_snapshot.get('intent', 'unknown').upper()
                                status.update(label=f"🔄 Routing Intent: {intent}", state="running")
                                log_msg = f"**Router:** Detected intent `{intent}`"
                                st.write(log_msg)
                                agent_logs.append(log_msg)
                                
                            elif node_name == "metadata_retriever" or node_name == "enhanced_metadata_retriever":
                                table = state_snapshot.get('selected_table', 'None')
                                status.update(label=f"📚 Checking Metadata...", state="running")
                                if table:
                                    log_msg = f"**Retriever:** Selected table `{table}`"
                                    st.write(log_msg)
                                    agent_logs.append(log_msg)
                            
                            elif node_name == "sql_agent" or node_name == "enhanced_sql_agent":
                                status.update(label="💻 Generating SQL...", state="running")
                                sql = state_snapshot.get('validated_sql')
                                if sql:
                                    st.code(sql, language="sql")
                                    agent_logs.append(f"**SQL Agent:** Generated SQL Query")
                                    
                            elif node_name == "sql_executor":
                                status.update(label="🗄️ Executing Database Query...", state="running")
                                res = state_snapshot.get('execution_result', {})
                                if res.get('success'):
                                    rows = res.get('row_count', 0)
                                    log_msg = f"**Executor:** Retrieved {rows} rows"
                                    st.write(log_msg)
                                    agent_logs.append(log_msg)
                                else:
                                    st.error(f"Executor Error: {res.get('error')}")
                                    
                            elif node_name == "forecast_agent" or node_name == "enhanced_forecast_agent":
                                status.update(label="📈 Calculating Forecast...", state="running")
                                res = state_snapshot.get('forecast_result', {})
                                if res.get('success'):
                                    method = res['forecast'].get('method')
                                    log_msg = f"**Forecaster:** Predicted using `{method}`"
                                    st.write(log_msg)
                                    agent_logs.append(log_msg)
                                    
                            elif node_name == "clarify_agent":
                                # Cek apakah ini Web Search
                                if not state_snapshot.get("needs_clarification"):
                                    status.update(label="🌐 Searching Internet (Tavily)...", state="running")
                                    st.write("**Clarify Agent:** Performing Web Search...")
                                    agent_logs.append("**Clarify Agent:** Searching Internet via Tavily")

                    status.update(label="✅ Process Complete", state="complete", expanded=False)

                # Ambil hasil akhir dari state terakhir
                # Note: 'value' di akhir loop adalah state terakhir
                final_state = value
                
                if final_state.get("final_answer"):
                    final_response = final_state["final_answer"]
                else:
                    final_response = "Maaf, terjadi kesalahan internal. Tidak ada jawaban akhir."

                # Tampilkan Jawaban
                response_container.markdown(final_response)

                # Simpan ke history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "details": agent_logs
                })

            except Exception as e:
                st.error(f"Workflow Error: {str(e)}")
                # Jika error print traceback ke console untuk debug
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()