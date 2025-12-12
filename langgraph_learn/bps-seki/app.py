import streamlit as st
import os
import pandas as pd
from typing import List, Dict
from sqlalchemy import create_engine, inspect

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

# ================== CONFIGURATION ==================
st.set_page_config(
    page_title="BPS Data Chatbot (Hybrid)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)
engine = create_engine(f"sqlite:///{db_path}") #define engine diluar
insp = inspect(engine)
# ================== DATABASE & DOC FUNCTIONS ==================

@st.cache_resource
def load_all_tables(db_path: str) -> Dict[str, pd.DataFrame]:
    """Load semua tabel dari database SQLite"""
    
    tables = {}
    for name in insp.get_table_names():
        df = pd.read_sql_table(name, engine)
        tables[name] = df
    return tables

def create_table_metadata_docs(tables: Dict[str, pd.DataFrame]) -> List[Document]:
    """Buat dokumen metadata schema tabel"""
    docs = []
    for table_name, df in tables.items():
        metadata = {"type": "table_metadata", "table": table_name, "row_count": len(df)}
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            if unique_count < 50:
                sample_values = df[col].unique()[:10].tolist()
                columns_info.append(f"- {col} ({dtype}): {unique_count} unique. Ex: {sample_values}")
            else:
                columns_info.append(f"- {col} ({dtype}): {unique_count} unique")
        
        content = f"Table: {table_name}\nRows: {len(df)}\nColumns:\n" + "\n".join(columns_info)
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def create_table_summary_docs(tables: Dict[str, pd.DataFrame]) -> List[Document]:
    """Buat dokumen summary statistik sederhana"""
    docs = []
    for table_name, df in tables.items():
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if categorical_cols and len(categorical_cols) <= 3:
            group_cols = categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols[:1]
            try:
                grouped = df.groupby(group_cols)
                for group_key, group_df in grouped:
                    if len(group_df) > 0:
                        group_desc = str(group_key)
                        stats = []
                        for num_col in numeric_cols:
                            if num_col in group_df.columns:
                                mean_val = group_df[num_col].mean()
                                if pd.notna(mean_val):
                                    stats.append(f"{num_col} avg: {mean_val:.2f}")
                        
                        content = f"Table: {table_name}\nFilter: {group_cols}={group_desc}\nStats: {', '.join(stats)}"
                        docs.append(Document(page_content=content, metadata={"type": "summary", "table": table_name}))
            except Exception:
                continue
    return docs

def create_row_docs(tables: Dict[str, pd.DataFrame], sample_ratio: float = 0.3) -> List[Document]:
    """Buat dokumen sampel baris data"""
    docs = []
    for table_name, df in tables.items():
        sampled_df = df.sample(n=int(len(df) * sample_ratio), random_state=42) if len(df) > 100 else df
        for idx, row in sampled_df.iterrows():
            fields = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and val != '']
            content = f"Table: {table_name} Data Row:\n" + " | ".join(fields[:15])
            docs.append(Document(page_content=content, metadata={"type": "row_data", "table": table_name}))
    return docs

def format_docs(docs):
    """Format dokumen retrieved dari Vectorstore"""
    return "\n\n".join([d.page_content for d in docs])

def format_tavily_results(results):
    """Format hasil pencarian Tavily menjadi string bersih"""
    if not results: return "No web results found."
    if isinstance(results, str): return results
    formatted = []
    for r in results:
        content = r.get('content', '')[:300] + "..." 
        url = r.get('url', '')
        formatted.append(f"- {content} (Source: {url})")
    return "\n\n".join(formatted)

# ================== SYSTEM INITIALIZATION ==================

@st.cache_resource
def initialize_system(db_path: str, tavily_api_key: str = None):
    """Initialize Hybrid RAG System"""
    
    # 1. Load Local DB
    tables = load_all_tables(db_path)
    all_docs = (
        create_table_metadata_docs(tables) + 
        create_table_summary_docs(tables) + 
        create_row_docs(tables, sample_ratio=0.5)
    )
    
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(all_docs, emb)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    # 2. Setup Tools
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        tavily_tool = TavilySearchResults(max_results=3) 
    else:
        tavily_tool = None

    # Setup LLM
    # Pastikan model Ollama kamu support streaming (qwen/gemma biasanya support)
    llm = OllamaLLM(model="gemma3:latest") 
    
    # 3. Construct Chain
    prompt = ChatPromptTemplate.from_template("""
    You are an expert Data Assistant for BPS (Badan Pusat Statistik) Indonesia.
    You have access to two sources:
    
    1. INTERNAL DATABASE (Primary Source): Official statistics, exact numbers, table structures.
    2. INTERNET SEARCH (Secondary Source): Definitions, news, reasons, comparisons.

    === DATABASE CONTEXT ===
    {db_context}

    === INTERNET CONTEXT ===
    {web_context}

    INSTRUCTIONS:
    - Answer in Indonesian.
    - If asked for data/numbers available in the database, USE THE DATABASE values precisely.
    - Use the Internet Context to explain "Why" or provide recent news not in the database.
    - Always cite the table name if using database data.
    - If the database is empty on the topic, rely on the Internet Context but mention it comes from the web.

    Question: {question}
    Answer:
    """)
    
    def run_web_search(query):
        if not tavily_tool:
            return "Web search disabled (No API Key)."
        try:
            res = tavily_tool.invoke(query)
            return format_tavily_results(res)
        except Exception as e:
            return f"Web search failed: {str(e)}"

    qa_chain = (
        {
            "db_context": lambda x: format_docs(retriever.invoke(x["question"])),
            "web_context": lambda x: run_web_search(x["question"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
    )
    
    stats = {
        "tables": list(tables.keys()),
        "doc_count": len(all_docs)
    }
    
    return qa_chain, stats

# ================== MAIN APP UI (STREAMING ENABLED) ==================

def main():
    st.markdown('<div class="main-header">📊 BPS Data Chatbot (Hybrid)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Database Lokal + Internet Search (Tavily)</div>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state: st.session_state.messages = []
    if "qa_chain" not in st.session_state: st.session_state.qa_chain = None
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        db_path = st.text_input("📁 Database Path (.db)", value="bps_seki.db")
        st.markdown("---")
        tavily_key = st.text_input("🌐 Tavily API Key", type="password", help="Get free key at tavily.com")
        
        if st.button("🚀 Initialize System", type="primary"):
            if not os.path.exists(db_path):
                st.error(f"File {db_path} tidak ditemukan!")
            else:
                with st.spinner("Indexing Database & Connecting to Tools..."):
                    try:
                        qa_chain, stats = initialize_system(db_path, tavily_key)
                        st.session_state.qa_chain = qa_chain
                        st.session_state.stats = stats
                        
                        st.success("System Ready!")
                        if tavily_key:
                            st.info("✅ Internet Access: ENABLED")
                        else:
                            st.warning("⚠️ Internet Access: DISABLED (No Key)")
                            
                    except Exception as e:
                        st.error(f"Init Error: {str(e)}")

        if "stats" in st.session_state and st.session_state.stats:
            st.markdown("---")
            st.subheader("📊 Data Stats")
            st.write(f"**Indexed Docs:** {st.session_state.stats['doc_count']}")
            with st.expander("View Tables"):
                st.write(st.session_state.stats['tables'])
                
        st.markdown("---")
        if st.button("🗑️ Reset Chat"):
            st.session_state.messages = []
            st.rerun()

    if st.session_state.qa_chain is None:
        st.info("👈 Silakan masukkan Database Path & Tavily API Key di sidebar, lalu klik Initialize.")
        return

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Tanyakan data BPS atau info terkait..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant Response (STREAMING)
        with st.chat_message("assistant"):
            # Langkah 1: Spinner hanya untuk proses retrieval (Database + Web Search)
            # Begitu retrieval selesai, spinner hilang dan teks mulai mengetik.
            with st.spinner("🔍 Retrieving data & thinking..."):
                try:
                    # Kita buat generator stream
                    stream_generator = st.session_state.qa_chain.stream({"question": prompt})
                    
                    # Langkah 2: Gunakan st.write_stream
                    # Fungsi ini otomatis mengambil generator dan menampilkannya sebagai typewriter
                    response = st.write_stream(stream_generator)
                    
                    # Langkah 3: Simpan hasil akhir ke history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()