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

st.set_page_config(
    page_title="BPS Data Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_tables(db_path: str) -> Dict[str, pd.DataFrame]:
    """Load semua tabel dari database SQLite"""
    engine = create_engine(f"sqlite:///{db_path}")
    insp = inspect(engine)
    tables = {}
    for name in insp.get_table_names():
        df = pd.read_sql_table(name, engine)
        tables[name] = df
    return tables

def create_table_metadata_docs(tables: Dict[str, pd.DataFrame]) -> List[Document]:
    """Buat dokumen metadata untuk setiap tabel"""
    docs = []
    
    for table_name, df in tables.items():
        metadata = {
            "type": "table_metadata",
            "table": table_name,
            "row_count": len(df)
        }
        
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            
            if unique_count < 50:
                sample_values = df[col].unique()[:10].tolist()
                columns_info.append(
                    f"- {col} ({dtype}): {unique_count} unique values. "
                    f"Examples: {sample_values}"
                )
            else:
                columns_info.append(
                    f"- {col} ({dtype}): {unique_count} unique values"
                )
        
        content = f"""
Table: {table_name}
Description: BPS statistical data table
Row count: {len(df)}
Columns:
{chr(10).join(columns_info)}

This table contains BPS (Badan Pusat Statistik) data about {table_name.replace('_', ' ')}.
"""
        
        docs.append(Document(page_content=content, metadata=metadata))
    
    return docs

def create_table_summary_docs(tables: Dict[str, pd.DataFrame]) -> List[Document]:
    """Buat dokumen summary agregat"""
    docs = []
    
    for table_name, df in tables.items():
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if categorical_cols and len(categorical_cols) <= 3:
            for group_cols in [categorical_cols[:2]] if len(categorical_cols) >= 2 else [categorical_cols[:1]]:
                try:
                    grouped = df.groupby(group_cols)
                    
                    for group_key, group_df in grouped:
                        if len(group_df) > 0:
                            if isinstance(group_key, tuple):
                                group_desc = ", ".join([f"{col}={val}" for col, val in zip(group_cols, group_key)])
                            else:
                                group_desc = f"{group_cols[0]}={group_key}"
                            
                            stats = []
                            for num_col in numeric_cols:
                                if num_col in group_df.columns:
                                    mean_val = group_df[num_col].mean()
                                    if pd.notna(mean_val):
                                        stats.append(f"{num_col} avg: {mean_val:.2f}")
                            
                            content = f"""
                                Table: {table_name}
                                Filter: {group_desc}
                                Records: {len(group_df)}
                                Statistics: {', '.join(stats) if stats else 'N/A'}

                                Data available for: {group_desc} in {table_name}
                            """
                            
                            metadata = {
                                "type": "table_summary",
                                "table": table_name,
                                "filter": group_desc,
                                "row_count": len(group_df)
                            }
                            
                            docs.append(Document(page_content=content, metadata=metadata))
                except Exception as e:
                    continue
    
    return docs

def create_row_docs(tables: Dict[str, pd.DataFrame], sample_ratio: float = 0.3) -> List[Document]:
    """Buat dokumen per-row dengan format compact"""
    docs = []
    
    for table_name, df in tables.items():
        if len(df) > 100:
            sampled_df = df.sample(n=int(len(df) * sample_ratio), random_state=42)
        else:
            sampled_df = df
        
        for idx, row in sampled_df.iterrows():
            important_fields = []
            for col, val in row.items():
                if pd.notna(val) and val != '':
                    important_fields.append(f"{col}: {val}")
            
            content = f"Table: {table_name}\n" + " | ".join(important_fields[:10])
            
            metadata = {
                "type": "row_data",
                "table": table_name,
                "row_index": idx
            }
            
            docs.append(Document(page_content=content, metadata=metadata))
    
    return docs

def format_docs(docs):
    """Format dokumen dengan prioritas: metadata > summary > row data"""
    metadata_docs = [d for d in docs if d.metadata.get("type") == "table_metadata"]
    summary_docs = [d for d in docs if d.metadata.get("type") == "table_summary"]
    row_docs = [d for d in docs if d.metadata.get("type") == "row_data"]
    
    formatted = []
    
    if metadata_docs:
        formatted.append("=== TABLE SCHEMAS ===")
        formatted.extend([d.page_content for d in metadata_docs[:2]])
    
    if summary_docs:
        formatted.append("\n=== DATA SUMMARIES ===")
        formatted.extend([d.page_content for d in summary_docs[:5]])
    
    if row_docs:
        formatted.append("\n=== SAMPLE DATA ===")
        formatted.extend([d.page_content for d in row_docs[:5]])
    
    return "\n\n".join(formatted)

# ================== INITIALIZATION ==================
@st.cache_resource
def initialize_system(db_path: str):
    """Initialize database, vectorstore, dan LLM"""
    
    # Load database
    tables = load_all_tables(db_path)
    
    # Create documents
    metadata_docs = create_table_metadata_docs(tables)
    summary_docs = create_table_summary_docs(tables)
    row_docs = create_row_docs(tables, sample_ratio=0.5)
    
    all_docs = metadata_docs + summary_docs + row_docs
    
    # Build vectorstore
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(all_docs, emb)
    
    # Initialize LLM
    llm = OllamaLLM(model="gemma3:latest")
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template("""
        You are a knowledgeable assistant specialized in Indonesian BPS (Badan Pusat Statistik) data.

        Use the following context to answer the question accurately. The context includes:
        1. Table schemas (structure and column information)
        2. Data summaries (aggregated statistics)
        3. Sample data (actual records)

        IMPORTANT INSTRUCTIONS:
        - If the answer involves numbers, provide specific values from the context
        - If you need to compare data across years, regions, or categories, use the summaries
        - If the context doesn't contain enough information, say "I don't have enough data to answer this accurately"
        - Always mention which table(s) the information comes from
        - Format numbers clearly (use thousands separators when appropriate)

        Context:
        {context}

        Question: {question}

        Answer (in Indonesian):
        """)
    
    # Create QA chain
    qa_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
    )
    
    stats = {
        "tables": tables,
        "doc_counts": {
            "metadata": len(metadata_docs),
            "summary": len(summary_docs),
            "row": len(row_docs),
            "total": len(all_docs)
        }
    }
    
    return qa_chain, stats

# ================== SESSION STATE ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
    st.session_state.stats = None

# ================== MAIN APP ==================
def main():
    # Header
    st.markdown('<div class="main-header">📊 BPS Data Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Tanya jawab interaktif dengan data Badan Pusat Statistik</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        db_path = st.text_input("Database Path", value="bps_seki.db")
        
        if st.button("🔄 Initialize System", type="primary"):
            if os.path.exists(db_path):
                with st.spinner("Loading database and building vectorstore..."):
                    try:
                        qa_chain, stats = initialize_system(db_path)
                        st.session_state.qa_chain = qa_chain
                        st.session_state.stats = stats
                        st.success("✅ System initialized successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.error("❌ Database file not found!")
        
        st.markdown("---")
        
        # Display statistics
        if st.session_state.stats:
            st.header("📈 Statistics")
            
            stats = st.session_state.stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tables", len(stats["tables"]))
            with col2:
                st.metric("Documents", stats["doc_counts"]["total"])
            
            with st.expander("📋 Document Breakdown"):
                st.write(f"**Metadata:** {stats['doc_counts']['metadata']}")
                st.write(f"**Summaries:** {stats['doc_counts']['summary']}")
                st.write(f"**Row Data:** {stats['doc_counts']['row']}")
            
            with st.expander("📊 Tables in Database"):
                for table_name, df in stats["tables"].items():
                    st.write(f"**{table_name}**: {len(df)} rows")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.caption("💡 Tip: Initialize the system first before asking questions!")
    
    # Main chat interface
    if st.session_state.qa_chain is None:
        st.info("👈 Please initialize the system from the sidebar first!")
        
        # Show example questions
        st.markdown("### 📝 Example Questions:")
        examples = [
            "Berapa inflasi Indonesia tahun 2023?",
            "Tunjukkan data kelahiran di Jawa Timur",
            "Bandingkan PDB berbagai region",
            "Apa saja kategori data yang tersedia?"
        ]
        
        for example in examples:
            st.markdown(f"- {example}")
        
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Tanyakan sesuatu tentang data BPS..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain.invoke({"question": prompt})
                    
                    # Extract answer
                    if isinstance(result, dict):
                        response = result.get("answer") or result.get("output_text") or str(result)
                    else:
                        response = str(result)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()