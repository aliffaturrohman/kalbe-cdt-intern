"""Workflow builder untuk LangGraph - Basic & Enhanced Versions"""

from langgraph.graph import StateGraph, END
from typing import Dict, Any, Callable
import inspect

from .state import AgentState
from .logger import AuditLogger
from .nodes import (
    # Basic Nodes
    router_node,
    metadata_retriever_node_basic,
    planner_node,
    sql_agent_node_basic,
    sql_executor_node,
    forecast_agent_node_basic,
    clarify_agent_node,
    response_formatter_node,
    error_handler_node,
    end_node,
    
    # Enhanced Nodes
    enhanced_metadata_retriever_node,
    enhanced_sql_agent_node,
    # enhanced_forecast_agent_node - Tidak ada di nodes.py, jadi kita buat sendiri nanti
)
from .forecast_agent import EnhancedForecastAgent
from .llm_client import llm_client

logger = AuditLogger()

# Buat instance EnhancedForecastAgent untuk digunakan di workflow
enhanced_forecast_agent = EnhancedForecastAgent(llm_client)

# 🔧 Enhanced Forecast Agent Node (kita buat di sini karena tidak ada di nodes.py)
def enhanced_forecast_agent_node(state: AgentState) -> AgentState:
    """Enhanced forecasting node dengan auto-detection dan multiple methods"""
    logger.log("NODE_ENTER", {"node": "enhanced_forecast_agent"})
    
    if not state.get("selected_table"):
        state["error"] = "No table selected for forecasting"
        state["next_node"] = "error_handler"
        return state
    
    table_name = state["selected_table"]
    table_meta = state.get("table_metadata", {})
    user_context = state.get("user_context", {})
    
    # Gunakan enhanced forecast agent
    result = enhanced_forecast_agent.enhanced_forecast(
        table_name=table_name,
        metadata=table_meta,
        region=user_context.get("region"),
        user_query=state.get("user_input")
    )
    
    if result["success"]:
        state["forecast_result"] = result
        state["next_node"] = "response_formatter"
    else:
        state["error"] = result.get("error", "Forecast failed")
        state["next_node"] = "error_handler"
    
    return state

# 🔄 ROUTING FUNCTIONS (Umum untuk semua workflow)
def route_after_router(state: AgentState) -> str:
    """Routing setelah router"""
    return state.get("next_node", "metadata_retriever")

def route_after_metadata(state: AgentState) -> str:
    """Routing setelah metadata retriever"""
    if state.get("needs_clarification", False):
        return "clarify_agent"
    return "planner"

def route_after_planner(state: AgentState) -> str:
    """Routing setelah planner"""
    return state.get("next_node", "clarify_agent")

def route_after_clarify(state: AgentState) -> str:
    """Routing setelah clarify agent"""
    if state.get("needs_clarification", False):
        return "end"
    return state.get("next_node", "end")

def build_basic_workflow() -> StateGraph:
    """
    Buat basic workflow TANPA enhanced features.
    
    Fitur Basic:
    - Manual table selection (user perlu pilih)
    - Basic SQL generation tanpa tahun detection
    - Simple forecasting tanpa auto-detection
    - Tidak ada auto-table selection dengan LLM
    """
    
    print("🔨 Building BASIC Workflow (No Enhanced Features)...")
    
    # Initialize workflow
    workflow = StateGraph(AgentState)
    
    # 🔥 BASIC NODES (tanpa enhanced features)
    basic_nodes = [
        ("router", router_node),
        ("metadata_retriever", metadata_retriever_node_basic),
        ("planner", planner_node),
        ("sql_agent", sql_agent_node_basic),
        ("sql_executor", sql_executor_node),
        ("forecast_agent", forecast_agent_node_basic),
        ("clarify_agent", clarify_agent_node),
        ("response_formatter", response_formatter_node),
        ("error_handler", error_handler_node),
        ("end", end_node)
    ]
    
    # Add nodes
    for node_name, node_func in basic_nodes:
        workflow.add_node(node_name, node_func)
        print(f"  ✅ Added BASIC node: {node_name}")
    
    # Set entry point
    workflow.set_entry_point("router")
    print(f"  ✅ Entry point: router")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "metadata_retriever": "metadata_retriever",
            "clarify_agent": "clarify_agent",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "metadata_retriever",
        route_after_metadata,
        {
            "planner": "planner",
            "clarify_agent": "clarify_agent",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "sql_agent": "sql_agent",
            "forecast_agent": "forecast_agent",
            "clarify_agent": "clarify_agent",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "clarify_agent",
        route_after_clarify,
        {
            "planner": "planner",
            "end": "end",
            "error_handler": "error_handler"
        }
    )
    
    # Add fixed edges
    workflow.add_edge("sql_agent", "sql_executor")
    workflow.add_edge("sql_executor", "response_formatter")
    workflow.add_edge("forecast_agent", "response_formatter")
    workflow.add_edge("response_formatter", "end")
    workflow.add_edge("error_handler", "end")
    workflow.add_edge("end", END)
    
    # Compile basic graph
    basic_graph = workflow.compile()
    
    print("🎉 BASIC Workflow built successfully!")
    print("   Features: Manual table selection, Basic SQL generation, Simple forecasting")
    print("   Note: User perlu memilih tabel secara manual jika ada multiple candidates")
    print(f"   Total nodes: {len(basic_nodes)}")
    
    return basic_graph

def build_enhanced_workflow() -> StateGraph:
    """
    Buat enhanced workflow DENGAN semua fitur advanced.
    
    Fitur Enhanced:
    - Auto-table selection dengan LLM
    - Smart SQL generation dengan tahun detection
    - Enhanced forecasting dengan auto-detection
    - No user intervention needed
    """
    
    print("🔨 Building ENHANCED Workflow (With Advanced Features)...")
    
    workflow = StateGraph(AgentState)
    
    # 🚀 ENHANCED NODES (dengan semua fitur)
    enhanced_nodes = [
        ("router", router_node),
        ("metadata_retriever", enhanced_metadata_retriever_node),
        ("planner", planner_node),
        ("sql_agent", enhanced_sql_agent_node),
        ("sql_executor", sql_executor_node),
        ("forecast_agent", enhanced_forecast_agent_node),  # Gunakan yang kita buat
        ("clarify_agent", clarify_agent_node),
        ("response_formatter", response_formatter_node),
        ("error_handler", error_handler_node),
        ("end", end_node)
    ]
    
    # Add nodes
    for node_name, node_func in enhanced_nodes:
        workflow.add_node(node_name, node_func)
        print(f"  ✅ Added ENHANCED node: {node_name}")
    
    # Set entry point
    workflow.set_entry_point("router")
    print(f"  ✅ Entry point: router")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "metadata_retriever": "metadata_retriever",
            "clarify_agent": "clarify_agent",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "metadata_retriever",
        route_after_metadata,
        {
            "planner": "planner",
            "clarify_agent": "clarify_agent",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "sql_agent": "sql_agent",
            "forecast_agent": "forecast_agent",
            "clarify_agent": "clarify_agent",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "clarify_agent",
        route_after_clarify,
        {
            "planner": "planner",
            "end": "end",
            "error_handler": "error_handler"
        }
    )
    
    # Add fixed edges
    workflow.add_edge("sql_agent", "sql_executor")
    workflow.add_edge("sql_executor", "response_formatter")
    workflow.add_edge("forecast_agent", "response_formatter")
    workflow.add_edge("response_formatter", "end")
    workflow.add_edge("error_handler", "end")
    workflow.add_edge("end", END)
    
    # Compile enhanced graph
    enhanced_graph = workflow.compile()
    
    print("🎉 ENHANCED Workflow built successfully!")
    print("   Features: Auto-table selection, Smart SQL generation, Enhanced forecasting")
    print("   Note: Semua otomatis, user tidak perlu intervensi")
    print(f"   Total nodes: {len(enhanced_nodes)}")
    
    return enhanced_graph

def build_hybrid_workflow() -> StateGraph:
    """
    Buat hybrid workflow: beberapa fitur enhanced, beberapa basic.
    Berguna untuk gradual migration atau testing.
    """
    
    print("🔨 Building HYBRID Workflow (Mixed Features)...")
    
    workflow = StateGraph(AgentState)
    
    # 🤝 HYBRID NODES: campuran enhanced dan basic
    hybrid_nodes = [
        ("router", router_node),
        ("metadata_retriever", enhanced_metadata_retriever_node),
        ("planner", planner_node),
        ("sql_agent", enhanced_sql_agent_node),
        ("sql_executor", sql_executor_node),
        ("forecast_agent", forecast_agent_node_basic),  # Basic forecasting
        ("clarify_agent", clarify_agent_node),
        ("response_formatter", response_formatter_node),
        ("error_handler", error_handler_node),
        ("end", end_node)
    ]
    
    # Add nodes
    for node_name, node_func in hybrid_nodes:
        workflow.add_node(node_name, node_func)
        print(f"  ✅ Added HYBRID node: {node_name}")
    
    # Set entry point
    workflow.set_entry_point("router")
    print(f"  ✅ Entry point: router")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "metadata_retriever": "metadata_retriever",
            "clarify_agent": "clarify_agent",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "metadata_retriever",
        route_after_metadata,
        {
            "planner": "planner",
            "clarify_agent": "clarify_agent",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "sql_agent": "sql_agent",
            "forecast_agent": "forecast_agent",
            "clarify_agent": "clarify_agent",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "clarify_agent",
        route_after_clarify,
        {
            "planner": "planner",
            "end": "end",
            "error_handler": "error_handler"
        }
    )
    
    # Add fixed edges
    workflow.add_edge("sql_agent", "sql_executor")
    workflow.add_edge("sql_executor", "response_formatter")
    workflow.add_edge("forecast_agent", "response_formatter")
    workflow.add_edge("response_formatter", "end")
    workflow.add_edge("error_handler", "end")
    workflow.add_edge("end", END)
    
    # Compile hybrid graph
    hybrid_graph = workflow.compile()
    
    print("🎉 HYBRID Workflow built successfully!")
    print("   Features: Auto-table selection + Smart SQL, tetapi basic forecasting")
    print("   Good for: Gradual migration dari basic ke enhanced")
    print(f"   Total nodes: {len(hybrid_nodes)}")
    
    return hybrid_graph

def compare_workflows() -> Dict[str, Dict]:
    """
    Bandingkan semua workflow untuk analisis.
    
    Returns:
        Dict dengan perbandingan fitur setiap workflow
    """
    
    comparison = {
        "basic": {
            "name": "Basic Workflow",
            "description": "Workflow tanpa enhanced features",
            "table_selection": "Manual (user perlu pilih)",
            "sql_generation": "Basic (tanpa tahun detection)",
            "forecasting": "Simple linear regression",
            "user_intervention": "Required for table selection",
            "llm_usage": "Minimal (hanya untuk SQL generation)",
            "best_for": "Testing, debugging, simple use cases",
            "node_count": 10
        },
        "enhanced": {
            "name": "Enhanced Workflow",
            "description": "Workflow dengan semua enhanced features",
            "table_selection": "Auto dengan LLM (confidence scoring)",
            "sql_generation": "Smart dengan tahun detection & context",
            "forecasting": "Enhanced dengan auto-detection & multi-methods",
            "user_intervention": "Not required (fully automatic)",
            "llm_usage": "Extensive (table selection, SQL, forecasting)",
            "best_for": "Production, complex queries, user experience",
            "node_count": 10
        },
        "hybrid": {
            "name": "Hybrid Workflow",
            "description": "Campuran enhanced dan basic features",
            "table_selection": "Auto dengan LLM",
            "sql_generation": "Smart dengan tahun detection",
            "forecasting": "Basic linear regression",
            "user_intervention": "Not required for table selection",
            "llm_usage": "Moderate (table selection & SQL)",
            "best_for": "Gradual migration, cost optimization",
            "node_count": 10
        }
    }
    
    print("\n" + "="*60)
    print("🔄 WORKFLOW COMPARISON")
    print("="*60)
    
    for workflow_type, info in comparison.items():
        print(f"\n📊 {info['name']}:")
        print(f"   Description: {info['description']}")
        print(f"   Table Selection: {info['table_selection']}")
        print(f"   SQL Generation: {info['sql_generation']}")
        print(f"   Forecasting: {info['forecasting']}")
        print(f"   User Intervention: {info['user_intervention']}")
        print(f"   Best For: {info['best_for']}")
    
    print("\n🎯 RECOMMENDATION:")
    print("   • Start with BASIC for testing and debugging")
    print("   • Use ENHANCED for production with complex queries")
    print("   • Use HYBRID for gradual migration or cost optimization")
    print("="*60)
    
    return comparison

def get_workflow_summary() -> Dict[str, Any]:
    """Dapatkan summary semua workflow yang tersedia"""
    
    workflows = {
        "basic": {
            "builder": build_basic_workflow,
            "description": "Basic workflow tanpa enhanced features",
            "features": ["Manual table selection", "Basic SQL generation", "Simple forecasting"]
        },
        "enhanced": {
            "builder": build_enhanced_workflow,
            "description": "Enhanced workflow dengan semua fitur advanced",
            "features": ["Auto-table selection", "Smart SQL generation", "Enhanced forecasting"]
        },
        "hybrid": {
            "builder": build_hybrid_workflow,
            "description": "Hybrid workflow dengan campuran fitur",
            "features": ["Auto-table selection", "Smart SQL generation", "Basic forecasting"]
        }
    }
    
    return workflows

def print_workflow_debug_info(graph: StateGraph) -> None:
    """Print debug information tentang workflow"""
    
    print("\n🔍 WORKFLOW DEBUG INFO:")
    print(f"   Nodes: {list(graph.nodes.keys())}")
    print(f"   Entry point: {graph.entry_point}")
    
    # Cek edges
    print("\n   Conditional edges:")
    for node_name, node_info in graph.nodes.items():
        if hasattr(node_info, 'conditional_edges'):
            edges = node_info.conditional_edges
            if edges:
                print(f"     {node_name}: {edges}")
    
    print(f"\n   Total nodes: {len(graph.nodes)}")

# Export semua workflow builders
__all__ = [
    "build_basic_workflow",
    "build_enhanced_workflow", 
    "build_hybrid_workflow",
    "compare_workflows",
    "get_workflow_summary",
    "print_workflow_debug_info"
]