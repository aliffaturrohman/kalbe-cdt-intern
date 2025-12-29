# src/__init__.py
"""Agentic AI System - Package initialization"""

__version__ = "1.0.0"
__author__ = "AI Engineer"
__description__ = "Multi-agent AI system with LangGraph and Azure OpenAI"

# 1. Configuration & Core
# Import 'config' instance agar bisa akses config.USER_CONTEXT
from .config import Config, config 
from .logger import AuditLogger
from .state import AgentState

# 2. Service Clients
from .llm_client import LLMClient, llm_client
from .sql_executor import SQLExecutor
from .metadata_manager import MetadataManager
from .sql_validator import SQLValidator

# 3. Agents & Selectors
from .forecast_agent import ForecastAgent, EnhancedForecastAgent
from .smart_selector import SmartTableSelector

# 4. Nodes (Building blocks)
from .nodes import (
    router_node,
    enhanced_metadata_retriever_node,
    metadata_retriever_node_basic,
    planner_node,
    enhanced_sql_agent_node,
    sql_agent_node_basic,
    sql_executor_node,
    forecast_agent_node_basic,
    clarify_agent_node,
    response_formatter_node,
    error_handler_node,
    end_node
)

# 5. Workflows (Ready to run graphs)
from .workflow import (
    build_basic_workflow,
    build_enhanced_workflow,
    build_hybrid_workflow,
    compare_workflows,
    get_workflow_summary
)

# Define what is available when using: from src import *
__all__ = [
    # Config
    "Config",
    "config",
    "AuditLogger",
    "AgentState",
    
    # Clients
    "LLMClient",
    "llm_client",
    "SQLExecutor",
    "MetadataManager",
    "SQLValidator",
    
    # Agents
    "ForecastAgent",
    "EnhancedForecastAgent",
    "SmartTableSelector",
    
    # Nodes
    "router_node",
    "enhanced_metadata_retriever_node",
    "planner_node",
    "enhanced_sql_agent_node",
    "sql_executor_node",
    "clarify_agent_node",
    "response_formatter_node",
    "error_handler_node",
    "end_node",
    
    # Workflows
    "build_basic_workflow",
    "build_enhanced_workflow",
    "build_hybrid_workflow",
    "compare_workflows",
    "get_workflow_summary"
]