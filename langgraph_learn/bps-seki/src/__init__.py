# src/__init__.py
"""Agentic AI System - Package initialization"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Multi-agent AI system with LangGraph and local LLMs"

# Import utama
from .config import Config, USER_CONTEXT
from .llm_client import LLMClient
from .metadata_manager import MetadataManager
from .sql_validator import SQLValidator
from .sql_executor import SQLExecutor
from .forecast_agent import ForecastAgent, EnhancedForecastAgent
from .smart_selector import SmartTableSelector
from .logger import AuditLogger
from .state import AgentState
from .nodes import (
    router_node,
    enhanced_metadata_retriever_node,
    planner_node,
    enhanced_sql_agent_node,
    sql_executor_node,
    enhanced_forecast_agent,
    clarify_agent_node,
    response_formatter_node,
    error_handler_node,
    end_node
)
from .workflow import build_basic_workflow, build_enhanced_workflow

__all__ = [
    "Config",
    "USER_CONTEXT",
    "LLMClient",
    "MetadataManager",
    "SQLValidator",
    "SQLExecutor",
    "ForecastAgent",
    "EnhancedForecastAgent",
    "SmartTableSelector",
    "AuditLogger",
    "AgentState",
    "build_basic_workflow",
    "build_enhanced_workflow"
]