# src/state.py
"""LangGraph State definition"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """State yang mengalir melalui workflow LangGraph"""
    
    # User Input & Context
    user_input: str
    user_context: Dict[str, str]
    
    # Agent Communication
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Processing State
    intent: Optional[str]  # "sql", "forecast", "clarify"
    needs_clarification: bool
    clarification_question: Optional[str]
    clarification_response: Optional[str]
    
    # Metadata & Data
    relevant_tables: List[Dict]
    selected_table: Optional[str]
    table_metadata: Optional[Dict]
    selection_confidence: Optional[float]
    selection_reason: Optional[str]
    
    # SQL Generation
    raw_sql: Optional[str]
    validated_sql: Optional[str]
    
    # Execution Results
    execution_result: Optional[Dict]
    forecast_result: Optional[Dict]
    
    # Final Output
    final_answer: Optional[str]
    error: Optional[str]
    
    # Routing
    next_node: Optional[str]