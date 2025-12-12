# main.py
"""Main entry point untuk Agentic AI System"""

import argparse
import json
from pathlib import Path

from src import (
    config,
    llm_client,
    metadata_manager,
    sql_executor,
    workflow,
    logger
)
from src.state import AgentState

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Agentic AI System")
    parser.add_argument("--query", type=str, help="User query")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--region", type=str, default="RM III JABAR", help="User region")
    parser.add_argument("--leveldata", type=str, default="2_KABUPATEN_JAWA_BARAT", help="User leveldata")
    
    args = parser.parse_args()
    
    # Initialize components
    print("🚀 Initializing Agentic AI System...")
    
    # Initialize LLM
    llm_client.llm_client.initialize()
    
    # Initialize metadata manager
    meta_manager = metadata_manager.MetadataManager()
    
    # Test database connection
    sql_exec = sql_executor.SQLExecutor()
    if not sql_exec.test_connection():
        print("⚠️  Warning: Database connection failed")
    
    # Build workflow
    print("🔨 Building workflow...")
    agent_workflow = workflow.build_enhanced_workflow()
    
    if args.query:
        # Process single query
        user_context = {
            "region": args.region,
            "leveldata": args.leveldata
        }
        
        initial_state = AgentState(
            user_input=args.query,
            user_context=user_context,
            messages=[],
            intent=None,
            needs_clarification=False,
            clarification_question=None,
            clarification_response=None,
            relevant_tables=[],
            selected_table=None,
            table_metadata=None,
            raw_sql=None,
            validated_sql=None,
            execution_result=None,
            forecast_result=None,
            final_answer=None,
            error=None,
            next_node=None
        )
        
        result = agent_workflow.invoke(initial_state)
        
        if result.get("final_answer"):
            print(f"\n🤖 RESULT:\n{result['final_answer']}")
        elif result.get("error"):
            print(f"\n❌ ERROR: {result['error']}")
        
    elif args.interactive:
        # Interactive mode
        print("\n🎮 INTERACTIVE MODE")
        print("Type 'exit' to quit, 'help' for commands")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'help':
                    print("Commands: exit, help, context, logs")
                    continue
                elif user_input.lower() == 'context':
                    print(f"Current context: {config.USER_CONTEXT}")
                    continue
                elif user_input.lower() == 'logs':
                    logs = logger.AuditLogger().get_recent_logs(5)
                    for log in logs:
                        print(f"[{log.get('timestamp')}] {log.get('event_type')}: {log.get('message', '')}")
                    continue
                
                initial_state = AgentState(
                    user_input=user_input,
                    user_context=config.USER_CONTEXT,
                    messages=[],
                    intent=None,
                    needs_clarification=False,
                    clarification_question=None,
                    clarification_response=None,
                    relevant_tables=[],
                    selected_table=None,
                    table_metadata=None,
                    raw_sql=None,
                    validated_sql=None,
                    execution_result=None,
                    forecast_result=None,
                    final_answer=None,
                    error=None,
                    next_node=None
                )
                
                result = agent_workflow.invoke(initial_state)
                
                if result.get("final_answer"):
                    print(f"\n🤖 System: {result['final_answer']}")
                elif result.get("error"):
                    print(f"\n❌ Error: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ System error: {e}")
    
    elif args.test:
        # Run tests
        print("🧪 Running tests...")
        # Implement test suite
        pass
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()