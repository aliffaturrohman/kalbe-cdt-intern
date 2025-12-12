# src/llm_client.py
"""LLM Client wrapper untuk Ollama dan LangChain"""

import json
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager
from langchain_core.prompts import PromptTemplate

from .config import config
from .logger import AuditLogger

logger = AuditLogger()

class LLMClient:
    """Client untuk menghandle LLM calls"""
    
    def __init__(self):
        self.user_llm = None
        self.sql_llm = None
        self._initialized = False
        
    def initialize(self):
        """Initialize LLM models"""
        try:
            print("🔄 Initializing LLM models...")
            
            self.user_llm = Ollama(
                model=config.USER_MODEL,
                temperature=config.USER_TEMPERATURE,
                base_url=config.OLLAMA_BASE_URL,
                num_predict=512
            )
            
            self.sql_llm = Ollama(
                model=config.SQL_MODEL,
                temperature=config.SQL_TEMPERATURE,
                base_url=config.OLLAMA_BASE_URL,
                num_predict=1024
            )
            
            self._initialized = True
            print(f"✅ LLM models initialized: {config.USER_MODEL}, {config.SQL_MODEL}")
            
        except Exception as e:
            print(f"❌ Failed to initialize LLM models: {e}")
            raise
    
    def call_user_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Panggil user LLM untuk general tasks"""
        if not self._initialized:
            self.initialize()
        
        try:
            logger.log_llm_call(config.USER_MODEL, prompt[:200], "Calling...")
            response = self.user_llm.invoke(prompt, **kwargs)
            
            logger.log_llm_call(
                config.USER_MODEL, 
                prompt[:200], 
                response.content[:200] if hasattr(response, 'content') else str(response)[:200]
            )
            
            return {
                "success": True,
                "content": response.content if hasattr(response, 'content') else str(response),
                "model": config.USER_MODEL
            }
            
        except Exception as e:
            logger.log("LLM_CALL_ERROR", {
                "model": config.USER_MODEL,
                "error": str(e),
                "prompt_preview": prompt[:100]
            }, level="ERROR")
            
            return {
                "success": False,
                "error": str(e),
                "model": config.USER_MODEL
            }
    
    def call_sql_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Panggil SQL LLM untuk SQL generation"""
        if not self._initialized:
            self.initialize()
        
        try:
            logger.log_llm_call(config.SQL_MODEL, prompt[:200], "Calling...")
            response = self.sql_llm.invoke(prompt, **kwargs)
            
            logger.log_llm_call(
                config.SQL_MODEL,
                prompt[:200],
                response.content[:200] if hasattr(response, 'content') else str(response)[:200]
            )
            
            return {
                "success": True,
                "content": response.content if hasattr(response, 'content') else str(response),
                "model": config.SQL_MODEL
            }
            
        except Exception as e:
            logger.log("LLM_CALL_ERROR", {
                "model": config.SQL_MODEL,
                "error": str(e),
                "prompt_preview": prompt[:100]
            }, level="ERROR")
            
            return {
                "success": False,
                "error": str(e),
                "model": config.SQL_MODEL
            }
    
    def create_sql_chain(self, template: str):
        """Buat LLMChain untuk SQL generation"""
        if not self._initialized:
            self.initialize()
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "schema", "region", "leveldata"]
        )
        
        return prompt | self.sql_llm

# Global LLM client instance
llm_client = LLMClient()