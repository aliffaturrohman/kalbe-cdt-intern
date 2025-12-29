# src/llm_client.py
"""LLM Client wrapper untuk Azure OpenAI dan LangChain"""

import os
from typing import Dict, Any, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from .config import config
from .logger import AuditLogger

logger = AuditLogger()

class LLMClient:
    """Client untuk menghandle LLM calls menggunakan Azure OpenAI"""
    
    def __init__(self):
        self.user_llm = None
        self.sql_llm = None
        self._initialized = False
        
    def initialize(self):
        """Initialize Azure OpenAI models"""
        if self._initialized:
            return

        try:
            print("🔄 Initializing Azure OpenAI models...")
            
            if not config.AZURE_OPENAI_API_KEY or not config.AZURE_OPENAI_ENDPOINT:
                raise ValueError("Azure OpenAI Credentials belum diset di .env")

            # --- DETEKSI MODEL O1 ---
            # Model o1 (reasoning) memiliki aturan parameter yang berbeda
            model_name = config.USER_MODEL.lower()
            is_o1_model = "o1" in model_name or "mini" in model_name

            # --- KONFIGURASI PARAMETER DINAMIS ---
            common_kwargs = {
                "azure_deployment": config.USER_MODEL,
                "openai_api_version": config.MODEL_VERSION,
                "azure_endpoint": config.AZURE_OPENAI_ENDPOINT,
                "api_key": config.AZURE_OPENAI_API_KEY,
            }

            # Khusus O1: Hapus temperature & Hapus/Perbesar max_tokens
            if is_o1_model:
                print(f"⚠️ Model '{config.USER_MODEL}' detected as O1/Reasoning model.")
                print("   -> Removing 'temperature' constraint.")
                print("   -> Removing 'max_tokens' constraint to allow reasoning.")
                # Kita TIDAK set temperature dan max_tokens untuk O1 agar default Azure yang dipakai
                # O1 butuh ribuan token untuk reasoning, 512 terlalu kecil.
            else:
                # Model Standar (GPT-4o, GPT-3.5)
                common_kwargs["temperature"] = config.MODEL_TEMPERATURE
                common_kwargs["max_tokens"] = config.MODEL_MAX_TOKEN

            # Inisialisasi Model Utama
            self.user_llm = AzureChatOpenAI(**common_kwargs)
            
            # Inisialisasi Model SQL
            # Untuk SQL, jika bukan O1, kita paksa temperature 0
            sql_kwargs = common_kwargs.copy()
            if not is_o1_model:
                sql_kwargs["temperature"] = 0
            
            self.sql_llm = AzureChatOpenAI(**sql_kwargs)
            
            self._initialized = True
            print(f"✅ Azure OpenAI models initialized: {config.USER_MODEL}")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR initializing Azure OpenAI: {str(e)}")
            raise
    
    def call_user_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Panggil user LLM untuk general tasks"""
        if not self._initialized:
            self.initialize()
        
        try:
            logger.log_llm_call(config.USER_MODEL, prompt, "Calling Azure...")
            
            # Gunakan invoke langsung
            response = self.user_llm.invoke(prompt, **kwargs)
            content_result = response.content if hasattr(response, 'content') else str(response)

            # Debugging jika kosong
            if not content_result:
                print("⚠️ WARNING: LLM returned empty content!")
                print(f"   Raw Response: {response}")

            logger.log_llm_call(
                config.USER_MODEL, 
                prompt, 
                content_result
            )
            
            return {
                "success": True,
                "content": content_result,
                "model": config.USER_MODEL
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.log("LLM_CALL_ERROR", {
                "model": config.USER_MODEL,
                "error": error_msg,
                "prompt_preview": prompt[:100]
            }, level="ERROR")
            
            return {
                "success": False,
                "error": error_msg,
                "model": config.USER_MODEL
            }
    
    def call_sql_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Panggil SQL LLM untuk SQL generation"""
        if not self._initialized:
            self.initialize()
        
        try:
            logger.log_llm_call(config.USER_MODEL, prompt, "Generating SQL...")
            
            response = self.sql_llm.invoke(prompt, **kwargs)
            content_result = response.content if hasattr(response, 'content') else str(response)
            
            logger.log_llm_call(
                config.USER_MODEL,
                prompt,
                content_result
            )
            
            return {
                "success": True,
                "content": content_result,
                "model": config.USER_MODEL
            }
            
        except Exception as e:
            logger.log("LLM_CALL_ERROR", {
                "model": config.USER_MODEL,
                "type": "SQL_GENERATION",
                "error": str(e),
                "prompt_preview": prompt[:100]
            }, level="ERROR")
            
            return {
                "success": False,
                "error": str(e),
                "model": config.USER_MODEL
            }
    
    def create_sql_chain(self, template: str):
        if not self._initialized:
            self.initialize()
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "schema", "region", "leveldata"]
        )
        
        return prompt | self.sql_llm

# Global instance
llm_client = LLMClient()