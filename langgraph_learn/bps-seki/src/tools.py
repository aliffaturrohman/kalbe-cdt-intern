# src/tools.py
"""External tools and services integration"""

import os
from typing import Dict, Any, List, Optional, Union
from langchain_community.tools.tavily_search import TavilySearchResults

from .config import config
from .logger import AuditLogger

logger = AuditLogger()

class WebSearchTool:
    """Wrapper untuk Tavily Search API"""
    
    def __init__(self):
        self.tool = None
        self.is_active = False
        self._initialize()
        
    def _initialize(self):
        """Inisialisasi tool jika API Key tersedia"""
        api_key = config.TAVILY_API_KEY
        
        if api_key:
            # Set environment variable yang dibutuhkan LangChain/Tavily
            os.environ["TAVILY_API_KEY"] = api_key
            
            try:
                self.tool = TavilySearchResults(
                    max_results=config.TAVILY_MAX_RESULTS
                )
                self.is_active = True
                print(f"✅ WebSearchTool initialized (Max results: {config.TAVILY_MAX_RESULTS})")
            except Exception as e:
                print(f"❌ Failed to initialize Tavily: {e}")
                self.is_active = False
        else:
            print("⚠️ WebSearchTool disabled (No TAVILY_API_KEY in config)")
            self.is_active = False
    
    def search(self, query: str) -> str:
        """
        Jalankan pencarian web.
        Returns: String yang sudah diformat rapi untuk LLM.
        """
        if not self.is_active or not self.tool:
            return "Web search is disabled or not configured."
        
        try:
            # Log aktivitas
            logger.log("TOOL_USE", {
                "tool": "tavily_search",
                "query": query
            })
            
            # Eksekusi search
            raw_results = self.tool.invoke(query)
            
            # Format hasil
            formatted_results = self._format_results(raw_results)
            
            return formatted_results
            
        except Exception as e:
            error_msg = f"Error during web search: {str(e)}"
            logger.log("TOOL_ERROR", {
                "tool": "tavily_search",
                "query": query,
                "error": str(e)
            }, level="ERROR")
            return error_msg

    def _format_results(self, results: Union[List[Dict], str]) -> str:
        """Format raw JSON result dari Tavily menjadi string text"""
        # Handle jika Tavily mengembalikan string error langsung
        if isinstance(results, str):
            return results
            
        if not results:
            return "No relevant web results found."
            
        formatted_list = []
        for i, res in enumerate(results, 1):
            content = res.get("content", "").strip()
            url = res.get("url", "No URL")
            
            # Potong konten jika terlalu panjang (misal 300 char per result)
            if len(content) > 400:
                content = content[:400] + "..."
                
            entry = f"Result {i}:\nContent: {content}\nSource: {url}"
            formatted_list.append(entry)
            
        return "\n\n".join(formatted_list)

# Global instance untuk diimport
web_search_tool = WebSearchTool()