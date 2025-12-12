# src/config.py
"""Konfigurasi sistem"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Config:
    """Konfigurasi utama sistem"""
    
    # Paths
    BASE_DIR: Path = Path.cwd()
    METADATA_DIR: Path = BASE_DIR / "metadata"
    DB_PATH: Path = BASE_DIR / "database.db"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    # LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    USER_MODEL: str = "qwen2.5:7b"
    SQL_MODEL: str = "qwen2.5-coder:latest"
    USER_TEMPERATURE: float = 0.1
    SQL_TEMPERATURE: float = 0.0
    
    # User Context (default)
    USER_CONTEXT: Dict[str, str] = field(
        default_factory=lambda: {
            "leveldata": "2_KABUPATEN_JAWA_BARAT",
            "region": "RM III JABAR"
        }
    )
    
    # SQL Configuration
    DEFAULT_LIMIT: int = 100
    MAX_QUERY_LENGTH: int = 1000
    
    # Forecasting Configuration
    DEFAULT_FORECAST_PERIODS: int = 3
    MIN_DATA_POINTS: int = 3
    
    # Security Configuration
    ENABLE_SQL_VALIDATION: bool = True
    ENABLE_REGION_FILTERING: bool = True
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        if os.getenv("OLLAMA_BASE_URL"):
            config.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
        if os.getenv("USER_MODEL"):
            config.USER_MODEL = os.getenv("USER_MODEL")
        if os.getenv("SQL_MODEL"):
            config.SQL_MODEL = os.getenv("SQL_MODEL")
        
        # Create directories
        config.METADATA_DIR.mkdir(exist_ok=True)
        config.LOG_DIR.mkdir(exist_ok=True)
        
        return config

# Global configuration instance
config = Config.from_env()
USER_CONTEXT = config.USER_CONTEXT
