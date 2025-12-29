# src/config.py
"""Konfigurasi sistem menggunakan Azure OpenAI"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # --- Root Path ---
    BASE_DIR: Path = Path.cwd()
    
    # --- Azure OpenAI & Model ---
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    USER_MODEL: str = os.getenv("AZURE_MODEL_DEPLOYMENT", "gpt-5-mini")
    MODEL_VERSION: str = os.getenv("AZURE_MODEL_VERSION", "2024-02-15-preview")
    
    # Casting ke tipe data yang benar agar aman saat dipanggil
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    MODEL_MAX_TOKEN: int = int(os.getenv("MODEL_MAX_TOKEN", "512"))
    
    # --- Web Search (Tavily) ---
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
    
    # --- Dynamic Paths ---
    DB_NAME: str = os.getenv("DB_NAME", "database.db")
    METADATA_FOLDER: str = os.getenv("METADATA_DIR_NAME", "metadata")
    LOG_FOLDER: str = os.getenv("LOG_DIR_NAME", "logs")
    DATA_FOLDER: str = os.getenv("DATA_DIR_NAME", "data")
    
    # --- Forecasting Config ---
    MIN_DATA_POINTS: int = int(os.getenv("MIN_DATA_POINTS", "3"))
    DEFAULT_FORECAST_PERIODS: int = int(os.getenv("DEFAULT_FORECAST_PERIODS", "3"))
    DEFAULT_LIMIT: int = int(os.getenv("DEFAULT_LIMIT", "5"))

    @property
    def DB_PATH(self) -> Path:
        return self.BASE_DIR / self.DB_NAME

    @property
    def LOG_DIR(self) -> Path:
        return self.BASE_DIR / self.LOG_FOLDER

    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / self.DATA_FOLDER

    @property
    def METADATA_DIR(self) -> Path:
        subfolder = os.getenv("ACTIVE_METADATA_SUBFOLDER", "")
        return self.BASE_DIR / self.METADATA_FOLDER / subfolder

    USER_CONTEXT: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls):
        config = cls()
        
        # Load User Context
        config.USER_CONTEXT = {
            "leveldata": os.getenv("LEVELDATA", "2_KABUPATEN_JAWA_BARAT"),
            "region": os.getenv("REGION", "RM III JABAR")
        }

        # Auto-create directories jika belum ada
        config.LOG_DIR.mkdir(exist_ok=True)
        config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
        
        return config

# Inisialisasi
config = Config.from_env()

# Export untuk akses instan
DB_PATH = config.DB_PATH
METADATA_DIR = config.METADATA_DIR
DATA_DIR = config.DATA_DIR