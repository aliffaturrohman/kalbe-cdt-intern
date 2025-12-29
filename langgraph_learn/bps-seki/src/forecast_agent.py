# src/forecast_agent.py
"""Forecasting agent module using Moving Average"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .config import config
from .logger import AuditLogger
from .sql_executor import SQLExecutor

logger = AuditLogger()
sql_executor = SQLExecutor()

class ForecastAgent:
    """Basic forecasting agent"""
    
    @staticmethod
    def detect_time_series_columns(metadata: Dict) -> Dict[str, str]:
        """Deteksi kolom tanggal dan nilai dari metadata"""
        columns = metadata.get("columns", {})
        
        date_keywords = ['tahun', 'year', 'bulan', 'month', 'tanggal', 'date', 'periode', 'waktu']
        value_keywords = ['nilai', 'value', 'jumlah', 'total', 'devisa', 'pendapatan', 'qty', 'quantity', 'volume', 'harga', 'price']
        
        date_candidates = []
        value_candidates = []
        
        for col_name, col_info in columns.items():
            col_lower = col_name.lower()
            # Handle jika col_info string atau dict
            col_type = str(col_info.get("type", "")).lower() if isinstance(col_info, dict) else "string"
            
            # Deteksi kolom tanggal
            is_date = any(keyword in col_lower for keyword in date_keywords)
            is_date = is_date or col_type in ['date', 'datetime', 'timestamp', 'year', 'month']
            
            # Deteksi kolom nilai
            is_value = any(keyword in col_lower for keyword in value_keywords)
            is_value = is_value or col_type in ['integer', 'int', 'float', 'double', 'decimal', 'numeric', 'number']
            
            if is_date:
                score = sum(1 for keyword in date_keywords if keyword in col_lower)
                date_candidates.append((col_name, score))
            
            if is_value:
                score = sum(1 for keyword in value_keywords if keyword in col_lower)
                value_candidates.append((col_name, score))
        
        # Pilih kolom dengan score tertinggi
        date_col = max(date_candidates, key=lambda x: x[1])[0] if date_candidates else None
        value_col = max(value_candidates, key=lambda x: x[1])[0] if value_candidates else None
        
        # Fallback logic
        col_keys = list(columns.keys())
        if not date_col and col_keys:
            date_col = col_keys[0]
        
        if not value_col and len(col_keys) > 1:
            value_col = col_keys[1] if col_keys[1] != date_col else col_keys[0]
        elif not value_col and col_keys:
            value_col = col_keys[0]
            
        confidence = "high" if date_candidates and value_candidates else "low"
        
        return {
            "date_column": date_col,
            "value_column": value_col,
            "confidence": confidence
        }
    
    @staticmethod
    def moving_average_forecast(df: pd.DataFrame, date_col: str, value_col: str, 
                                periods: int = 3, window_size: int = 3) -> Dict[str, Any]:
        """
        Melakukan forecasting menggunakan Recursive Moving Average.
        Mengambil rata-rata 'window_size' data terakhir untuk memprediksi data berikutnya.
        """
        min_data = getattr(config, 'MIN_DATA_POINTS', 3)
        
        if len(df) < min_data:
            return {
                "success": False,
                "error": f"Data points kurang. Butuh minimal {min_data}, punya {len(df)}"
            }
        
        try:
            df_clean = df.copy()
            
            # Pastikan kolom nilai numerik
            if not pd.api.types.is_numeric_dtype(df_clean[value_col]):
                df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
            
            # Hapus NaN/Null
            df_clean = df_clean.dropna(subset=[value_col])
            
            if df_clean.empty:
                return {"success": False, "error": "Semua data bernilai NaN setelah cleaning"}

            # --- Logic Moving Average ---
            history_values = list(df_clean[value_col].values)
            predictions = []
            
            # Generate periode masa depan (Label)
            last_date_val = df_clean[date_col].iloc[-1]
            future_periods = []
            
            # Coba deteksi apakah tanggal berupa integer (tahun) atau string
            is_year_int = False
            try:
                last_year = int(str(last_date_val))
                if 1900 < last_year < 2100:
                    is_year_int = True
            except:
                pass

            for i in range(periods):
                # 1. Hitung Average dari window terakhir
                # Jika history kurang dari window_size, ambil semua yang ada
                current_window = history_values[-window_size:] if len(history_values) >= window_size else history_values
                
                avg_val = sum(current_window) / len(current_window)
                
                # Hitung standar deviasi window untuk interval kepercayaan sederhana
                std_dev = np.std(current_window) if len(current_window) > 1 else (avg_val * 0.05)
                
                # 2. Simpan prediksi
                predictions.append({
                    "prediction": avg_val,
                    "confidence_low": avg_val - (1.96 * std_dev), # Approx 95% CI
                    "confidence_high": avg_val + (1.96 * std_dev)
                })
                
                # 3. Masukkan hasil prediksi ke history untuk perhitungan periode berikutnya (Recursive)
                history_values.append(avg_val)
                
                # 4. Generate label periode
                if is_year_int:
                    future_periods.append(str(last_year + i + 1))
                else:
                    future_periods.append(f"Periode +{i+1}")

            # Format Output
            formatted_preds = []
            for i, pred in enumerate(predictions):
                formatted_preds.append({
                    "period": future_periods[i],
                    "prediction": float(pred["prediction"]),
                    "confidence_low": float(pred["confidence_low"]),
                    "confidence_high": float(pred["confidence_high"])
                })
            
            return {
                "success": True,
                "forecast": {
                    "table_name": None, # Diisi caller
                    "data_points": len(df_clean),
                    "date_range": {
                        "start": str(df_clean[date_col].iloc[0]),
                        "end": str(df_clean[date_col].iloc[-1])
                    },
                    "last_value": float(df_clean[value_col].iloc[-1]),
                    "forecast_periods": periods,
                    "predictions": formatted_preds,
                    "method": f"moving_average (window={window_size})",
                    "metadata": {
                        "date_column": date_col,
                        "value_column": value_col,
                        "window_size": window_size
                    }
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Moving average forecast failed: {str(e)}"
            }

    # Compatibility wrapper
    def linear_forecast(self, df: pd.DataFrame, date_column: str, value_column: str, periods: int) -> Dict[str, Any]:
        """Wrapper agar tetap kompatibel dengan panggilan method lama"""
        return self.moving_average_forecast(df, date_column, value_column, periods)

class SimpleForecastAgent(ForecastAgent):
    """Alias untuk ForecastAgent"""
    pass

class EnhancedForecastAgent(ForecastAgent):
    """Enhanced forecasting wrapper"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def enhanced_forecast(self, table_name: str, metadata: Dict, 
                          region: str = None, user_query: str = None) -> Dict[str, Any]:
        """Orchestrator untuk forecasting"""
        
        logger.log("ENHANCED_FORECAST_START", {
            "table": table_name,
            "method": "Moving Average"
        })
        
        # 1. Deteksi Kolom
        columns = self.detect_time_series_columns(metadata)
        if not columns.get("date_column") or not columns.get("value_column"):
            return {
                "success": False, 
                "error": "Kolom tanggal/nilai tidak terdeteksi otomatis."
            }
            
        # 2. Ambil Data via SQL
        sql = f"SELECT {columns['date_column']}, {columns['value_column']} FROM {table_name}"
        
        access_column = metadata.get("access_column")
        if access_column and region:
            sql += f" WHERE {access_column} LIKE '{region}%'"
            
        sql += f" ORDER BY {columns['date_column']}"
        
        result = sql_executor.execute(sql)
        
        if not result["success"] or "data" not in result:
             return {"success": False, "error": "Gagal mengambil data dari database."}
             
        df = result["data"]
        
        # 3. Lakukan Forecasting (Moving Average)
        forecast_periods = getattr(config, 'DEFAULT_FORECAST_PERIODS', 3)
        
        forecast_result = self.moving_average_forecast(
            df, 
            columns['date_column'], 
            columns['value_column'], 
            periods=forecast_periods
        )
        
        if forecast_result["success"]:
            # Inject metadata tambahan
            forecast_result["forecast"]["table_name"] = table_name
            
            logger.log("FORECAST_SUCCESS", {
                "table": table_name,
                "method": "Moving Average"
            }, level="SUCCESS")
            
        else:
            logger.log("FORECAST_ERROR", {
                "error": forecast_result.get("error")
            }, level="ERROR")
            
        return forecast_result