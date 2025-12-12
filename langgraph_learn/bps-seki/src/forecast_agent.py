"""Forecasting agent module"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LinearRegression
from datetime import datetime

from .config import config
from .logger import AuditLogger
from .sql_executor import SQLExecutor

logger = AuditLogger()
sql_executor = SQLExecutor()

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

class ForecastAgent:
    """Basic forecasting agent"""
    
    @staticmethod
    def detect_time_series_columns(metadata: Dict) -> Dict[str, str]:
        """Deteksi kolom tanggal dan nilai dari metadata"""
        columns = metadata.get("columns", {})
        
        date_keywords = ['tahun', 'year', 'bulan', 'month', 'tanggal', 'date', 'periode']
        value_keywords = ['nilai', 'value', 'jumlah', 'total', 'devisa', 'pendapatan', 'qty', 'quantity', 'volume']
        
        date_candidates = []
        value_candidates = []
        
        for col_name, col_info in columns.items():
            col_lower = col_name.lower()
            col_type = str(col_info.get("type", "")).lower()
            
            # Deteksi kolom tanggal
            is_date = any(keyword in col_lower for keyword in date_keywords)
            is_date = is_date or col_type in ['date', 'datetime', 'timestamp', 'year', 'month']
            
            # Deteksi kolom nilai
            is_value = any(keyword in col_lower for keyword in value_keywords)
            is_value = is_value or col_type in ['integer', 'int', 'float', 'double', 'decimal', 'numeric', 'number']
            
            # Prioritaskan berdasarkan keyword match
            if is_date:
                score = sum(1 for keyword in date_keywords if keyword in col_lower)
                date_candidates.append((col_name, score))
            
            if is_value:
                score = sum(1 for keyword in value_keywords if keyword in col_lower)
                value_candidates.append((col_name, score))
        
        # Pilih kolom dengan score tertinggi
        date_col = max(date_candidates, key=lambda x: x[1])[0] if date_candidates else None
        value_col = max(value_candidates, key=lambda x: x[1])[0] if value_candidates else None
        
        # Fallback jika tidak ada yang terdeteksi
        if not date_col and columns:
            # Cari kolom dengan tipe date/datetime
            for col_name, col_info in columns.items():
                col_type = str(col_info.get("type", "")).lower()
                if col_type in ['date', 'datetime', 'timestamp']:
                    date_col = col_name
                    break
            if not date_col:
                date_col = list(columns.keys())[0]
        
        if not value_col and len(columns) > 1:
            # Hindari menggunakan kolom yang sama dengan date_col
            available = [col for col in list(columns.keys()) if col != date_col]
            if available:
                value_col = available[0]
            else:
                value_col = list(columns.keys())[0]
        elif not value_col and columns:
            value_col = list(columns.keys())[0]
        
        # Jika date_col dan value_col sama, cari alternatif
        if date_col == value_col and len(columns) > 1:
            for col in columns.keys():
                if col != date_col:
                    value_col = col
                    break
        
        confidence = "high"
        if not (date_candidates and value_candidates):
            confidence = "medium"
        if not date_col or not value_col:
            confidence = "low"
        
        return {
            "date_column": date_col,
            "value_column": value_col,
            "confidence": confidence
        }
    
    @staticmethod
    def simple_linear_forecast(df: pd.DataFrame, date_col: str, value_col: str, 
                               periods: int = 3) -> Dict[str, Any]:
        """Simple linear regression forecast dengan return dictionary"""
        if len(df) < config.MIN_DATA_POINTS:
            return {
                "success": False,
                "error": f"Minimum {config.MIN_DATA_POINTS} data points required, got {len(df)}"
            }
        
        try:
            # Konversi kolom tanggal jika diperlukan
            df_clean = df.copy()
            
            # Pastikan kolom nilai numerik
            if not pd.api.types.is_numeric_dtype(df_clean[value_col]):
                df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
            
            # Konversi kolom tanggal ke numerik untuk regresi
            # Jika tanggal berupa string tahun (misal: "2021")
            if not pd.api.types.is_numeric_dtype(df_clean[date_col]):
                try:
                    # Coba konversi ke tahun numerik
                    df_clean[date_col] = pd.to_numeric(df_clean[date_col], errors='coerce')
                    # Jika masih gagal, gunakan indeks
                    if df_clean[date_col].isnull().any():
                        df_clean[date_col] = range(len(df_clean))
                except:
                    # Gunakan indeks sebagai fallback
                    df_clean[date_col] = range(len(df_clean))
            
            # Hapus NaN
            df_clean = df_clean.dropna(subset=[date_col, value_col])
            
            if len(df_clean) < config.MIN_DATA_POINTS:
                return {
                    "success": False,
                    "error": f"After cleaning, only {len(df_clean)} valid data points"
                }
            
            # Prepare data
            X = np.array(df_clean[date_col]).reshape(-1, 1)
            y = np.array(df_clean[value_col])
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate predictions
            last_value = float(df_clean[date_col].iloc[-1])
            future_values = np.array([last_value + i + 1 for i in range(periods)]).reshape(-1, 1)
            predictions = model.predict(future_values)
            
            # Calculate confidence intervals
            residuals = y - model.predict(X)
            std_error = np.std(residuals)
            confidence_low = predictions - 1.96 * std_error
            confidence_high = predictions + 1.96 * std_error
            
            # Create result
            result_df = pd.DataFrame({
                'period': [f'Periode {i+1}' for i in range(periods)],
                'prediction': predictions,
                'confidence_low': confidence_low,
                'confidence_high': confidence_high
            })
            
            return {
                "success": True,
                "forecast": {
                    "table_name": None,  # Akan diisi oleh caller
                    "data_points": len(df_clean),
                    "date_range": {
                        "start": str(df_clean[date_col].iloc[0]),
                        "end": str(df_clean[date_col].iloc[-1])
                    },
                    "last_value": float(df_clean[value_col].iloc[-1]),
                    "forecast_periods": periods,
                    "predictions": result_df.to_dict("records"),
                    "method": "linear_regression",
                    "metadata": {
                        "date_column": date_col,
                        "value_column": value_col,
                        "confidence": "basic"
                    }
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Linear forecast failed: {str(e)}"
            }
    
    def linear_forecast(self, df: pd.DataFrame, date_column: str, value_column: str, periods: int) -> Dict[str, Any]:
        """Wrapper untuk compatibility dengan nodes.py"""
        return self.simple_linear_forecast(df, date_column, value_column, periods)

class SimpleForecastAgent(ForecastAgent):
    """Alias untuk ForecastAgent untuk digunakan di nodes.py"""
    pass

class EnhancedForecastAgent(ForecastAgent):
    """Enhanced forecasting with LLM and multiple methods"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def enhanced_forecast(self, table_name: str, metadata: Dict, 
                         region: str = None, user_query: str = None) -> Dict[str, Any]:
        """Enhanced forecasting dengan auto-detection"""
        
        logger.log("ENHANCED_FORECAST_START", {
            "table": table_name,
            "user_query": user_query
        })
        
        # Deteksi kolom
        columns = self.detect_time_series_columns(metadata)
        
        if not columns.get("date_column") or not columns.get("value_column"):
            return {
                "success": False,
                "error": "Tidak dapat mendeteksi kolom tanggal dan nilai",
                "metadata": columns
            }
        
        # Build SQL
        sql = f"SELECT {columns['date_column']}, {columns['value_column']} FROM {table_name}"
        
        # Add region filter
        access_column = metadata.get("access_column")
        if access_column and region:
            sql += f" WHERE {access_column} LIKE '{region}%'"
        
        sql += f" ORDER BY {columns['date_column']}"
        
        # Execute query
        result = sql_executor.execute(sql)
        
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("error", "Unknown SQL error"),
                "sql": sql
            }
        
        if "data" not in result:
            return {
                "success": False,
                "error": "SQL tidak mengembalikan data",
                "sql": sql
            }
        
        df = result["data"]
        
        if len(df) < config.MIN_DATA_POINTS:
            return {
                "success": False,
                "error": f"Insufficient data: {len(df)} rows",
                "min_required": config.MIN_DATA_POINTS
            }
        
        # Choose forecasting method
        method = "linear_regression"
        
        try:
            # Gunakan Prophet jika tersedia dan data cukup
            if PROPHET_AVAILABLE and len(df) >= 12:
                forecast_result = self.prophet_forecast(df, columns["date_column"], columns["value_column"], config.DEFAULT_FORECAST_PERIODS)
                method = "prophet"
            else:
                forecast_result = self.simple_linear_forecast(df, columns["date_column"], columns["value_column"], config.DEFAULT_FORECAST_PERIODS)
            
            if not forecast_result["success"]:
                return forecast_result
            
            # Tambahkan metadata
            forecast_summary = forecast_result["forecast"]
            forecast_summary["table_name"] = table_name
            forecast_summary["method"] = method
            forecast_summary["metadata"] = {
                "date_column": columns["date_column"],
                "value_column": columns["value_column"],
                "confidence": columns["confidence"]
            }
            
            logger.log("ENHANCED_FORECAST_SUCCESS", {
                "table": table_name,
                "method": method,
                "data_points": len(df)
            }, level="SUCCESS")
            
            return {
                "success": True,
                "forecast": forecast_summary,
                "sql": sql,
                "method": method
            }
            
        except Exception as e:
            error_msg = f"Forecast failed: {str(e)}"
            logger.log("ENHANCED_FORECAST_ERROR", {
                "error": str(e),
                "table": table_name,
                "method": method
            }, level="ERROR")
            
            return {
                "success": False,
                "error": error_msg,
                "sql": sql
            }
    
    @staticmethod
    def prophet_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> Dict[str, Any]:
        """Forecast menggunakan Facebook Prophet"""
        if not PROPHET_AVAILABLE:
            return {
                "success": False,
                "error": "Prophet library not available"
            }
        
        try:
            # Prepare data for Prophet
            prophet_df = df[[date_col, value_col]].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Konversi kolom tanggal
            if not pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
                try:
                    # Coba konversi langsung
                    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                except:
                    # Coba format tahun saja
                    try:
                        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'].astype(str), format='%Y')
                    except:
                        # Tambahkan bulan dan hari default
                        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'].astype(str) + '-01-01', errors='coerce')
            
            # Konversi kolom nilai
            prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
            
            # Hapus NaN
            prophet_df = prophet_df.dropna()
            
            if len(prophet_df) < config.MIN_DATA_POINTS:
                return {
                    "success": False,
                    "error": f"Prophet requires at least {config.MIN_DATA_POINTS} data points after cleaning"
                }
            
            # Create and fit model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model.fit(prophet_df)
            
            # Make predictions
            future = model.make_future_dataframe(periods=periods, freq='MS')
            forecast = model.predict(future)
            
            # Extract results
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
            # Format results
            result_df = pd.DataFrame({
                'period': result['ds'].dt.strftime('%Y-%m-%d'),
                'prediction': result['yhat'],
                'confidence_low': result['yhat_lower'],
                'confidence_high': result['yhat_upper']
            })
            
            return {
                "success": True,
                "forecast": {
                    "table_name": None,
                    "data_points": len(prophet_df),
                    "date_range": {
                        "start": str(prophet_df['ds'].min()),
                        "end": str(prophet_df['ds'].max())
                    },
                    "last_value": float(prophet_df['y'].iloc[-1]),
                    "forecast_periods": periods,
                    "predictions": result_df.to_dict("records"),
                    "method": "prophet",
                    "metadata": {
                        "date_column": date_col,
                        "value_column": value_col,
                        "confidence": "high"
                    }
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Prophet forecast failed: {str(e)}"
            }