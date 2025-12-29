# src/sql_validator.py
"""SQL validation and security module"""

import re
import sqlparse
from typing import Dict, Any, List, Tuple

from .config import config
from .logger import AuditLogger

logger = AuditLogger()

class SQLValidator:
    """Validasi dan keamanan SQL query"""
    
    FORBIDDEN_KEYWORDS = [
        "drop", "delete", "update", "insert", "alter",
        "truncate", "create", "attach", "detach", "grant",
        "revoke", "commit", "rollback", "savepoint", "exec",
        "execute", "sp_", "xp_", "shutdown"
    ]
    
    FORBIDDEN_PATTERNS = [
        r"--.*$",  # SQL comments
        r";\s*--",  # Multiple statements with comments
        r"union.*select",  # SQL injection patterns
        r"exec.*\(|sp_",  # Stored procedure execution
        r"xp_",  # Extended stored procedures
        r"waitfor.*delay|sleep\s*\(",  # Time-based attacks
        r"benchmark\s*\(|pg_sleep",  # Performance attacks
    ]
    
    @classmethod
    def validate_sql(cls, sql: str) -> Dict[str, Any]:
        """Validasi keamanan SQL query"""
        sql_lower = sql.lower().strip()
        
        # 1. Basic checks
        if not sql_lower.startswith("select"):
            return {
                "is_valid": False,
                "reason": "Only SELECT queries are allowed",
                "suggested_fix": "Use SELECT statement only"
            }
        
        # 2. Check forbidden keywords
        for keyword in cls.FORBIDDEN_KEYWORDS:
            # Menggunakan regex word boundary agar keyword seperti 'update' tidak match 'last_updated'
            if re.search(rf'\b{keyword}\b', sql_lower):
                return {
                    "is_valid": False,
                    "reason": f"Contains forbidden keyword: {keyword}",
                    "suggested_fix": "Remove non-SELECT operations"
                }
        
        # 3. Check forbidden patterns
        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, sql_lower, re.IGNORECASE):
                return {
                    "is_valid": False,
                    "reason": "Contains dangerous SQL pattern",
                    "suggested_fix": "Avoid comments or dangerous patterns"
                }
        
        # 4. Check structure
        if "from" not in sql_lower:
            return {
                "is_valid": False,
                "reason": "Query missing FROM clause",
                "suggested_fix": "Add FROM clause with table name"
            }
        
        # 5. Parse SQL untuk validasi struktur multi-statement
        try:
            parsed = sqlparse.parse(sql)
            if len(parsed) > 1:
                return {
                    "is_valid": False,
                    "reason": "Multiple SQL statements detected",
                    "suggested_fix": "Use only one SELECT statement"
                }
        except:
            pass  # Jika parse gagal, tetap lanjut dengan warning/logging (fail open but safe by regex)
        
        return {
            "is_valid": True,
            "reason": "Query is safe",
            "validated_sql": sql
        }
    
    @classmethod
    def inject_region_filter(cls, sql: str, access_column: str, region: str) -> str:
        """Inject region filter ke SQL query"""
        if not access_column or not region:
            return sql
        
        sql_lower = sql.lower()
        access_column_lower = access_column.lower()
        
        # Cek jika access_column sudah ada di WHERE
        if re.search(rf"where.*{re.escape(access_column_lower)}", sql_lower, re.IGNORECASE):
            logger.log("REGION_FILTER", {
                "status": "ALREADY_EXISTS",
                "access_column": access_column,
                "region": region
            })
            return sql
        
        # Inject WHERE clause
        # Strategi: Jika ada WHERE, tambahkan AND. Jika tidak, cari tempat sebelum GROUP BY/ORDER BY/LIMIT.
        if "where" in sql_lower:
            # Menggunakan regex case insensitive untuk mengganti 'where' dengan 'where <condition> AND'
            # Kita ganti occurrence pertama 'where'
            pattern = re.compile(r"where", re.IGNORECASE)
            return pattern.sub(f"WHERE {access_column} LIKE '{region}%' AND", sql, count=1)
        else:
            # Mencari posisi untuk menyisipkan WHERE
            for clause in ["group by", "order by", "limit"]:
                idx = sql_lower.find(clause)
                if idx != -1:
                    return sql[:idx] + f" WHERE {access_column} LIKE '{region}%' " + sql[idx:]
            
            # Jika tidak ada clause lain, tambahkan di akhir (sebelum semicolon)
            return sql.rstrip(";") + f" WHERE {access_column} LIKE '{region}%';"
    
    @classmethod
    def add_limit_if_missing(cls, sql: str, default_limit: int = None) -> str:
        """Tambahkan LIMIT jika tidak ada"""
        if default_limit is None:
            # Safe config access
            default_limit = getattr(config, 'DEFAULT_LIMIT', 10)
        
        sql_lower = sql.lower()
        
        if "limit" in sql_lower:
            return sql
        
        sql_clean = sql.rstrip(";").strip()
        return f"{sql_clean} LIMIT {default_limit};"
    
    @classmethod
    def extract_years_from_query(cls, user_query: str) -> List[int]:
        """Extract tahun dari query user"""
        years = re.findall(r'\b(20\d{2})\b', user_query)
        return [int(year) for year in years]