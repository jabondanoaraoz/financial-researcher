"""
SQLite Cache System
===================
Persistent cache for financial data to minimize API calls and improve performance.

Features:
- Automatic TTL (time-to-live) expiration
- Support for multiple data types (JSON, DataFrames, primitives)
- Thread-safe operations
- Cache statistics and monitoring
- Source tracking (yfinance, FRED, Alpha Vantage, etc.)

Author: Financial Researcher Team
"""

import sqlite3
import json
import pickle
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict, List
import logging
import threading
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """
    SQLite-based cache manager for financial data.

    Handles caching of API responses with automatic expiration,
    serialization of complex types (DataFrames), and cache statistics.
    """

    def __init__(
        self,
        db_path: str = "./cache/financial_data.db",
        default_ttl_hours: int = 24,
        enabled: bool = True
    ):
        """
        Initialize cache manager.

        Args:
            db_path: Path to SQLite database file
            default_ttl_hours: Default time-to-live in hours (default: 24)
            enabled: Enable/disable caching (default: True)
        """
        self.db_path = db_path
        self.default_ttl_hours = default_ttl_hours
        self.enabled = enabled
        self._lock = threading.Lock()

        # Create cache directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        if self.enabled:
            self._init_database()
            logger.info(f"Cache initialized at {db_path} (TTL: {default_ttl_hours}h)")
        else:
            logger.info("Cache is disabled")

    def _init_database(self):
        """Initialize SQLite database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    value_type TEXT NOT NULL,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                )
            """)

            # Create index on expires_at for faster cleanup
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON cache(expires_at)
            """)

            conn.commit()

    def _serialize_value(self, value: Any) -> tuple[bytes, str]:
        """
        Serialize a value for storage.

        Args:
            value: Value to serialize

        Returns:
            Tuple of (serialized_bytes, value_type)
        """
        # Handle pandas DataFrames
        if isinstance(value, pd.DataFrame):
            return pickle.dumps(value), "dataframe"

        # Handle dictionaries with DataFrames
        elif isinstance(value, dict):
            # Check if any values are DataFrames
            has_dataframe = any(isinstance(v, pd.DataFrame) for v in value.values())
            if has_dataframe:
                return pickle.dumps(value), "dict_with_dataframe"
            else:
                return json.dumps(value).encode('utf-8'), "json"

        # Handle lists
        elif isinstance(value, list):
            return json.dumps(value).encode('utf-8'), "json"

        # Handle primitives (str, int, float, bool, None)
        elif isinstance(value, (str, int, float, bool, type(None))):
            return json.dumps(value).encode('utf-8'), "json"

        # Default to pickle for other types
        else:
            return pickle.dumps(value), "pickle"

    def _deserialize_value(self, data: bytes, value_type: str) -> Any:
        """
        Deserialize a value from storage.

        Args:
            data: Serialized data
            value_type: Type of serialization used

        Returns:
            Deserialized value
        """
        if value_type in ("dataframe", "dict_with_dataframe", "pickle"):
            return pickle.loads(data)
        elif value_type == "json":
            return json.loads(data.decode('utf-8'))
        else:
            # Fallback to pickle
            return pickle.loads(data)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if not self.enabled:
            return None

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    # Get value and check expiration
                    cursor.execute("""
                        SELECT value, value_type, expires_at
                        FROM cache
                        WHERE key = ?
                    """, (key,))

                    row = cursor.fetchone()

                    if row is None:
                        logger.debug(f"Cache miss: {key}")
                        return None

                    value_data, value_type, expires_at = row

                    # Check if expired
                    expires_dt = datetime.fromisoformat(expires_at)
                    if datetime.now() > expires_dt:
                        logger.debug(f"Cache expired: {key}")
                        # Delete expired entry
                        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                        conn.commit()
                        return None

                    # Update access statistics
                    cursor.execute("""
                        UPDATE cache
                        SET access_count = access_count + 1,
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE key = ?
                    """, (key,))
                    conn.commit()

                    # Deserialize and return
                    value = self._deserialize_value(value_data, value_type)
                    logger.debug(f"Cache hit: {key}")
                    return value

            except Exception as e:
                logger.error(f"Cache get error for {key}: {str(e)}")
                return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_hours: Optional[int] = None,
        source: Optional[str] = None
    ) -> bool:
        """
        Store a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_hours: Time-to-live in hours (uses default if not specified)
            source: Data source identifier (e.g., 'yfinance', 'FRED')

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        with self._lock:
            try:
                # Calculate expiration time
                ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
                expires_at = datetime.now() + timedelta(hours=ttl)

                # Serialize value
                value_data, value_type = self._serialize_value(value)

                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    # Insert or replace cache entry
                    cursor.execute("""
                        INSERT OR REPLACE INTO cache
                        (key, value, value_type, source, expires_at, access_count)
                        VALUES (?, ?, ?, ?, ?, 0)
                    """, (key, value_data, value_type, source, expires_at.isoformat()))

                    conn.commit()

                logger.debug(f"Cache set: {key} (TTL: {ttl}h, source: {source})")
                return True

            except Exception as e:
                logger.error(f"Cache set error for {key}: {str(e)}")
                return False

    def invalidate(self, key: str) -> bool:
        """
        Invalidate (delete) a specific cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was deleted, False otherwise
        """
        if not self.enabled:
            return False

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                    deleted = cursor.rowcount > 0
                    conn.commit()

                if deleted:
                    logger.info(f"Cache invalidated: {key}")
                return deleted

            except Exception as e:
                logger.error(f"Cache invalidate error for {key}: {str(e)}")
                return False

    def invalidate_by_source(self, source: str) -> int:
        """
        Invalidate all cache entries from a specific source.

        Args:
            source: Data source identifier

        Returns:
            Number of entries deleted
        """
        if not self.enabled:
            return 0

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM cache WHERE source = ?", (source,))
                    deleted = cursor.rowcount
                    conn.commit()

                logger.info(f"Invalidated {deleted} entries from source: {source}")
                return deleted

            except Exception as e:
                logger.error(f"Cache invalidate by source error: {str(e)}")
                return 0

    def clear_expired(self) -> int:
        """
        Remove all expired cache entries.

        Returns:
            Number of entries deleted
        """
        if not self.enabled:
            return 0

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM cache WHERE expires_at < ?",
                        (datetime.now().isoformat(),)
                    )
                    deleted = cursor.rowcount
                    conn.commit()

                if deleted > 0:
                    logger.info(f"Cleared {deleted} expired cache entries")
                return deleted

            except Exception as e:
                logger.error(f"Cache clear expired error: {str(e)}")
                return 0

    def clear_all(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM cache")
                    conn.commit()

                logger.info("All cache entries cleared")
                return True

            except Exception as e:
                logger.error(f"Cache clear all error: {str(e)}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics:
            - total_entries: Total number of cached items
            - expired_entries: Number of expired items
            - total_size_mb: Total cache size in MB
            - entries_by_source: Count of entries per source
            - most_accessed: Top 5 most accessed keys
        """
        if not self.enabled:
            return {"enabled": False}

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    # Total entries
                    cursor.execute("SELECT COUNT(*) FROM cache")
                    total_entries = cursor.fetchone()[0]

                    # Expired entries
                    cursor.execute(
                        "SELECT COUNT(*) FROM cache WHERE expires_at < ?",
                        (datetime.now().isoformat(),)
                    )
                    expired_entries = cursor.fetchone()[0]

                    # Database size
                    db_size_bytes = os.path.getsize(self.db_path)
                    db_size_mb = db_size_bytes / (1024 * 1024)

                    # Entries by source
                    cursor.execute("""
                        SELECT source, COUNT(*)
                        FROM cache
                        GROUP BY source
                    """)
                    entries_by_source = dict(cursor.fetchall())

                    # Most accessed entries
                    cursor.execute("""
                        SELECT key, access_count, source
                        FROM cache
                        ORDER BY access_count DESC
                        LIMIT 5
                    """)
                    most_accessed = [
                        {"key": row[0], "count": row[1], "source": row[2]}
                        for row in cursor.fetchall()
                    ]

                return {
                    "enabled": True,
                    "total_entries": total_entries,
                    "expired_entries": expired_entries,
                    "active_entries": total_entries - expired_entries,
                    "total_size_mb": round(db_size_mb, 2),
                    "entries_by_source": entries_by_source,
                    "most_accessed": most_accessed,
                    "db_path": self.db_path
                }

            except Exception as e:
                logger.error(f"Cache stats error: {str(e)}")
                return {"error": str(e)}


# Global cache instance (singleton pattern)
_cache_instance: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """
    Get the global cache instance (singleton).

    Returns:
        CacheManager instance
    """
    global _cache_instance

    if _cache_instance is None:
        # Load settings from environment or use defaults
        from dotenv import load_dotenv
        load_dotenv()

        db_path = os.getenv("CACHE_DB_PATH", "./cache/financial_data.db")
        ttl_hours = int(os.getenv("CACHE_TTL_HOURS", "24"))
        enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"

        _cache_instance = CacheManager(
            db_path=db_path,
            default_ttl_hours=ttl_hours,
            enabled=enabled
        )

    return _cache_instance


if __name__ == "__main__":
    # Quick test
    print("Testing cache system...")

    cache = get_cache()

    # Test primitive values
    cache.set("test_string", "Hello, World!", source="test")
    cache.set("test_number", 42, source="test")
    cache.set("test_dict", {"key": "value", "number": 123}, source="test")

    print(f"String: {cache.get('test_string')}")
    print(f"Number: {cache.get('test_number')}")
    print(f"Dict: {cache.get('test_dict')}")

    # Test DataFrame
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    cache.set("test_dataframe", df, source="test")
    df_cached = cache.get("test_dataframe")
    print(f"DataFrame:\n{df_cached}")

    # Show stats
    stats = cache.get_stats()
    print(f"\nCache stats: {json.dumps(stats, indent=2)}")
