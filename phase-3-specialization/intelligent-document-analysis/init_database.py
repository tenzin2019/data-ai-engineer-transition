#!/usr/bin/env python3
"""
Database initialization script.
Creates all tables defined in the SQLAlchemy models.
"""

import os
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.database import create_tables, test_connection, engine
from src.models import Base

def wait_for_database(max_retries=30, delay=2):
    """Wait for database to be available."""
    print("🔄 Waiting for database to be available...")
    
    for attempt in range(max_retries):
        try:
            with engine.connect() as connection:
                from sqlalchemy import text
                connection.execute(text("SELECT 1"))
                print("✅ Database is available!")
                return True
        except Exception as e:
            print(f"⏳ Attempt {attempt + 1}/{max_retries}: Database not ready yet... ({e})")
            time.sleep(delay)
    
    print("❌ Database connection timeout!")
    return False

def main():
    """Initialize the database."""
    print("🚀 Starting database initialization...")
    
    # Wait for database to be available
    if not wait_for_database():
        sys.exit(1)
    
    # Test connection
    if not test_connection():
        sys.exit(1)
    
    # Create tables
    if create_tables():
        print("🎉 Database initialization completed successfully!")
        
        # Show created tables
        print("\n📋 Created tables:")
        for table_name in Base.metadata.tables.keys():
            print(f"  - {table_name}")
    else:
        print("❌ Database initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
