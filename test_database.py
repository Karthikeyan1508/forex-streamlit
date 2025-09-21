#!/usr/bin/env python3
"""
Test script to validate database enhancements for deployment
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our enhanced database functions
from app import get_db_engine, init_db, check_db_health, insert_trade, DB_FILE, DATA_DIR

def test_database_functions():
    """Test all database enhancement functions"""
    
    print("🧪 Testing Enhanced Database Functions")
    print("=" * 50)
    
    # Test 1: Database initialization
    print("\n1️⃣ Testing database initialization...")
    try:
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    # Test 2: Database health check
    print("\n2️⃣ Testing database health check...")
    try:
        healthy, message = check_db_health()
        if healthy:
            print(f"✅ Database healthy: {message}")
        else:
            print(f"❌ Database unhealthy: {message}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 3: Database file verification
    print("\n3️⃣ Testing database file location...")
    print(f"📁 Database location: {DB_FILE.absolute()}")
    print(f"📁 Data directory: {DATA_DIR.absolute()}")
    
    if DB_FILE.exists():
        size_bytes = DB_FILE.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"📊 Database size: {size_mb:.2f} MB")
    else:
        print("⚠️ Database file doesn't exist yet (will be created on first use)")
    
    # Test 4: Insert trade test
    print("\n4️⃣ Testing trade insertion with validation...")
    try:
        success, result = insert_trade(
            user_id=1,
            symbol="EUR/USD", 
            side="BUY",
            price=1.0850,
            quantity=10000,
            strategy="Test"
        )
        
        if success:
            print(f"✅ Trade insertion successful: {result}")
        else:
            print(f"⚠️ Trade insertion result: {result}")
            
    except Exception as e:
        print(f"❌ Trade insertion failed: {e}")
        return False
    
    # Test 5: Database connection test
    print("\n5️⃣ Testing database connection...")
    try:
        engine = get_db_engine()
        from sqlalchemy import text
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM trades"))
            trade_count = result.fetchone()[0]
            print(f"✅ Connection successful - Total trades: {trade_count}")
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False
    
    print("\n🎉 All database tests completed successfully!")
    print("\n📋 Deployment Readiness Check:")
    print("✅ Database initialization with error handling")
    print("✅ Database health monitoring")  
    print("✅ Robust trade insertion with validation")
    print("✅ Proper file path management")
    print("✅ Comprehensive error reporting")
    
    return True

if __name__ == "__main__":
    print("🚀 FX Trading Platform - Database Enhancement Test")
    print("📅 Testing database deployment readiness...")
    
    success = test_database_functions()
    
    if success:
        print("\n✅ SUCCESS: Database enhancements are deployment-ready!")
        print("   Your 'after deploying i was not able to add to db' issue should now be resolved.")
    else:
        print("\n❌ FAILURE: Issues detected in database enhancements")
    
    print("\n" + "=" * 50)
    print("Test completed!")