#!/usr/bin/env python
"""Test embedding creation with timeout"""

import os
import sys
from threading import Thread
from dotenv import load_dotenv

load_dotenv()

def test_embedding():
    """Test embedding initialization"""
    print("Starting embedding test...")
    
    try:
        from langchain_openai import OpenAIEmbeddings
        
        print("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small",
            timeout=30,
            max_retries=2
        )
        
        print("✅ Embeddings initialized successfully!")
        
        # Try to embed a simple text
        print("\nTesting embedding a simple text...")
        result = embeddings.embed_query("test")
        print(f"✅ Embedding successful! Vector size: {len(result)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run with timeout
    thread = Thread(target=test_embedding)
    thread.daemon = True
    thread.start()
    thread.join(timeout=45)
    
    if thread.is_alive():
        print("\n❌ TIMEOUT: Embedding initialization took too long (>45 seconds)")
        sys.exit(1)
    
    print("\n✅ Test completed!")
