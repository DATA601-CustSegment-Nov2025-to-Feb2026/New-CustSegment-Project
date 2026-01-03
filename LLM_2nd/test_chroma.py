#!/usr/bin/env python
"""Test Chroma vector store creation"""

import os
import sys
from threading import Thread
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb

load_dotenv()

def test_chroma():
    """Test Chroma vector store creation"""
    print("Starting Chroma test...")
    
    try:
        print("Initializing embeddings...")
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small",
            timeout=30,
            max_retries=2
        )
        print("✅ Embeddings initialized")
        
        # Create simple test documents
        docs = [
            Document(page_content="Test document 1", metadata={"id": 1}),
            Document(page_content="Test document 2", metadata={"id": 2}),
            Document(page_content="Test document 3", metadata={"id": 3}),
        ]
        
        print("\nCreating Chroma vector store...")
        print("This will embed 3 documents...")
        
        client = chromadb.EphemeralClient()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="test",
            client=client
        )
        
        print("✅ Vector store created successfully!")
        print(f"Number of documents in store: {vectorstore._collection.count()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run with timeout
    thread = Thread(target=test_chroma)
    thread.daemon = True
    thread.start()
    thread.join(timeout=60)
    
    if thread.is_alive():
        print("\n❌ TIMEOUT: Chroma initialization took too long (>60 seconds)")
        print("\nThis suggests the issue is in Chroma.from_documents(), not embeddings.")
        sys.exit(1)
    
    print("\n✅ Test completed!")
