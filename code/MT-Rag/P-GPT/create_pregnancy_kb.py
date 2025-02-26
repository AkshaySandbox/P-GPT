#!/usr/bin/env python3
"""Main script to create and update the pregnancy knowledge base."""
import argparse
from typing import List

from pregnancy_kb.document_processor import load_pdf_documents, split_documents
from pregnancy_kb.vector_store import initialize_vector_store, add_documents_in_batches

def create_pregnancy_knowledge_base(force_reprocess: bool = False):
    """Create or update the pregnancy knowledge base."""
    try:
        # Load only new or modified documents
        documents = load_pdf_documents()
        
        if not documents:
            print("No new documents to process.")
            return
            
        # Split documents into chunks
        print(f"Splitting {len(documents)} documents into chunks...")
        splits = split_documents(documents)
        
        if not splits:
            print("No document chunks to process.")
            return
            
        # Initialize vector store
        print("Initializing vector store...")
        vector_store, client = initialize_vector_store()
        
        if not vector_store:
            print("Failed to initialize vector store.")
            return
            
        # Process documents in batches
        print(f"Adding {len(splits)} document chunks to vector store...")
        categories = add_documents_in_batches(vector_store, splits)
        
        # Print category distribution
        print("\nDocument distribution by category:")
        for cat, count in categories.items():
            print(f"  {cat}: {count} chunks")
            
    except Exception as e:
        print(f"Error creating knowledge base: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or update the pregnancy knowledge base")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all documents")
    args = parser.parse_args()
    
    create_pregnancy_knowledge_base(force_reprocess=args.force)