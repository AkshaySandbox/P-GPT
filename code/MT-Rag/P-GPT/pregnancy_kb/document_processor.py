"""Document processing functions for the pregnancy knowledge base."""
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pregnancy_kb.config import DATA_DIR, PREGNANCY_CATEGORIES, CHUNK_SIZE, CHUNK_OVERLAP
from pregnancy_kb.utils import extract_title_from_pdf, file_hash, get_processed_files, save_processed_files, tiktoken_len

def extract_category(content: str) -> str:
    """Determine the category of the content based on keywords."""
    content_lower = content.lower()
    
    # Count keyword matches for each category
    category_scores = {cat: 0 for cat in PREGNANCY_CATEGORIES}
    for cat, keywords in PREGNANCY_CATEGORIES.items():
        for keyword in keywords:
            if keyword in content_lower:
                category_scores[cat] += 1
    
    # Return the category with the most matches, or "general" if no matches
    max_category = max(category_scores.items(), key=lambda x: x[1])
    return max_category[0] if max_category[1] > 0 else "general"

def load_pdf_documents() -> List[Document]:
    """Load only new or modified PDF documents."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
    documents = []
    
    # Get already processed files
    processed_data = get_processed_files()
    processed_files = processed_data["processed_files"]
    
    # Track newly processed files
    newly_processed = {}
    
    for pdf_file in pdf_files:
        file_path = str(pdf_file)
        try:
            current_hash = file_hash(file_path)
            
            # Skip if file hasn't changed
            if file_path in processed_files and processed_files[file_path]["hash"] == current_hash:
                print(f"Skipping unchanged file: {file_path}")
                newly_processed[file_path] = processed_files[file_path]
                continue
                
            print(f"Processing file: {file_path}")
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            
            title = extract_title_from_pdf(file_path)
            
            for i, doc in enumerate(pdf_docs):
                doc_id = f"{Path(file_path).stem}_{i}"
                page_num = doc.metadata.get('page', i)
                
                # Only calculate category for content that has enough text
                category = "general"
                if len(doc.page_content.strip()) > 50:
                    category = extract_category(doc.page_content)
                
                doc.metadata.update({
                    "source": file_path,
                    "title": title,
                    "doc_id": doc_id,
                    "page": page_num,
                    "category": category,
                    "total_pages": len(pdf_docs),
                    "chunk_index": i,
                })
                
                documents.append(doc)
            
            # Record this file as processed
            newly_processed[file_path] = {
                "hash": current_hash,
                "title": title,
                "pages": len(pdf_docs),
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
                
        except Exception as e:
            print(f"Error loading {pdf_file}: {str(e)}")
    
    # Update processed files list
    processed_data["processed_files"] = newly_processed
    save_processed_files(processed_data)
    
    if not documents:
        print(f"No new or modified PDF documents found in {DATA_DIR}")
        
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks while preserving metadata."""
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=tiktoken_len,
    )
    
    splits = text_splitter.split_documents(documents)
    
    # Add section identification to each chunk
    for i, doc in enumerate(splits):
        content = doc.page_content
        first_line = content.split('\n')[0] if '\n' in content else content[:50]
        
        section = first_line.strip()
        if len(section) > 100:
            section = section[:100] + "..."
            
        doc.metadata["section"] = section
        doc.metadata["chunk_id"] = f"{doc.metadata.get('doc_id', 'unknown')}_{i}"
    
    return splits 