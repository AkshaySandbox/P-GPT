"""Utility functions for the pregnancy knowledge base system."""
import hashlib
import json
import time
import re
import tiktoken
from pathlib import Path
from typing import Dict, Any

from pregnancy_kb.config import METADATA_FILE

def extract_title_from_pdf(file_path: str) -> str:
    """Extract a title from the PDF filename or content."""
    filename = Path(file_path).stem
    # Convert snake/kebab case to title case
    clean_title = re.sub(r'[_-]', ' ', filename).title()
    return clean_title

def tiktoken_len(text: str) -> int:
    """Calculate the number of tokens in a text using tiktoken."""
    try:
        tokens = tiktoken.encoding_for_model("gpt-4").encode(text)
        return len(tokens)
    except Exception:
        # Fallback to approximate token count if tiktoken fails
        return len(text.split()) * 1.3

def get_processed_files() -> Dict[str, Any]:
    """Load the list of already processed files."""
    Path(METADATA_FILE).parent.mkdir(parents=True, exist_ok=True)
    if Path(METADATA_FILE).exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading metadata file. Creating new one.")
    return {"processed_files": {}, "last_updated": ""}

def save_processed_files(processed_files: Dict[str, Any]) -> None:
    """Save the list of processed files with timestamp."""
    processed_files["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(METADATA_FILE, 'w') as f:
        json.dump(processed_files, f, indent=2)

def file_hash(file_path: str) -> str:
    """Generate a hash of the file to detect changes."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest() 