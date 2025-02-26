"""Configuration settings for the pregnancy knowledge base system."""
import os
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Directory structure
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data" / "pdf"
METADATA_DIR = BASE_DIR / "data" / "metadata"
QDRANT_PATH = BASE_DIR / "data" / "Qdrantdb"
METADATA_FILE = METADATA_DIR / "processed_files.json"

# Qdrant settings
COLLECTION_NAME = "pregnancy_info"
VECTOR_SIZE = 1536
DISTANCE = "COSINE"

# Processing settings
BATCH_SIZE = 100
CHUNK_SIZE = 300
EMBEDDING_MODEL = "AkshaySandbox/pregnancy-mpnet-embeddings"
EMBEDDING_TYPE = "sentence_transformers"  # Options: "openai", "sentence_transformers"
# EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

# Ensure directories exist
for directory in [DATA_DIR, METADATA_DIR, QDRANT_PATH]:
    directory.mkdir(parents=True, exist_ok=True)

# Categories for classification
PREGNANCY_CATEGORIES = {
    "prenatal": ["prenatal", "first trimester", "second trimester", "third trimester", "conception", "fetus", "pregnancy test", "expecting"],
    "labor_delivery": ["labor", "delivery", "birth", "contractions", "cesarean", "c-section", "water breaking", "hospital bag"],
    "postpartum": ["postpartum", "after birth", "recovery", "breastfeeding", "nursing", "lactation", "lochia"],
    "newborn_care": ["newborn", "infant", "baby care", "diaper", "feeding", "sleep training", "swaddling", "bathing baby"],
    "maternal_health": ["maternal health", "mother's health", "mom health", "self-care", "pelvic floor", "kegel"],
    "complications": ["complication", "risk", "emergency", "warning sign", "danger", "preeclampsia", "gestational diabetes"],
    "nutrition": ["nutrition", "diet", "food", "eating", "vitamins", "supplements", "folate", "iron"],
    "mental_health": ["mental health", "depression", "anxiety", "postpartum depression", "emotional", "mood swings", "baby blues"],
    "canadian_benefits": ["canada child benefit", "maternity leave", "parental leave", "ei", "employment insurance", "ccb", "tax credit"]
} 