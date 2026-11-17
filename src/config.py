import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "olist_order_reviews_dataset.csv"
CHROMA_PATH = ROOT_DIR / os.getenv("CHROMA_PATH", "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.75"))
COLLECTION_NAME = "olist_reviews"
