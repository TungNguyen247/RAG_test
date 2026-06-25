import os
from dotenv import load_dotenv

load_dotenv()

# ── Qwen LLM ──────────────────────────────────────────────────────────────────
# Hỗ trợ 2 chế độ:
#   1. DashScope (API của Alibaba): đặt QWEN_API_KEY và QWEN_BASE_URL
#   2. Ollama (chạy local):         đặt QWEN_BASE_URL=http://localhost:11434/v1
#                                   và QWEN_API_KEY=ollama

QWEN_API_KEY  = os.getenv("QWEN_API_KEY", "your-api-key-here")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL    = os.getenv("QWEN_MODEL", "qwen-plus")

# ── Embeddings ────────────────────────────────────────────────────────────────
# BAAI/bge-m3 hỗ trợ đa ngôn ngữ (Tiếng Việt + Tiếng Anh) rất tốt
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ── Vector Store ──────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME", "technical_docs")

# ── Document Chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))

# ── LLM Generation ────────────────────────────────────────────────────────────
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "2048"))
