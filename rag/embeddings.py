"""
Khởi tạo embedding model (BAAI/bge-m3 - hỗ trợ đa ngôn ngữ).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Trả về embedding model (singleton, lazy-load)."""
    print(f"[INFO] Đang tải embedding model: {config.EMBEDDING_MODEL}")
    model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[INFO] Embedding model đã sẵn sàng.")
    return model
