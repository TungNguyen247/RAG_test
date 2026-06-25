"""
Quản lý ChromaDB vector store: thêm tài liệu và tìm kiếm.
"""
from __future__ import annotations

from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from rag.embeddings import get_embedding_model


def get_vector_store() -> Chroma:
    """Trả về ChromaDB vector store (tạo hoặc mở hiện có)."""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def add_documents(docs: List[Document]) -> int:
    """Thêm tài liệu vào vector store, trả về số chunk đã thêm."""
    if not docs:
        return 0
    vs = get_vector_store()
    vs.add_documents(docs)
    print(f"[INFO] Đã thêm {len(docs)} chunk vào vector store.")
    return len(docs)


def get_collection_info() -> dict:
    """Trả về thông tin về collection hiện tại."""
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    try:
        col = client.get_collection(config.COLLECTION_NAME)
        return {"name": config.COLLECTION_NAME, "count": col.count()}
    except Exception:
        return {"name": config.COLLECTION_NAME, "count": 0}


def clear_collection() -> None:
    """Xóa toàn bộ dữ liệu trong collection."""
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)
        print("[INFO] Đã xóa collection.")
    except Exception:
        pass
