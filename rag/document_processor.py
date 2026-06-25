"""
Xử lý và phân đoạn tài liệu kỹ thuật từ nhiều định dạng: PDF, DOCX, TXT, MD.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

import config


_LOADERS = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": Docx2txtLoader,
    ".md":   UnstructuredMarkdownLoader,
}


def load_documents(file_paths: List[str]) -> List[Document]:
    """Tải tài liệu từ danh sách đường dẫn file."""
    docs: List[Document] = []
    for path in file_paths:
        ext = Path(path).suffix.lower()
        loader_cls = _LOADERS.get(ext)
        if loader_cls is None:
            print(f"[WARNING] Định dạng '{ext}' chưa được hỗ trợ, bỏ qua: {path}")
            continue
        try:
            loader = loader_cls(path)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = os.path.basename(path)
            docs.extend(loaded)
            print(f"[INFO] Đã tải {len(loaded)} trang/đoạn từ: {os.path.basename(path)}")
        except Exception as e:
            print(f"[ERROR] Không thể tải '{path}': {e}")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Chia nhỏ tài liệu thành các chunk để lưu vào vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Tổng số chunk: {len(chunks)}")
    return chunks


def load_and_split(file_paths: List[str]) -> List[Document]:
    """Pipeline đầy đủ: tải + chia nhỏ tài liệu."""
    docs = load_documents(file_paths)
    if not docs:
        return []
    return split_documents(docs)
