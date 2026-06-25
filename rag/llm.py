"""
Khởi tạo Qwen LLM thông qua OpenAI-compatible API.
Hỗ trợ: Alibaba DashScope và Ollama (chạy local).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

import config


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Trả về Qwen LLM client (singleton)."""
    print(f"[INFO] Đang kết nối Qwen model: {config.QWEN_MODEL}")
    llm = ChatOpenAI(
        model=config.QWEN_MODEL,
        openai_api_key=config.QWEN_API_KEY,
        openai_api_base=config.QWEN_BASE_URL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        streaming=True,
    )
    return llm
