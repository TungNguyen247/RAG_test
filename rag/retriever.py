"""
Pipeline RAG: tìm kiếm tài liệu liên quan và tạo câu trả lời bằng Qwen.
"""
from __future__ import annotations

from typing import Generator, List, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

import config
from rag.llm import get_llm
from rag.vector_store import get_vector_store

# Prompt hệ thống cho tra cứu tài liệu kỹ thuật
_SYSTEM_PROMPT = """Bạn là trợ lý kỹ thuật thông minh, chuyên tra cứu và giải thích tài liệu kỹ thuật.

Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp từ tài liệu. Nếu thông tin không có trong tài liệu, hãy thành thật nói rằng bạn không tìm thấy thông tin liên quan.

Ngữ cảnh từ tài liệu:
{context}

Lịch sử trò chuyện:
{chat_history}

Câu hỏi: {question}

Trả lời chi tiết, rõ ràng bằng tiếng Việt hoặc ngôn ngữ phù hợp với câu hỏi:"""


def _build_chain() -> ConversationalRetrievalChain:
    vs = get_vector_store()
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.TOP_K, "fetch_k": config.TOP_K * 2},
    )

    combine_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=_SYSTEM_PROMPT,
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        k=5,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt},
        return_source_documents=True,
        verbose=False,
    )
    return chain


# Giữ một chain instance per session (stateful memory)
_chain: ConversationalRetrievalChain | None = None


def get_chain() -> ConversationalRetrievalChain:
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain


def reset_chain() -> None:
    """Xóa lịch sử trò chuyện và khởi tạo lại chain."""
    global _chain
    _chain = None


def query(question: str) -> Tuple[str, List[dict]]:
    """
    Đặt câu hỏi và nhận câu trả lời cùng danh sách nguồn tài liệu.

    Trả về:
        answer   : chuỗi câu trả lời
        sources  : list[{"source": str, "content": str}]
    """
    chain = get_chain()
    result = chain.invoke({"question": question})

    answer = result.get("answer", "")
    source_docs = result.get("source_documents", [])

    sources = []
    seen = set()
    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        key = f"{src}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": f"{src}" + (f" (trang {page + 1})" if isinstance(page, int) else ""),
                "content": doc.page_content[:300].strip(),
            })

    return answer, sources
