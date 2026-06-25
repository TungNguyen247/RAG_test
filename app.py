"""
Giao diện web RAG với Qwen – tra cứu tài liệu kỹ thuật.
Chạy: python app.py
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Tuple

import gradio as gr

import config
from rag.document_processor import load_and_split
from rag.retriever import query, reset_chain
from rag.vector_store import add_documents, get_collection_info, clear_collection

# ─── CSS tùy chỉnh ────────────────────────────────────────────────────────────
_CSS = """
.gradio-container { font-family: 'Segoe UI', sans-serif; }
.source-box { background: #f0f4ff; border-left: 4px solid #4f6ef7;
              padding: 10px; border-radius: 6px; margin: 4px 0;
              font-size: 0.85rem; }
.source-title { font-weight: 600; color: #2d3a8c; }
footer { display: none !important; }
"""

# ─── Callbacks ────────────────────────────────────────────────────────────────

def upload_documents(files: List[tempfile._TemporaryFileWrapper]) -> str:
    """Xử lý upload và index tài liệu."""
    if not files:
        return "⚠️ Vui lòng chọn ít nhất một file."

    paths = [f.name for f in files]
    chunks = load_and_split(paths)
    if not chunks:
        return "❌ Không thể đọc tài liệu. Kiểm tra định dạng file (PDF, DOCX, TXT, MD)."

    added = add_documents(chunks)
    info = get_collection_info()
    reset_chain()  # reload chain với dữ liệu mới
    names = ", ".join(os.path.basename(p) for p in paths)
    return (
        f"✅ Đã index **{added}** chunk từ {len(paths)} file: {names}\n"
        f"📚 Tổng số chunk trong kho: **{info['count']}**"
    )


def clear_database() -> str:
    """Xóa toàn bộ vector store."""
    clear_collection()
    reset_chain()
    return "🗑️ Đã xóa toàn bộ dữ liệu trong kho tài liệu."


def get_db_status() -> str:
    info = get_collection_info()
    count = info.get("count", 0)
    if count == 0:
        return "📭 Kho tài liệu **trống** – hãy upload tài liệu ở tab **📂 Quản lý tài liệu**."
    return f"📚 Kho tài liệu: **{count}** chunk đã được index."


def chat(
    user_message: str,
    history: List[Tuple[str, str]],
) -> Tuple[str, List[Tuple[str, str]], str]:
    """Xử lý câu hỏi người dùng và trả lời từ RAG pipeline."""
    if not user_message.strip():
        return "", history, ""

    info = get_collection_info()
    if info.get("count", 0) == 0:
        bot_reply = "⚠️ Kho tài liệu đang trống. Vui lòng upload tài liệu trước khi đặt câu hỏi."
        history = history + [(user_message, bot_reply)]
        return "", history, ""

    try:
        answer, sources = query(user_message)
    except Exception as e:
        answer = f"❌ Lỗi khi truy vấn: {e}"
        sources = []

    history = history + [(user_message, answer)]

    # Render nguồn tài liệu
    if sources:
        src_html = "<div style='margin-top:8px'><b>📖 Nguồn tài liệu tham khảo:</b>"
        for s in sources:
            src_html += (
                f"<div class='source-box'>"
                f"<span class='source-title'>📄 {s['source']}</span><br>"
                f"<i>{s['content']}…</i></div>"
            )
        src_html += "</div>"
    else:
        src_html = ""

    return "", history, src_html


def new_conversation() -> Tuple[List, str]:
    """Bắt đầu cuộc trò chuyện mới."""
    reset_chain()
    return [], ""


# ─── Giao diện Gradio ─────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(css=_CSS, title="RAG Kỹ Thuật – Qwen") as demo:

        gr.Markdown(
            """
            # 🔍 Hệ thống Tra cứu Tài liệu Kỹ thuật
            ### Powered by **Qwen** · Retrieval-Augmented Generation (RAG)
            """
        )

        with gr.Tabs():

            # ── Tab Trò chuyện ────────────────────────────────────────────────
            with gr.Tab("💬 Trò chuyện"):
                db_status = gr.Markdown(value=get_db_status)

                chatbot = gr.Chatbot(
                    label="Hội thoại",
                    height=450,
                    show_copy_button=True,
                    bubble_full_width=False,
                )

                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Nhập câu hỏi kỹ thuật của bạn…",
                        show_label=False,
                        scale=8,
                        container=False,
                    )
                    send_btn = gr.Button("Gửi ▶", variant="primary", scale=1)

                source_display = gr.HTML(label="Nguồn tài liệu")

                with gr.Row():
                    new_chat_btn = gr.Button("🔄 Cuộc trò chuyện mới", size="sm")

                # Events
                send_btn.click(
                    fn=chat,
                    inputs=[msg_box, chatbot],
                    outputs=[msg_box, chatbot, source_display],
                )
                msg_box.submit(
                    fn=chat,
                    inputs=[msg_box, chatbot],
                    outputs=[msg_box, chatbot, source_display],
                )
                new_chat_btn.click(
                    fn=new_conversation,
                    outputs=[chatbot, source_display],
                )

            # ── Tab Quản lý tài liệu ─────────────────────────────────────────
            with gr.Tab("📂 Quản lý tài liệu"):
                gr.Markdown("### Tải lên tài liệu kỹ thuật\nHỗ trợ: **PDF, DOCX, TXT, MD**")

                file_upload = gr.File(
                    label="Chọn file tài liệu",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                )
                upload_btn  = gr.Button("📥 Index tài liệu", variant="primary")
                upload_status = gr.Markdown()

                gr.Markdown("---")
                gr.Markdown("### Quản lý kho dữ liệu")

                with gr.Row():
                    refresh_btn = gr.Button("🔃 Làm mới trạng thái")
                    clear_btn   = gr.Button("🗑️ Xóa toàn bộ dữ liệu", variant="stop")

                db_info = gr.Markdown(value=get_db_status)

                upload_btn.click(
                    fn=upload_documents,
                    inputs=[file_upload],
                    outputs=[upload_status],
                )
                clear_btn.click(
                    fn=clear_database,
                    outputs=[upload_status],
                )
                refresh_btn.click(
                    fn=get_db_status,
                    outputs=[db_info],
                )

            # ── Tab Cài đặt ───────────────────────────────────────────────────
            with gr.Tab("⚙️ Cài đặt"):
                gr.Markdown("### Cấu hình hiện tại")
                gr.Markdown(
                    f"""
| Thông số | Giá trị |
|---|---|
| **Qwen Model** | `{config.QWEN_MODEL}` |
| **API Base URL** | `{config.QWEN_BASE_URL}` |
| **Embedding Model** | `{config.EMBEDDING_MODEL}` |
| **Chunk Size** | `{config.CHUNK_SIZE}` |
| **Chunk Overlap** | `{config.CHUNK_OVERLAP}` |
| **Top-K Retrieval** | `{config.TOP_K}` |
| **Temperature** | `{config.TEMPERATURE}` |
| **Max Tokens** | `{config.MAX_TOKENS}` |
| **Vector Store** | `{config.CHROMA_PERSIST_DIR}` |

> 💡 Thay đổi cấu hình trong file `.env` hoặc biến môi trường, sau đó khởi động lại ứng dụng.
                    """
                )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
