# 🔍 Hệ thống Tra cứu Tài liệu Kỹ thuật (RAG + Qwen)

Hệ thống **Retrieval-Augmented Generation (RAG)** sử dụng **Qwen** làm mô hình ngôn ngữ, giúp tra cứu và trả lời câu hỏi dựa trên tài liệu kỹ thuật của bạn.

## ✨ Tính năng

- 📄 **Hỗ trợ nhiều định dạng**: PDF, DOCX, TXT, Markdown
- 🔎 **Tìm kiếm ngữ nghĩa** với embedding đa ngôn ngữ (BAAI/bge-m3)
- 💬 **Trò chuyện có ngữ cảnh** – ghi nhớ lịch sử hội thoại
- 📖 **Hiển thị nguồn tài liệu** – biết câu trả lời đến từ đâu
- ⚙️ **Cấu hình linh hoạt** qua biến môi trường
- 🌐 **Giao diện web** thân thiện với Gradio

## 🏗️ Kiến trúc

```
Tài liệu → Document Loader → Text Splitter → ChromaDB (vector store)
                                                      ↓
Câu hỏi ──────────────────────────────────→ Retriever (MMR)
                                                      ↓
                                             Qwen LLM + Context → Câu trả lời
```

## 📦 Cài đặt

```bash
# 1. Clone repo
git clone <repo-url>
cd RAG_test

# 2. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Cấu hình API key
cp .env.example .env
# Sửa file .env và điền QWEN_API_KEY của bạn
```

## ⚙️ Cấu hình

Chỉnh sửa file `.env`:

### Dùng Alibaba DashScope (Cloud)
```env
QWEN_API_KEY=sk-xxxxxxxxxxxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
```
> Lấy API key tại: https://dashscope.console.aliyun.com/

### Dùng Ollama (Local – không cần internet)
```env
QWEN_API_KEY=ollama
QWEN_BASE_URL=http://localhost:11434/v1
QWEN_MODEL=qwen2.5:7b
```
> Cài Ollama: https://ollama.ai · Tải model: `ollama pull qwen2.5:7b`

## 🚀 Chạy ứng dụng

```bash
python app.py
```

Mở trình duyệt tại: **http://localhost:7860**

## 📖 Hướng dẫn sử dụng

1. **Tab 📂 Quản lý tài liệu** → Upload file PDF/DOCX/TXT/MD → Nhấn **Index tài liệu**
2. **Tab 💬 Trò chuyện** → Nhập câu hỏi kỹ thuật → Nhận câu trả lời kèm nguồn tài liệu
3. **Tab ⚙️ Cài đặt** → Xem cấu hình hiện tại

## 📁 Cấu trúc dự án

```
RAG_test/
├── app.py                      # Giao diện web Gradio
├── config.py                   # Đọc cấu hình từ .env
├── requirements.txt            # Dependencies
├── .env.example                # Mẫu cấu hình
├── data/                       # Thư mục chứa tài liệu (tùy chọn)
└── rag/
    ├── document_processor.py   # Tải & chia nhỏ tài liệu
    ├── embeddings.py           # Embedding model (BAAI/bge-m3)
    ├── vector_store.py         # ChromaDB operations
    ├── llm.py                  # Qwen LLM client
    └── retriever.py            # RAG pipeline & chain
```

## 🔧 Tùy chỉnh nâng cao

| Biến môi trường | Mặc định | Mô tả |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Model embedding (hỗ trợ tiếng Việt) |
| `CHUNK_SIZE` | `1000` | Độ dài mỗi chunk (token) |
| `CHUNK_OVERLAP` | `200` | Độ chồng lấp giữa các chunk |
| `TOP_K` | `5` | Số tài liệu tìm kiếm cho mỗi câu hỏi |
| `TEMPERATURE` | `0.1` | Độ sáng tạo của Qwen (0=chính xác, 1=sáng tạo) |
| `MAX_TOKENS` | `2048` | Độ dài tối đa câu trả lời |
