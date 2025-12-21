# 🤖 RagArt - Turkish RAG System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_UI-green?logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A Retrieval-Augmented Generation (RAG) system with Turkish language support**

Upload your JSON knowledge base → Ask questions → Get intelligent answers

</div>

---

## ✨ Features

- 📁 **Drag & Drop File Upload** - Upload your JSON knowledge base easily
- 🔍 **Semantic Search** - Vector-based similarity search with ChromaDB
- 🤖 **Multiple LLM Providers** - DeepSeek, OpenAI, Ollama, or Local fallback
- 🌐 **Modern Web Interface** - Beautiful, responsive UI
- 🇹🇷 **Turkish Language Optimized** - Multilingual embeddings

---

## 🖼️ Screenshots

### Question & Answer Interface
![Question Interface](görsel/1.png)

### Drop File Screen
![Drop File Screen](görsel/2.png)

### Source Files Screen
![Drop File Screen](görsel/3.png)
---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/kadiryonak/RagArt.git
cd RagArt
pip install -r requirements.txt
```

### 2. Configure API (Optional)

Create a `.env` file:

```env
DEEPSEEK_API_KEY=your_api_key_here
MODEL_TYPE=deepseek  # or 'local' for no API
```

### 3. Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 📖 Usage

### Step 1: Upload Files

1. Go to **"Manage Files"** tab
2. Drag & drop your JSON files
3. Click **"Reindex Knowledge Base"**

### Step 2: Ask Questions

1. Go to **"Ask Question"** tab
2. Type your question
3. Get answers from your knowledge base!

---

## 📄 JSON Format

```json
[
  {
    "topic": "Algoritma",
    "definition": "Bir problemi çözmek için tasarlanmış adımlar.",
    "details": "Bilgisayar biliminin temel kavramlarından biri."
  }
]
```

---

## 🏗️ Project Structure

```
RagArt/
├── src/                 # Core modules
│   ├── rag_system.py   # Main RAG logic
│   ├── document_loader.py
│   ├── embeddings.py
│   └── llm_providers.py
├── config/             # Configuration
├── tests/              # Test suite
├── templates/          # Web UI
├── data/               # Knowledge base (JSON files)
├── app.py              # Flask server
└── run.py              # Entry point
```

---

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API key | - |
| `MODEL_TYPE` | `deepseek`, `openai`, `ollama`, `local` | `local` |
| `DATA_FOLDER` | Knowledge base folder | `./data` |
| `PORT` | Server port | `5000` |

---

## 🧪 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Ask a question |
| `/upload` | POST | Upload JSON file |
| `/list-files` | GET | List files |
| `/delete-file` | POST | Delete file |
| `/reindex` | POST | Rebuild index |

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

---

<div align="center">

Made with by [kadiryonak](https://github.com/kadiryonak)

</div>
