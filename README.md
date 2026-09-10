# 🍽️ RAG Recipe Generator Agent

A **Retrieval-Augmented Generation (RAG)** powered recipe assistant built with:
- 🧠 **LLM**: Meta Llama 3.1 8B (open-source) via [Groq](https://console.groq.com/) (free)
- 📚 **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no API key)
- 🗂️ **Vector Store**: FAISS (in-memory, persisted to disk)
- 🔗 **Orchestration**: LangChain RetrievalQA
- 🌐 **Backend**: Flask
- 🎨 **Frontend**: Vanilla HTML/CSS/JS

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[Flask API /api/ask]
    │
    ├─► [HuggingFace Embeddings]  ← embed the query
    │         │
    │         ▼
    │   [FAISS Vector Store]  ← similarity search on recipes_data.txt
    │         │
    │         ▼ top-3 recipe chunks
    │
    └─► [LangChain RetrievalQA]
              │
              ▼
       [Llama 3.1 8B via Groq]  ← generate the recipe answer
              │
              ▼
        JSON response → Frontend
```

---

## 🚀 Quick Start

### 1. Clone & set up environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Get a FREE Groq API key
1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up for free
3. Create an API key
4. Copy `.env.example` to `.env` and paste your key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 📁 Project Structure

```
├── app.py                 # Flask backend + RAG pipeline
├── requirements.txt       # Python dependencies
├── recipes_data.txt       # Recipe knowledge base (RAG source)
├── .env.example           # Environment variable template
├── templates/
│   └── index.html         # Frontend UI
├── static/
│   └── style.css          # Stylesheet
└── faiss_index/           # Auto-created: persisted vector store
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend UI |
| POST | `/api/ask` | RAG query → LLM response |
| GET | `/api/suggest` | Recipe suggestions list |
| POST | `/api/search` | Semantic similarity search |
| GET | `/api/health` | Health check |

### Example: POST `/api/ask`
```json
// Request
{ "query": "Give me a quick vegetarian pasta recipe" }

// Response
{
  "answer": "Here's a delicious ...",
  "sources": ["Creamy Mushroom Risotto"],
  "status": "success"
}
```

---

## 🧩 Extending the Knowledge Base

Add more recipes to `recipes_data.txt` following the existing format, then delete the `faiss_index/` folder so the vector store is rebuilt on next startup.

---

## 🛠️ Tech Stack

| Component | Technology | License |
|-----------|-----------|---------|
| LLM | Meta Llama 3.1 8B | Open-source (Meta Llama 3.1 Community) |
| LLM API | Groq (free tier) | Free |
| Embeddings | all-MiniLM-L6-v2 | Apache 2.0 |
| Vector Store | FAISS | MIT |
| Framework | LangChain | MIT |
| Web | Flask | BSD |
