# CineRAG - Movie Question Answering System

A movie Q&A system that uses RAG (Retrieval-Augmented Generation) to answer questions about movies using MongoDB Atlas Vector Search and an LLM of your choice.

---

## What it does

You ask a movie-related question, the system finds the most relevant movies from the database using vector search, and then generates a natural language answer based on those movies.

---

## Tech Used

- **Frontend** - Streamlit
- **Backend** - FastAPI
- **Database** - MongoDB Atlas (Vector Search)
- **Embedding Model** - thenlper/gte-large
- **LLM** - Mistral API / Claude API / OpenAI API / HuggingFace

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/cinerag.git
cd cinerag
```

### 2. Create virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create a .env file
```env
MONGO_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?appName=Cluster0
DB_NAME=mongorag
COLLECTION_NAME=mongoragcollection
VECTOR_INDEX=vector_index
ACTIVE_API=mistral
MISTRAL_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
HF_API_TOKEN=your_key_here
```

### 5. Run the app

Terminal 1:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2:
```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Project Files
