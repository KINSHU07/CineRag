"""
FastAPI backend for Movie RAG system.
Supports: huggingface | mistral | claude | openai
Set ACTIVE_API in .env to switch between them.
"""

import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_loader import (
    collection, get_embedding, VECTOR_INDEX,
    ACTIVE_API, HF_API_URL, HF_HEADERS,
    mistral_client, claude_client, openai_client,
)

app = FastAPI(title="Movie RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class MovieResult(BaseModel):
    title: str
    genres: list
    plot: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[MovieResult]
    api_used: str


# ─────────────────────────────────────────────
# RAG helpers
# ─────────────────────────────────────────────
def vector_search(query: str, top_k: int = 5) -> list[dict]:
    embedding = get_embedding(query)
    if not embedding:
        raise HTTPException(status_code=400, detail="Could not embed the query.")

    pipeline = [
        {
            "$vectorSearch": {
                "index":         VECTOR_INDEX,
                "queryVector":   embedding,
                "path":          "embedding",
                "numCandidates": top_k * 20,
                "limit":         top_k,
            }
        },
        {
            "$project": {
                "title":    1,
                "fullplot": 1,
                "genres":   1,
                "poster":   1,
                "year":     1,
                "imdb":     1,
                "score":    {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def build_context(docs: list[dict]) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        title = doc.get("title", "Unknown")
        plot  = doc.get("fullplot") or doc.get("plot", "")
        lines.append(f"[{i}] Title: {title}\nPlot: {plot}")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────
# System prompt (shared across all APIs)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful and knowledgeable movie assistant. "
    "Use ONLY the context provided to answer the user's question. "
    "If the answer is not in the context, say you don't have enough information. "
    "Keep your answer concise and helpful."
)


# ─────────────────────────────────────────────
# Generator — HuggingFace Inference API
# ─────────────────────────────────────────────
def generate_with_huggingface(question: str, context: str) -> str:
    prompt = (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question} [/INST]"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        },
    }
    response = http_requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json=payload,
        timeout=120,
    )
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"HuggingFace API error {response.status_code}: {response.text}"
        )
    result = response.json()

    # HF returns a list of dicts
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "").strip()
    return str(result).strip()


# ─────────────────────────────────────────────
# Generator — Mistral API
# ─────────────────────────────────────────────
def generate_with_mistral(question: str, context: str) -> str:
    if not mistral_client:
        raise HTTPException(status_code=500, detail="Mistral client not initialised. Check MISTRAL_API_KEY in .env")
    
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=400,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Generator — Claude (Anthropic)
# ─────────────────────────────────────────────
def generate_with_claude(question: str, context: str) -> str:
    if not claude_client:
        raise HTTPException(status_code=500, detail="Claude client not initialised. Check ANTHROPIC_API_KEY in .env")
    
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
    )
    return response.content[0].text.strip()


# ─────────────────────────────────────────────
# Generator — OpenAI
# ─────────────────────────────────────────────
def generate_with_openai(question: str, context: str) -> str:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialised. Check OPENAI_API_KEY in .env")
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=400,
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────
def generate_answer(question: str, context: str) -> str:
    if ACTIVE_API == "huggingface":
        return generate_with_huggingface(question, context)
    elif ACTIVE_API == "mistral":
        return generate_with_mistral(question, context)
    elif ACTIVE_API == "claude":
        return generate_with_claude(question, context)
    elif ACTIVE_API == "openai":
        return generate_with_openai(question, context)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown ACTIVE_API '{ACTIVE_API}' in .env")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "active_api": ACTIVE_API}


@app.post("/ask", response_model=QueryResponse)
def ask(req: QueryRequest):
    docs    = vector_search(req.question, top_k=req.top_k)
    context = build_context(docs)
    answer  = generate_answer(req.question, context)

    sources = [
        MovieResult(
            title=d.get("title", "N/A"),
            genres=d.get("genres") or [],
            plot=(d.get("fullplot") or d.get("plot", ""))[:300],
            score=round(d.get("score", 0.0), 4),
        )
        for d in docs
    ]
    return QueryResponse(
        question=req.question,
        answer=answer,
        sources=sources,
        api_used=ACTIVE_API,
    )


@app.get("/movies/top")
def top_movies(limit: int = 10):
    docs = list(
        collection.find(
            {"imdb.rating": {"$exists": True, "$ne": ""}},
            {"title": 1, "year": 1, "genres": 1, "poster": 1, "imdb": 1, "_id": 0},
        )
        .sort("imdb.rating", -1)
        .limit(limit)
    )
    return docs