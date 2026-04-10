import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

# ─────────────────────────────────────────────────────────────
# Load .env
# ─────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
MONGO_URI       = os.getenv("MONGO_URI")
DB_NAME         = os.getenv("DB_NAME",         "mongorag")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mongoragcollection")
VECTOR_INDEX    = os.getenv("VECTOR_INDEX",    "vector_index")

HF_API_TOKEN    = os.getenv("HF_API_TOKEN")
ACTIVE_API      = os.getenv("ACTIVE_API", "mistral")

MISTRAL_API_KEY   = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

EMBED_MODEL_ID  = "thenlper/gte-large"
DEVICE          = "cpu"

# ─────────────────────────────────────────────────────────────
# HuggingFace Inference API URL
# ─────────────────────────────────────────────────────────────
HF_MODEL_ID  = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL   = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_HEADERS   = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# ─────────────────────────────────────────────────────────────
# Embedding model  (runs locally — small + fast)
# ─────────────────────────────────────────────────────────────
print("[model_loader] Loading embedding model ...")
embedding_model = SentenceTransformer(EMBED_MODEL_ID, device=DEVICE)
print("[model_loader] Embedding model ready.")

# ─────────────────────────────────────────────────────────────
# API Clients  (only the active one needs a real key)
# ─────────────────────────────────────────────────────────────

# Option 1 — Mistral
mistral_client = None
if ACTIVE_API == "mistral" and MISTRAL_API_KEY:
    from mistralai import Mistral
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    print("[model_loader] Mistral API client ready.")

# Option 2 — Anthropic / Claude
claude_client = None
if ACTIVE_API == "claude" and ANTHROPIC_API_KEY:
    import anthropic
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print("[model_loader] Claude API client ready.")

# Option 3 — OpenAI
openai_client = None
if ACTIVE_API == "openai" and OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("[model_loader] OpenAI API client ready.")

# Option 4 — HuggingFace Inference API (no local model needed)
if ACTIVE_API == "huggingface":
    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN is missing in .env file!")
    print(f"[model_loader] HuggingFace Inference API ready → {HF_MODEL_ID}")

# ─────────────────────────────────────────────────────────────
# MongoDB
# ─────────────────────────────────────────────────────────────
print("[model_loader] Connecting to MongoDB ...")
mongo_client = MongoClient(MONGO_URI)
try:
    mongo_client.admin.command("ping")
    print("[model_loader] MongoDB connected successfully!")
except Exception as e:
    print(f"[model_loader] MongoDB connection failed: {e}")

db         = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
print(f"[model_loader] DB='{DB_NAME}'  Collection='{COLLECTION_NAME}'")

# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    return embedding_model.encode(text, normalize_embeddings=True).tolist()