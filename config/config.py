"""
config.py
=========
Configuración centralizada del sistema RAG.
Todos los parámetros ajustables están aquí; ningún magic number en el resto del código.
"""
 
from __future__ import annotations
import os
from dotenv import load_dotenv
 
load_dotenv()
 
 
# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSIONS = 1024          # debe coincidir con vector(1024) en Supabase
EMBEDDING_BATCH_SIZE = 32            # cuántos textos encodear a la vez
 
# ── Retriever ────────────────────────────────────────────────────────────────
MATCH_THRESHOLD = 0.30               # similitud coseno mínima para incluir un doc
MATCH_COUNT     = 5                  # máximo de docs recuperados por consulta
 
# ── Contexto y ventana de tokens ─────────────────────────────────────────────
MAX_CONTEXT_CHARS   = 3000           # ~750 tokens — espacio reservado al contexto RAG
MAX_HISTORY_TURNS   = 4              # últimos N turnos (user+assistant) en memoria
MAX_NEW_TOKENS      = 600            # tokens máximos que genera el LLM por respuesta
 
# ── LLM ──────────────────────────────────────────────────────────────────────
HF_MODEL    = "meta-llama/Meta-Llama-3-8B-Instruct"
TEMPERATURE = 0.1                   # baja para respuestas más factuales
 
# ── Grounding check ──────────────────────────────────────────────────────────
GROUNDING_OVERLAP_THRESHOLD = 0.28   # ratio mínimo de overlap léxico para aprobar
GROUNDING_MIN_WORD_LEN      = 6      # ignorar palabras cortas en el overlap
 
# ── API server ──────────────────────────────────  ──────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
 
# ── Credenciales ─────────────────────────────────────────────────────────────
SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]
HF_API_KEY    = os.environ["HUGGINGFACE_API_KEY"]