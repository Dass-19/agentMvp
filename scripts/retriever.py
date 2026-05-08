"""
Módulo de recuperación semántica sobre Supabase + pgvector.
 
Responsabilidades:
  · Cargar y cachear el modelo de embeddings Qwen3-Embedding-0.6B.
  · Generar el embedding de la consulta del usuario.
  · Ejecutar la función RPC match_documents en Supabase.
  · Construir el bloque de contexto truncado para el LLM.
"""
 
from __future__ import annotations
from functools import lru_cache
 
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
 
from config import config
 
 
# ── Singletons (se inicializan una sola vez por proceso) ─────────────────────
 
@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    return model
 
 
@lru_cache(maxsize=1)
def _get_supabase() -> Client:
    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
 
 
# ── Funciones públicas ────────────────────────────────────────────────────────
 
def embed_query(text: str) -> list[float]:
    """Genera el embedding de un texto de consulta (1024 dims)."""
    model = _get_embedding_model()
    vector = model.encode(
        text,
        normalize_embeddings=True,   # cosine similarity requiere vectores normalizados
        prompt_name="query",         # Qwen3 usa prompts distintos para query vs documento
    )
    return vector.tolist()
 
 
def embed_documents(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de documentos (para populate_db)."""
    model = _get_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        prompt_name="passage",       # prompt de documento, distinto al de query
    )
    return [v.tolist() for v in vectors]
 
 
def search(query: str) -> list[dict]:
    """
    Ejecuta la búsqueda vectorial en Supabase.
 
    Returns:
        Lista de dicts con keys: id, content, metadata, similarity.
        Lista vacía si no hay resultados sobre el threshold.
    """
    supabase = _get_supabase()
    query_embedding = embed_query(query)
 
    try:
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": config.MATCH_THRESHOLD,
                "match_count":     config.MATCH_COUNT,
            },
        ).execute()
        docs = response.data or []
        return docs
 
    except Exception:
        return []
 
 
def build_context_block(documents: list[dict]) -> str:
    """
    Construye el bloque de contexto para el LLM a partir de los documentos recuperados.
 
    · Ordena por similitud descendente (Supabase ya los trae así, pero lo garantizamos).
    · Trunca por MAX_CONTEXT_CHARS para no saturar la ventana del modelo.
    · Incluye la fuente y la similitud para que el LLM pueda citarlas.
    """
    if not documents:
        return ""

    # Garantizar orden descendente por similitud
    docs_sorted = sorted(documents, key=lambda d: d.get("similarity", 0), reverse=True)
 
    parts: list[str] = []
    total_chars = 0
 
    for doc in docs_sorted:
            metadata  = doc.get("metadata", {})
            sim       = doc.get("similarity", 0.0)
            content   = doc.get("content", "").strip()
            
            # 1. Extraer un nombre principal para la cabecera (intenta varias claves lógicas)
            fuente_principal = metadata.get("fuente", metadata.get("hospital", metadata.get("seguro", "Base de datos")))
            
            # 2. Convertir el resto de la metadata en un string legible para el LLM
            # Ejemplo: "Ciudad: Guayaquil | Red: A+ | Especialidades: ['Pediatría']"
            meta_tags = []
            for key, value in metadata.items():
                # Omitimos las claves que ya usamos como fuente principal para no duplicar
                if key not in ["fuente", "hospital"] or value != fuente_principal:
                    meta_tags.append(f"{key.capitalize()}: {value}")
            
            meta_str = " | ".join(meta_tags)
            
            # 3. Construir la entrada completa
            if meta_str:
                entry = f"[{fuente_principal} | Relevancia: {sim:.1%}]\n[Atributos: {meta_str}]\n{content}"
            else:
                entry = f"[{fuente_principal} | Relevancia: {sim:.1%}]\n{content}"
    
            if total_chars + len(entry) > config.MAX_CONTEXT_CHARS:
                # Añadir lo que quepa del último documento antes de cortar
                remaining = config.MAX_CONTEXT_CHARS - total_chars
                if remaining > 120:   # solo si queda espacio significativo
                    parts.append(entry[:remaining] + "…")
                break
    
            parts.append(entry)
            total_chars += len(entry)
    
    return "\n\n---\n\n".join(parts)


