"""
Script de prueba aislada del Retriever.
Verifica que Supabase devuelve contexto relevante ANTES
de ejecutar el grafo completo.

Uso:
    python test_retriever.py
    python test_retriever.py "¿Cuánto cubre la póliza en cirugías?"
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
MATCH_THRESHOLD = 0.3
MATCH_COUNT = 3


def retrieve(query: str) -> list[dict]:
    """Busca documentos relevantes en Supabase usando similitud coseno."""
    url: str = os.environ["SUPABASE_URL"]
    key: str = os.environ["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding: list[float] = model.encode(query).tolist()

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": MATCH_THRESHOLD,
            "match_count":     MATCH_COUNT,
        },
    ).execute()

    return response.data or []


def main() -> None:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "¿Cuál es la cobertura de pediatría en Hospital Kennedy?"
    )

    print("=" * 60)
    print(f"🔍 Consulta: {query}")
    print("=" * 60)

    results = retrieve(query)

    if not results:
        print("⚠️  No se encontraron documentos con similitud suficiente.")
        print(f"   Prueba a bajar MATCH_THRESHOLD (actual: {MATCH_THRESHOLD})")
        return

    print(f"\n📚 {len(results)} documento(s) recuperado(s):\n")
    for i, doc in enumerate(results, start=1):
        similarity = doc.get("similarity", 0.0)
        content    = doc.get("content", "")
        metadata   = doc.get("metadata", {})
        print(f"  [{i}] Similitud: {similarity:.4f}")
        print(f"       Fuente:    {metadata.get('fuente', '—')}")
        print(f"       Categoría: {metadata.get('categoria', '—')}")
        print(f"       Contenido: {content[:150]}{'...' if len(content) > 150 else ''}")
        print()

    print("✅ Test de Retriever completado.")


if __name__ == "__main__":
    main()
