"""
main.py
=======
MVP RAG por Consola - Reto 3
Flujo: Usuario → LangGraph → Supabase (pgvector) → Hugging Face → Consola

Componentes:
  A. Retriever  → Supabase (búsqueda vectorial con sentence-transformers local)
  B. Generator  → Hugging Face InferenceClient (meta-llama/Meta-Llama-3-8B-Instruct)
  C. Grafo      → LangGraph (orquestación de nodos)

Uso:
    python main.py
    python main.py "¿Qué hospitales cubren maternidad?"
"""

import os
import sys
from typing import TypedDict, Optional
from dotenv import load_dotenv

from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from langgraph.graph import StateGraph, END

load_dotenv()

# ============================================================
#  Configuración global
# ============================================================
EMBEDDING_MODEL  = "Qwen/Qwen3-Embedding-0.6B"
HF_MODEL         = "meta-llama/Meta-Llama-3-8B-Instruct"
MATCH_THRESHOLD  = 0.20
MATCH_COUNT      = 4
MAX_NEW_TOKENS   = 512

SYSTEM_PROMPT = """Eres un asesor experto en seguros médicos en Ecuador.
Tu función es responder preguntas sobre cobertura, copagos, hospitales en red y cláusulas de pólizas
basándote EXCLUSIVAMENTE en el contexto proporcionado.

Reglas:
1. Si el contexto contiene la información, responde de forma clara, precisa y estructurada.
2. Si el contexto NO contiene la información, responde: "No encontré información suficiente en la base de datos para responder esta pregunta."
3. Nunca inventes datos, porcentajes o nombres de hospitales.
4. Usa español neutro y un tono profesional pero accesible.
5. Cuando menciones porcentajes o montos, sé específico y cita la fuente del contexto."""


# ============================================================
#  A. Retriever - Búsqueda en Supabase
# ============================================================
_embedding_model: Optional[SentenceTransformer] = None
_supabase_client: Optional[Client] = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print("   📦 Cargando modelo de embeddings...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        _supabase_client = create_client(url, key)
    return _supabase_client


def search_supabase(query: str) -> list[dict]:
    """Genera embedding de la consulta y busca documentos similares en Supabase."""
    model    = _get_embedding_model()
    supabase = _get_supabase()

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


def build_context(documents: list[dict]) -> str:
    """Concatena los documentos recuperados en un bloque de contexto para el LLM."""
    if not documents:
        return "No se encontraron documentos relevantes."

    parts = []
    for i, doc in enumerate(documents, start=1):
        fuente   = doc.get("metadata", {}).get("fuente", "—")
        content  = doc.get("content", "")
        sim      = doc.get("similarity", 0.0)
        parts.append(f"[Fuente {i}: {fuente} | Similitud: {sim:.2f}]\n{content}")

    return "\n\n".join(parts)


# ============================================================
#  B. Generator - Llamada a Hugging Face
# ============================================================
_hf_client: Optional[InferenceClient] = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        token = os.environ["HUGGINGFACE_API_KEY"]
        _hf_client = InferenceClient(token=token)
    return _hf_client


def call_llm(user_question: str, context: str) -> str:
    """Envía el contexto + pregunta a Hugging Face y devuelve la respuesta."""
    client = _get_hf_client()

    user_message = (
        f"CONTEXTO DISPONIBLE:\n{context}\n\n"
        f"PREGUNTA DEL USUARIO:\n{user_question}\n\n"
        "Responde basándote únicamente en el contexto anterior."
    )

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]

    response = client.chat_completion(
        model=HF_MODEL,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# ============================================================
#  C. Grafo LangGraph
# ============================================================

class RAGState(TypedDict):
    """Esquema de estado compartido entre los nodos del grafo."""
    user_input: str
    context:    str
    response:   str


def node_retrieve_info(state: RAGState) -> RAGState:
    """Nodo 1: Recupera documentos relevantes de Supabase."""
    print("\n  🔍 [Nodo 1] Buscando en Supabase...")

    documents = search_supabase(state["user_input"])
    context   = build_context(documents)

    if documents:
        print(f"     ✅ {len(documents)} documento(s) recuperado(s).")
    else:
        print("     ⚠️  Sin documentos relevantes encontrados.")

    return {**state, "context": context}


def node_generate_answer(state: RAGState) -> RAGState:
    """Nodo 2: Genera la respuesta con el LLM de Hugging Face."""
    print("  🤖 [Nodo 2] Generando respuesta con Hugging Face...")

    response = call_llm(state["user_input"], state["context"])

    print("     ✅ Respuesta generada.")
    return {**state, "response": response}


def build_graph() -> StateGraph:
    """Construye y compila el grafo LangGraph."""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve_info",    node_retrieve_info)
    graph.add_node("generate_answer",  node_generate_answer)

    graph.set_entry_point("retrieve_info")
    graph.add_edge("retrieve_info",   "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ============================================================
#  Bucle de conversación por consola
# ============================================================

def run_console_loop() -> None:
    print("=" * 60)
    print("  🏥  Asesor de Seguros RAG - MVP Reto 3")
    print("  Conectado a: Supabase + Hugging Face vía LangGraph")
    print("  Escribe 'salir' o 'exit' para terminar.")
    print("=" * 60)

    rag_graph = build_graph()

    while True:
        try:
            user_input = input("\n💬 Tu pregunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Hasta luego.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"salir", "exit", "quit", "q"}:
            print("👋 Hasta luego.")
            break

        print("\n⚙️  Procesando...")

        initial_state: RAGState = {
            "user_input": user_input,
            "context":    "",
            "response":   "",
        }

        try:
            final_state = rag_graph.invoke(initial_state)
        except Exception as exc:
            print(f"\n❌ Error durante el procesamiento: {exc}")
            continue

        print("\n" + "-" * 60)
        print("📋 RESPUESTA DEL ASESOR:")
        print("-" * 60)
        print(final_state["response"])
        print("-" * 60)


# ============================================================
#  Modo de ejecución: argumento CLI o bucle interactivo
# ============================================================

def main() -> None:
    # Validar variables de entorno obligatorias
    missing = [v for v in ("SUPABASE_URL", "SUPABASE_KEY", "HUGGINGFACE_API_KEY") if not os.getenv(v)]
    if missing:
        print(f"❌ Variables de entorno faltantes: {', '.join(missing)}")
        print("   Crea un archivo .env basado en .env.example")
        sys.exit(1)

    if len(sys.argv) > 1:
        # Pregunta directa por argumento CLI (útil para testing)
        question = " ".join(sys.argv[1:])
        print(f"🔎 Pregunta (modo CLI): {question}\n")

        rag_graph = build_graph()
        initial_state: RAGState = {"user_input": question, "context": "", "response": ""}

        print("⚙️  Procesando...\n")
        final_state = rag_graph.invoke(initial_state)

        print("\n" + "=" * 60)
        print("📋 RESPUESTA:")
        print("=" * 60)
        print(final_state["response"])
    else:
        # Modo interactivo
        run_console_loop()


if __name__ == "__main__":
    main()
