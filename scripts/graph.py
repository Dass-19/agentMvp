"""
Definición y compilación del grafo LangGraph.
 
Estado (RAGState):
  · user_input  — pregunta del turno actual
  · history     — historial de conversación (lista de mensajes user/assistant)
  · context     — bloque de contexto recuperado de Supabase
  · response    — respuesta generada por el LLM
  · grounded    — flag de grounding check (útil para métricas)
 
Nodos:
  1. retrieve_info    — busca en Supabase y construye el bloque de contexto
  2. generate_answer  — llama al LLM con context + history
  3. verify_grounding — ya está integrado en generator.generate(); este nodo
                        actualiza el historial y prepara el estado para el próximo turno
 
Flujo:
  retrieve_info → generate_answer → update_history → END
"""
 
from __future__ import annotations
from typing import TypedDict
 
from langgraph.graph import StateGraph, END
 
from scripts import retriever, generator
 
 
# ── Estado compartido ─────────────────────────────────────────────────────────
 
class RAGState(TypedDict):
    user_input: str
    history:    list[dict]   # [{role: "user"|"assistant", content: str}, ...]
    context:    str
    response:   str
    grounded:   bool
 
 
# ── Nodos ─────────────────────────────────────────────────────────────────────
 
def node_retrieve_info(state: RAGState) -> RAGState:
    """
    Nodo 1 — Retriever.
    Busca documentos relevantes en Supabase y construye el contexto para el LLM.
    Si no hay resultados, context quedará vacío (el generador maneja ese caso).
    """
    documents = retriever.search(state["user_input"])
    context   = retriever.build_context_block(documents)
 
    return {**state, "context": context}
 
 
def node_generate_answer(state: RAGState) -> RAGState:
    """
    Nodo 2 — Generator.
    Llama al LLM con el contexto recuperado + historial de conversación.
    El grounding check ocurre dentro de generator.generate().
    """
    response, grounded = generator.generate(
        user_query=state["user_input"],
        context=state["context"],
        history=state["history"],
    )
 
    return {**state, "response": response, "grounded": grounded}
 
 
def node_update_history(state: RAGState) -> RAGState:
    """
    Nodo 3 — History update.
    Agrega el turno actual al historial de conversación.
    Este nodo es separado para que sea fácil extenderlo (ej: persistencia en DB).
    """
    new_history = state["history"] + [
        {"role": "user",      "content": state["user_input"]},
        {"role": "assistant", "content": state["response"]},
    ]
 
    return {**state, "history": new_history}
 
 
# ── Compilación del grafo ─────────────────────────────────────────────────────
 
def build_graph():
    """Construye y compila el grafo. Llamar una vez al arrancar la aplicación."""
    g = StateGraph(RAGState)
 
    g.add_node("retrieve_info",   node_retrieve_info)
    g.add_node("generate_answer", node_generate_answer)
    g.add_node("update_history",  node_update_history)
 
    g.set_entry_point("retrieve_info")
    g.add_edge("retrieve_info",   "generate_answer")
    g.add_edge("generate_answer", "update_history")
    g.add_edge("update_history",  END)
 
    return g.compile()
 
 
# ── Helper: estado inicial limpio ────────────────────────────────────────────
 
def fresh_state(user_input: str, history: list[dict] | None = None) -> RAGState:
    """Crea un RAGState con valores por defecto para un nuevo turno."""
    return RAGState(
        user_input=user_input,
        history=history or [],
        context="",
        response="",
        grounded=False,
    )
 
