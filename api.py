"""
Servidor FastAPI — expone el agente como API REST.
 
Gestión de sesiones:
  · El historial se guarda en memoria del servidor por session_id.
  · El cliente solo envía su mensaje y session_id — nunca maneja el historial.
  · El historial vive hasta que el proceso se reinicie (suficiente para una sesión de chat).
 
Endpoints:
  POST /ask                     — consulta principal
  DELETE /session/{session_id}  — limpiar historial (cuando el usuario cierra el chat)
  GET  /health                  — health check
 
Arranque:
  python api.py
  uvicorn api:app --reload      — modo desarrollo
"""
 
 
from __future__ import annotations
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from collections import defaultdict
from contextlib import asynccontextmanager
from config import config
from scripts.graph import build_graph, fresh_state
import re

# ── Historial en memoria ──────────────────────────────────────────────────────
# Vive mientras el proceso está corriendo
_session_store: dict[str, list[dict]] = defaultdict(list)

# ── Lifespan ───────────────────────────────────────────────────────── 
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_graph = build_graph()
    yield

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Estimador Agéntico de Copago y Cobertura para el Paciente",
    description="API del agente - Reto 3",
    version="1.0.0",
    lifespan=lifespan
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restringir en producción
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    message:    str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SourceInfo(BaseModel):
    fuente:     str
    similarity: float


class AskResponse(BaseModel):
    answer:     str
    session_id: str        # devuelto para que el front lo persista
    grounded:   bool
    sources:    list[SourceInfo]

# ── Endpoint principal ────────────────────────────────────────────────────────
 
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """
    Flujo:
      1. Recupera el historial de esta sesión desde memoria.
      2. Invoca el grafo RAG con el mensaje actual + historial.
      3. Guarda el historial actualizado (el grafo ya añadió el turno).
      4. Devuelve la respuesta al cliente.
    """
    history = _session_store[request.session_id]
 
    state = fresh_state(user_input=request.message, history=history)
 
    try:
        final_state = app.state.rag_graph.invoke(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error interno del agente.")
 
    # Persistir historial — node_update_history ya agregó el turno actual
    _session_store[request.session_id] = final_state["history"]
 
    sources = _extract_sources(final_state.get("context", ""))
 
    return AskResponse(
        answer=final_state["response"],
        session_id=request.session_id,
        grounded=final_state["grounded"],
        sources=sources,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict:
    """
    Llama a este endpoint cuando el usuario cierre el chat o la página
    para liberar la memoria del historial de esa sesión.
    """
    if session_id in _session_store:
        del _session_store[session_id]
    return {"detail": "Sesión eliminada."}


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "model": config.HF_MODEL, "embedding": config.EMBEDDING_MODEL}
 
 
# ── Helper ────────────────────────────────────────────────────────────────────

def _extract_sources(context_block: str) -> list[SourceInfo]:
    sources = []
    pattern = re.compile(r"(?i)\[(.+?)\s*\|\s*relevancia[:]?\s*([\d.]+)%\]")
    for match in pattern.finditer(context_block):
        sim = float(match.group(2)) / 100
        sources.append(SourceInfo(fuente=match.group(1).strip(), similarity=round(sim, 3)))
    return sources
 
 
# ── Arranque directo ──────────────────────────────────────────────────────────
 
# if __name__ == "__main__":
#     uvicorn.run(
#         "api:app",
#         host=config.API_HOST,
#         port=config.API_PORT,
#         reload=True
#     )
