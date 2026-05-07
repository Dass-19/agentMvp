"""
api.py
======
Servidor FastAPI — expone el agente RAG como API REST.
 
Endpoint principal:
  POST /ask
    Body:  { "message": "...", "history": [...] }
    200:   { "answer": "...", "grounded": true, "sources": [...] }
    422:   Validation error
    500:   Internal server error
 
Arranque:
  python api.py                       # usa host/port de .env
  uvicorn api:app --reload            # modo desarrollo
"""
 
from __future__ import annotations
 
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
 
from config import config
from scripts.graph import build_graph, fresh_state

# ── Grafo (singleton) ─────────────────────────────────────────────────────────
_rag_graph = None
 
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

class ChatMessage(BaseModel):
    role:    str = Field(..., description="Rol del mensaje: user o assistant")
    content: str = Field(..., min_length=1, description="Contenido del mensaje")


class AskRequest(BaseModel):
    message:    str  = Field(..., min_length=1, max_length=2000, description="Pregunta del usuario")
    history:    list[ChatMessage] = Field(default_factory=list, description="Historial de conversación")
 
 
class SourceInfo(BaseModel):
    fuente:     str
    categoria:  str
    similarity: float
 
 
class AskResponse(BaseModel):
    answer:     str
    grounded:   bool
    sources:    list[SourceInfo]
 
 
# ── Endpoint principal ────────────────────────────────────────────────────────
 
@app.post("/ask", response_model=AskResponse, summary="Consulta al asesor")
async def ask(request: AskRequest) -> AskResponse:
    """
    Procesa una pregunta del usuario y devuelve la respuesta del asesor.
 
    - Usa el historial enviado por el cliente (si aplica).
    - Recupera contexto relevante de Supabase antes de generar la respuesta.
    - Incluye un check de grounding anti-alucinación.
    """
    state = fresh_state(user_input=request.message, history=request.history)
 
    try:
        final_state = app.state.rag_graph.invoke(state)
    except Exception:
        raise HTTPException(status_code=500, detail="Error interno del agente.")
 
    # Extraer fuentes del contexto para incluirlas en la respuesta
    # (el contexto ya fue procesado; re-buscamos solo para el campo sources)
    sources = _extract_sources(final_state.get("context", ""))
 
    return AskResponse(
        answer=final_state["response"],
        grounded=final_state["grounded"],
        sources=sources,
    )
 
 
@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok", "model": config.HF_MODEL, "embedding": config.EMBEDDING_MODEL}
 
 
# ── Helper ────────────────────────────────────────────────────────────────────
 
def _extract_sources(context_block: str) -> list[SourceInfo]:
    """
    Extrae información de fuentes del bloque de contexto formateado.
    El bloque tiene el formato: [Fuente X | relevancia Y%]\ncontent...
    """
    import re
    sources = []
    pattern = re.compile(r"\[(.+?)\s*\|\s*relevancia\s*([\d.]+)%\]")
    for match in pattern.finditer(context_block):
        fuente_raw = match.group(1).strip()
        sim        = float(match.group(2)) / 100
        sources.append(SourceInfo(fuente=fuente_raw, categoria="", similarity=round(sim, 3)))
    return sources
 
 
# ── Arranque directo ──────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
