"""
console.py
==========
Test interactivo por consola del agente RAG.
Simula exactamente lo que haría el front-end llamando a /ask,
pero corriendo el grafo directamente (sin levantar el servidor HTTP).
 
Uso:
    python console.py
 
Comandos especiales durante la sesión:
    /reset      — limpia el historial de la conversación actual
    /history    — muestra el historial del turno en curso
    /status     — muestra la configuración activa
    salir | exit | q  — termina el programa
"""
 
from __future__ import annotations
import sys
 
from scripts.graph import build_graph, fresh_state
from config import config

# ── Helpers de presentación ───────────────────────────────────────────────────
 
W = 65   # ancho del separador
 
def hr(char: str = "─") -> str:
    return char * W
 
def banner() -> None:
    print()
    print(hr("═"))
    print("  🏥  Asesor de Seguros Médicos — Consola de Pruebas  ")
    print(f"  Modelo LLM:       {config.HF_MODEL}")
    print(f"  Modelo Embedding: {config.EMBEDDING_MODEL}")
    print(f"  Threshold:        {config.MATCH_THRESHOLD}  |  Max docs: {config.MATCH_COUNT}")
    print(hr("═"))
    print("  /reset    limpiar historial    /history    ver historial")
    print("  /status   ver configuración    salir       terminar")
    print(hr("═"))
    print()
 
def print_response(response: str, grounded: bool, turn: int) -> None:
    grounded_tag = "✅ grounded" if grounded else "⚠️  fallback"
    print()
    print(hr())
    print(f"  Asesor [{grounded_tag}]  —  turno {turn}")
    print(hr())
    # Reflow suave: imprimir el texto tal cual (ya viene bien formateado del LLM)
    print(response)
    print(hr())
 
def print_history(history: list[dict]) -> None:
    if not history:
        print("  (historial vacío)")
        return
    print(hr())
    for msg in history:
        role = "TÚ      " if msg["role"] == "user" else "ASESOR  "
        snippet = msg["content"][:120].replace("\n", " ")
        print(f"  {role}: {snippet}{'…' if len(msg['content']) > 120 else ''}")
    print(hr())
 
def print_status() -> None:
    print(hr())
    print(f"  HF_MODEL:             {config.HF_MODEL}")
    print(f"  EMBEDDING_MODEL:      {config.EMBEDDING_MODEL}")
    print(f"  EMBEDDING_DIMENSIONS: {config.EMBEDDING_DIMENSIONS}")
    print(f"  MATCH_THRESHOLD:      {config.MATCH_THRESHOLD}")
    print(f"  MATCH_COUNT:          {config.MATCH_COUNT}")
    print(f"  MAX_CONTEXT_CHARS:    {config.MAX_CONTEXT_CHARS}")
    print(f"  MAX_HISTORY_TURNS:    {config.MAX_HISTORY_TURNS}")
    print(f"  TEMPERATURE:          {config.TEMPERATURE}")
    print(hr())
 
 
# ── Loop principal ────────────────────────────────────────────────────────────
 
def run() -> None:
    # Validar variables de entorno antes de arrancar
    missing = [v for v in ("SUPABASE_URL", "SUPABASE_KEY", "HUGGINGFACE_API_KEY") if not config.__dict__.get(v.replace("HUGGINGFACE_API_KEY", "HF_API_KEY"), None)]
    # Re-check via os.environ directo para evitar falsos negativos
    import os
    missing = [v for v in ("SUPABASE_URL", "SUPABASE_KEY", "HUGGINGFACE_API_KEY") if not os.getenv(v)]
    if missing:
        print(f"\n❌  Variables de entorno faltantes: {', '.join(missing)}")
        print("    Crea un archivo .env basado en .env.example\n")
        sys.exit(1)
 
    banner()
 
    print("  Compilando grafo LangGraph…", end=" ", flush=True)
    rag_graph = build_graph()
    print("listo.\n")
 
    history: list[dict] = []
    turn = 0
 
    while True:
        try:
            raw = input("💬 Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Hasta luego. 👋\n")
            break
 
        if not raw:
            continue
 
        # Comandos especiales
        if raw.lower() in {"salir", "exit", "quit", "q"}:
            print("\n  Hasta luego. 👋\n")
            break
 
        if raw == "/reset":
            history = []
            turn    = 0
            print("  🔄 Historial limpiado. Nueva conversación iniciada.\n")
            continue
 
        if raw == "/history":
            print_history(history)
            continue
 
        if raw == "/status":
            print_status()
            continue
 
        # Turno normal
        turn += 1
        print(f"\n  ⚙️  Procesando turno {turn}…")
 
        state = fresh_state(user_input=raw, history=history)
 
        try:
            final_state = rag_graph.invoke(state)
        except Exception as exc:
            print(f"\n  ❌ Error en el grafo: {exc}\n")
            continue
 
        # Actualizar historial local para el próximo turno
        history = final_state["history"]
 
        print_response(
            response=final_state["response"],
            grounded=final_state["grounded"],
            turn=turn,
        )
 
        # Indicador de contexto recuperado (útil para depuración)
        if final_state.get("context"):
            n_docs = final_state["context"].count("[") if final_state["context"] else 0
            print(f"  📚 {n_docs} fuente(s) consultada(s) en este turno.\n")
        else:
            print("  📭 Sin contexto recuperado en este turno.\n")
 
 
if __name__ == "__main__":
    run()
