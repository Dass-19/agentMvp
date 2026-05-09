# AGENTS.md — Guía para Agentes IA

Este documento contiene información útil para trabajar en este proyecto.

---

## Comandos de Desarrollo

```bash
# Instalar dependencias
pip install -r requirements.txt

# Poblar base de datos
python -m db.populate_db

# Consola interactiva
python -m test.console

# Test retriever
python -m test.test_retriever

# Servidor API
python api.py

# Verificar con lint/typecheck (si disponible)
ruff check .
mypy .
```

---

## Estructura del Proyecto

| Archivo | Responsabilidad |
|---------|-----------------|
| `api.py` | Servidor FastAPI, endpoints `/ask` y `/health` |
| `config/config.py` | Configuración centralizada |
| `scripts/graph.py` | Definición del Grafo LangGraph |
| `scripts/retriever.py` | Búsqueda vectorial (Qwen + Supabase) |
| `scripts/generator.py` | Generación + grounding check |
| `db/supabase_setup.sql` | Schema de base de datos |
| `db/populate_db.py` | Poblar datos de prueba |

---

## Variables de Entorno Requeridas

- `SUPABASE_URL` — URL del proyecto Supabase
- `SUPABASE_KEY` — API key (Secret key)
- `HUGGINGFACE_API_KEY` — Token HF con permisos de lectura

---

## Modelos Configurados

| Componente | Modelo | Dimensiones |
|------------|--------|-------------|
| Embedding | `nomic-ai/nomic-embed-text-v1.5` | 768 |
| LLM | `meta-llama/Meta-Llama-3-8B-Instruct` | — |

---

## Configuración Clave

Parámetros ajustables en `config/config.py`:

- `MATCH_THRESHOLD` — Similitud mínima (default: 0.30)
- `MATCH_COUNT` — Docs máxima a recuperar (default: 5)
- `MAX_CONTEXT_CHARS` — Límite de contexto para LLM (default: 3000)
- `TEMPERATURE` — Creatividad del LLM (default: 0.25)
- `GROUNDING_OVERLAP_THRESHOLD` — Umbral anti-alucinación (default: 0.28)

---

## API

### POST /ask

Body:

```json
{
  "message": "...",
  "session_id": "..."
}
```

Ejemplo válido:

```json
{
  "message": "Tengo fiebre alta y dolor de cabeza, ¿qué hospital recomiendas?",
  "session_id": "user-123"
}
```

Respuesta:

```json
{
  "answer": "<div>...</div>",
  "grounded": true,
  "sources": [
    {"fuente": "...", "categoria": "", "similarity": 0.42}
  ]
}
```

Notas:
- `answer` siempre viene en HTML semántico.
- El historial lo maneja el backend por `session_id`.

---

## testing

### Prerequisites

1. Ejecutar `db/supabase_setup.sql` en Supabase SQL Editor
2. Poblar datos: `python -m db.populate_db`

### Consola Interactiva

```bash
python -m test.console
```

Comandos especiales durante la sesión:
- `/reset` — Limpiar historial
- `/history` — Ver historial
- `/status` — Ver configuración actual
- `salir` — Terminar

---

## Notas

- El retriever usa `normalize_embeddings=True` (similitud coseno)
- El grounding check verifica overlap léxico entre respuesta y contexto
- El historial se maneja en backend por `session_id`
- La base de conocimientos incluye más ciudades, hospitales y reglas de seguros
