# 🏥 MVP RAG Backend — Reto 3

> **Flujo validado:** Usuario/Front → API (/ask) → LangGraph → Supabase (pgvector) → Hugging Face → Respuesta HTML

---

## Arquitectura

```
┌─────────────┐     pregunta      ┌─────────────────────────────────────┐
│  Front/API  │ ─────────────────▶│           LangGraph Graph           │
│  (usuario)  │                   │                                     │
└─────────────┘                   │  ┌──────────────────────────────┐   │
       ▲                          │  │  Nodo 1: retrieve_info       │   │
       │                          │  │  · Embedding local (HF STF)  │   │
       │  respuesta                │  │  · match_documents → Supabase│   │
       └──────────────────────────│  └──────────────┬───────────────┘   │
                                  │                 │ contexto           │
                                  │  ┌──────────────▼───────────────┐   │
                                  │  │  Nodo 2: generate_answer     │   │
                                  │  │  · System Prompt + Contexto  │   │
                                  │  │  · Llama-3-8B (HF Inference) │   │
                                  │  └──────────────────────────────┘   │
                                  └─────────────────────────────────────┘
```

---

## Estructura del Proyecto

```
rag_mvp/
├── .env.example        ← Plantilla de variables de entorno
├── .env                ← Credenciales (ignorado por git)
├── requirements.txt    ← Dependencias Python
├── api.py              ← Servidor FastAPI (/ask, /health)
├── config/             ← Configuración centralizada
│   └── config.py
├── db/
│   ├── supabase_setup.sql  ← Schema SQL para Supabase
│   └── populate_db.py      ← Poblar BD con datos de prueba
├── scripts/            ← Núcleo del agente RAG
│   ├── graph.py        ← Definición del Grafo LangGraph
│   ├── retriever.py   ← Búsqueda vectorial (Supabase + Qwen)
│   └── generator.py   ← Generación de respuestas (HF Inference)
├── test/
│   ├── console.py     ← Consola interactiva
│   └── test_retriever.py  ← Test aislado del retriever
```

---

## Setup Paso a Paso

### 1. Clonar e instalar dependencias

```bash
# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus credenciales reales
```

Obtén tus credenciales en:
- **Supabase**: `https://supabase.com/dashboard` → Settings → API (Secret key)
- **Hugging Face**: `https://huggingface.co/settings/tokens` → New token (Read)

### 3. Configurar Supabase

Abre el **SQL Editor** en tu proyecto de Supabase y ejecuta todo el contenido de `db/supabase_setup.sql`.

Esto:
- Habilita la extensión `pgvector`
- Crea la tabla `doc_segments` con columna `vector(1024)`
- Crea el índice `hnsw` para búsqueda eficiente
- Crea la función RPC `match_documents`

### 4. Poblar la base de datos

```bash
python -m db.populate_db
```

Inserta 6 registros de prueba sobre hospitales de Guayaquil y cláusulas de póliza.

### 5. Verificar el Retriever (test aislado)

```bash
python -m test.test_retriever
```

### 6. Ejecutar el agente completo

```bash
# Consola interactiva
python -m test.console

# O iniciar el servidor API
uvicorn api:app
```

---

## API

### POST /ask

Body:

```json
{
  "message": "...",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Ejemplo válido:

```json
{
  "message": "Tengo fiebre alta y dolor de cabeza, ¿qué hospital recomiendas?",
  "history": [
    {"role": "user", "content": "Hola"},
    {"role": "assistant", "content": "<div>...</div>"}
  ]
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
- El historial lo maneja el cliente (no hay sesiones en el backend).

---

## Hoja de Ruta de Pruebas

| Paso | Comando | Qué valida |
|------|---------|-----------|
| 1 | `python -m db.populate_db` | Conexión a Supabase + embeddings |
| 2 | `python -m test.test_retriever` | Búsqueda vectorial funciona |
| 3 | `python -m test.console` | Consola interactiva |
| 4 | `python api.py` | Servidor API (Ctrl+C para detener) |

### Preguntas de prueba recomendadas

```
¿Cuál es la cobertura de pediatría en Hospital Kennedy?
¿Tiene copago el parto normal en Clínica Alcívar?
¿Qué necesito para atenderme en Hospital Luis Vernaza siendo afiliado IESS?
¿Cuánto cubre la Póliza Premium Salud EC anualmente?
¿Cuál es el porcentaje de copago en cirugías programadas?
```

---

## Notas Técnicas

| Componente | Detalle |
|-----------|---------|
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` (1024 dims, local, sin costo de API) |
| LLM | `meta-llama/Meta-Llama-3-8B-Instruct` vía HF Inference API |
| Threshold | `0.30` (similitud coseno mínima — ajustable en `config/config.py`) |
| Resultados | `5` documentos máximos por consulta |
| Temperatura | `0.25` (respuestas más deterministas y factuales) |

---

## Troubleshooting

**`No se encontraron documentos relevantes`**
→ Baja `MATCH_THRESHOLD` a `0.20` en `config/config.py`

**`Error 401` en Hugging Face**
→ Verifica que tu token tenga permisos de lectura y está en `.env`

**`relation "doc_segments" does not exist`**
→ Ejecuta `supabase_setup.sql` en el SQL Editor de Supabase

**Modelo de HF no disponible**
→ Cambia `HF_MODEL` en `config/config.py` a `mistralai/Mistral-7B-Instruct-v0.3`
