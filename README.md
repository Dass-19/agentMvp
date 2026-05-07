# 🏥 MVP RAG por Consola — Reto 3

> **Flujo validado:** Usuario → LangGraph → Supabase (pgvector) → Hugging Face → Consola

---

## Arquitectura

```
┌─────────────┐     pregunta      ┌─────────────────────────────────────┐
│   Consola   │ ─────────────────▶│           LangGraph Graph           │
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
├── requirements.txt    ← Dependencias Python
├── supabase_setup.sql  ← Script SQL (ejecutar en Supabase Editor)
├── populate_db.py      ← Poblar Supabase con datos de prueba
├── test_retriever.py   ← Test aislado del Retriever
└── main.py             ← Agente RAG completo (LangGraph)
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
- **Supabase**: `https://supabase.com/dashboard` → Settings → API
- **Hugging Face**: `https://huggingface.co/settings/tokens` → New token (Read)

### 3. Configurar Supabase

Abre el **SQL Editor** en tu proyecto de Supabase y ejecuta todo el contenido de `supabase_setup.sql`.

Esto:
- Habilita la extensión `pgvector`
- Crea la tabla `doc_segments` con columna `vector(384)`
- Crea el índice `ivfflat` para búsqueda eficiente
- Crea la función RPC `match_documents`

### 4. Poblar la base de datos

```bash
python populate_db.py
```

Inserta 6 registros de prueba sobre hospitales de Guayaquil y cláusulas de póliza.

### 5. Verificar el Retriever (test aislado)

```bash
# Pregunta por defecto
python test_retriever.py

# Pregunta personalizada
python test_retriever.py "¿Cuánto cubre el Hospital Kennedy en pediatría?"
```

Deberías ver los documentos recuperados con su similitud coseno.

### 6. Ejecutar el agente completo

```bash
# Modo interactivo (bucle de conversación)
python main.py

# Modo CLI (una sola pregunta)
python main.py "¿Cuáles son las condiciones del copago en cirugías?"
```

---

## Hoja de Ruta de Pruebas

| Paso | Comando | Qué valida |
|------|---------|-----------|
| 1 | `python populate_db.py` | Conexión a Supabase + embeddings |
| 2 | `python test_retriever.py` | Búsqueda vectorial funciona |
| 3 | `python main.py "pregunta"` | Flujo completo RAG end-to-end |

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
| Embeddings | `all-MiniLM-L6-v2` (384 dims, local, sin costo de API) |
| LLM | `meta-llama/Meta-Llama-3-8B-Instruct` vía HF Inference API |
| Threshold | `0.30` (similitud coseno mínima — ajustable en `main.py`) |
| Resultados | `4` documentos máximos por consulta |
| Temperatura | `0.2` (respuestas más deterministas y factuales) |

---

## Troubleshooting

**`No se encontraron documentos relevantes`**
→ Baja `MATCH_THRESHOLD` a `0.20` en `main.py` y `test_retriever.py`

**`Error 401` en Hugging Face**
→ Verifica que tu token tenga permisos de lectura y está en `.env`

**`relation "doc_segments" does not exist`**
→ Ejecuta `supabase_setup.sql` en el SQL Editor de Supabase

**Modelo de HF no disponible**
→ Cambia `HF_MODEL` en `main.py` a `mistralai/Mistral-7B-Instruct-v0.3`
