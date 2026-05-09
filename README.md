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
       │  respuesta(HTML)         │  │  · match_documents → Supabase│   │
       └──────────────────────────│  └──────────────┬───────────────┘   │
                                  │                 │ contexto          │
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
├── AGENTS.md            ← Guía para agentes IA
├── Dockerfile           ← Imagen para despliegue de la API
├── .env.example         ← Plantilla de variables de entorno
├── requirements.txt     ← Dependencias Python
├── api.py               ← Servidor FastAPI (/ask, /health, /session/{session_id})
├── config/              ← Configuración centralizada
│   └── config.py
├── db/
│   ├── supabase_setup.sql  ← Schema SQL para Supabase
│   └── populate_db.py      ← Poblar BD con datos de prueba
├── scripts/             ← Núcleo del agente RAG
│   ├── graph.py         ← Definición del Grafo LangGraph
│   ├── retriever.py     ← Búsqueda vectorial (Supabase + embeddings)
│   └── generator.py     ← Generación de respuestas (HF Inference)
└── test/
    ├── console.py       ← Consola interactiva
    └── test_retriever.py  ← Test aislado del retriever
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

Inserta registros de prueba con ciudades, hospitales y reglas de seguros.

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

## Hoja de Ruta de Pruebas

| Paso | Comando | Qué valida |
|------|---------|-----------|
| 1 | `python -m db.populate_db` | Conexión a Supabase + embeddings |
| 2 | `python -m test.test_retriever` | Búsqueda vectorial funciona |
| 3 | `python -m test.console` | Consola interactiva |
| 4 | `python api.py` | Servidor API (Ctrl+C para detener) |

### Preguntas de prueba recomendadas

Prueba 1 (BMI + Cuenca + Gasto):
```
- "Hola, tengo 35 años, vivo en Cuenca y tengo seguro BMI. Llevo dos días con un dolor abdominal agudo y acidez. ¿A dónde puedo ir y cuánto me cuesta?"

- Lo que debes esperar: Que derive a Gastroenterología y te compare el Hospital del Río (Nivel A) con la Clínica Paucarbamba (Nivel B), aplicando el copago fijo de $30 vs $15.

```
Prueba 2 (Saludsa + Guayaquil + Cardiología):
```
"Estoy en Guayaquil, tengo Saludsa. Siento una presión fuerte en el pecho y palpitaciones. ¿Qué opciones tengo?"

Lo que debes esperar: Que derive a Cardiología y compare el Omnihospital/Kennedy (Nivel A+/A calculando el 20/25%) con el San Francisco (Nivel B, 15%).
```

Prueba 3 (Falta Edad y Seguro):
```
"Hola, mi hijo está volando en fiebre, ¿a qué hospital lo llevo en Quito?"

Lo que debes esperar: El agente NO debe recomendar hospitales todavía. Debe preguntar la edad del hijo (para saber si aplica Pediatría/extensión) y con qué aseguradora cuentan.
```
Prueba 4 (Falta Síntoma Claro):
```
"Me siento fatal, quiero ir al médico en Guayaquil usando mi seguro de Saludsa."

Lo que debes esperar: Debe preguntar por los síntomas específicos para poder hacer el triaje, ya que "me siento fatal" no mapea con ninguna especialidad.
```
Prueba 5 (Derivación IESS - Copago $0):
```
"Me acaban de dar una derivación del IESS para un tema de traumatología aquí en Quito. ¿Cuánto es el copago y a dónde voy?"

Lo que debes esperar: Que identifique el convenio IESS, establezca el copago en $0 (¡crucial!) y sugiera el Hospital Vozandes o Clínica Pichincha.
```
Prueba 6 (Regla Estricta de Maternidad):
```
"Soy hombre, tengo BMI en Guayaquil y tengo un dolor agudo en la pelvis, ¿puedo ir a la Clínica Alcívar a que me revisen?"

Lo que debes esperar: El agente debe notar la contradicción. La regla dice "Ginecología/Maternidad -> SOLO si el paciente es mujer". Debería sugerir Medicina General o Gastroenterología en otra clínica, o pedir más detalles.
```
Prueba 7 (Ciudad/Especialidad no cubierta):
```
"Estoy en Manta, tengo seguro Humana y necesito un dentista urgente, ¿qué me recomiendas?"

Lo que debes esperar: El FALLBACK_RESPONSE literal o una disculpa indicando que no tiene información sobre Manta ni odontología en su red actual.
```

---

## Notas Técnicas

| Componente | Detalle |
|-----------|---------|
| Embeddings | `nomic-ai/nomic-embed-text-v1.5` (768 dims, local, sin costo de API) |
| LLM | `meta-llama/Meta-Llama-3-8B-Instruct` vía HF Inference API |
| Threshold | `0.30` (similitud coseno mínima — ajustable en `config/config.py`) |
| Resultados | `5` documentos máximos por consulta |
| Temperatura | `0.1` (respuestas más deterministas y factuales) |

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
