"""
Módulo de generación de respuestas con Hugging Face Inference API.
 
Responsabilidades:
  · Mantener el cliente HF como singleton.
  · Armar el payload de mensajes (system + history + context + query).
  · Verificar grounding post-generación (anti-alucinación léxica).
  · Exponer una función limpia generate() que el grafo consume.
"""
 
from __future__ import annotations
import re
from functools import lru_cache
 
from huggingface_hub import InferenceClient
 
from config import config
 
# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Eres un Asesor de Seguros experto para la red de salud en Ecuador. 
Tu respuesta DEBE estar formateada exclusivamente en HTML semántico.

Tu misión es guiar al paciente desde su malestar inicial hasta encontrar 
la opción más conveniente dentro de su red médica.

PROTOCOLO DE ANÁLISIS:
1. IDENTIFICAR:
   - Según el síntoma descrito, sugiere la especialidad médica adecuada.
   - Prioriza identificar correctamente casos relacionados con:
     * Pediatría
     * Ginecología / Maternidad
   - Nunca diagnostiques enfermedades.
   - Usa frases como:
     "Basado en tus síntomas, la especialidad recomendada suele ser..."

2. LOCALIZAR:
   - Busca en el contexto qué hospitales o clínicas ofrecen esa especialidad.
   - Si no aparece disponibilidad explícita, indica:
     "La disponibilidad de especialistas o turnos debe verificarse directamente con el hospital."

3. COMPARAR:
   - Compara hospitales según:
     * Copago
     * Beneficios
     * Nivel tecnológico
     * Tipo de red (Red B vs Red A+)
   - Recomienda:
     * La opción más económica (normalmente Red B)
     * La opción más tecnológica o premium (normalmente Red A+)

REGLAS DE INTERACCIÓN:
- Sé empático pero extremadamente preciso con cifras y coberturas.
- Si el copago es $15, escribe exactamente:
  <strong>$15</strong>
- Nunca uses términos ambiguos como:
  "barato", "económico", "accesible"
  sin acompañarlo de valores exactos.

REGLAS DE FORMATO HTML:
- Responde SOLO con HTML semántico.
- NO uses Markdown.
- NO uses etiquetas <html> ni <body>.
- Usa:
  * <div>
  * <p>
  * <strong>
  * <ul>
  * <li>
  * <table>
  * <thead>
  * <tbody>
  * <tr>
  * <td>

FORMATO VISUAL:
- Resalta hospitales, especialidades y montos usando <strong>.
- Si comparas hospitales o copagos, usa una <table>.
- Las recomendaciones importantes deben ir dentro de:
  <div style='border-left: 4px solid #2ecc71; padding-left: 10px;'>
    ...
  </div>

MANEJO DE INFORMACIÓN FALTANTE:
- Si el usuario pregunta por un hospital que no existe en el contexto:
  "No tengo registros de ese centro en tu red actual, pero en los hospitales disponibles te sugiero..."

- Si falta información crítica o no existe suficiente contexto,
  responde EXACTAMENTE usando FALLBACK_RESPONSE.

REGLA DE ORO:
- Nunca inventes hospitales, coberturas, copagos ni especialidades.
- Si algo no está explícitamente en el contexto, usa FALLBACK_RESPONSE.
"""

FALLBACK_RESPONSE = """
<div style='color: #e74c3c; font-weight: bold;'>
  <p>No encontré información suficiente en mi base de datos para darte una respuesta acorde a tu situación</p>
</div>
"""
 
 
# ── Singleton ─────────────────────────────────────────────────────────────────
 
@lru_cache(maxsize=1)
def _get_hf_client() -> InferenceClient:
    return InferenceClient(token=config.HF_API_KEY)
 
 
# ── Grounding check (anti-alucinación léxica) ─────────────────────────────────
 
def _tokenize(text: str) -> set[str]:
    """Extrae palabras significativas (longitud >= GROUNDING_MIN_WORD_LEN)."""
    words = re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]{%d,}" % config.GROUNDING_MIN_WORD_LEN, text)
    return {w.lower() for w in words}
 
 
def is_grounded(response: str, context: str) -> bool:
    """
    Verifica si la respuesta está anclada léxicamente al contexto.
 
    Lógica:
      · Extrae términos significativos de la respuesta.
      · Calcula qué fracción de ellos aparece también en el contexto.
      · Si el overlap es menor al umbral configurado, asumimos riesgo de alucinación.
 
    Limitaciones conocidas (aceptables para MVP):
      · No detecta alucinaciones semánticamente equivalentes con palabras distintas.
      · Falsos negativos en respuestas muy cortas (pocos tokens → el ratio es ruidoso).
    """
    if not context:
        # Sin contexto recuperado, cualquier respuesta factual es potencialmente inventada
        return False
 
    response_terms = _tokenize(response)
    if not response_terms:
        return True   # respuesta vacía o sin palabras largas — no hay qué chequear
 
    context_terms  = _tokenize(context)
    overlap_ratio  = len(response_terms & context_terms) / len(response_terms)
 
    return overlap_ratio >= config.GROUNDING_OVERLAP_THRESHOLD
 
 
# ── Construcción del payload de mensajes ──────────────────────────────────────
 
def build_messages(
    user_query:  str,
    context:     str,
    history:     list[dict],
) -> list[dict]:
    """
    Construye la lista de mensajes para la API de chat.
 
    Estructura:
      [system] → [historial truncado] → [user: contexto + pregunta actual]
 
    El contexto RAG va solo en el último mensaje de usuario para:
      · Evitar que el historial crezca con bloques de texto enormes.
      · Mantener el contexto siempre fresco (el más relevante para la pregunta actual).
    """
    # Truncar historial a los últimos N turnos
    max_msgs = config.MAX_HISTORY_TURNS * 2   # cada turno = user + assistant
    trimmed_history = history[-max_msgs:] if len(history) > max_msgs else history
 
    # Bloque de usuario: contexto + pregunta
    if context:
        user_content = (
            f"INFORMACIÓN DISPONIBLE EN BASE DE DATOS:\n{context}\n\n"
            f"CONSULTA: {user_query}"
        )
    else:
        user_content = (
            f"CONSULTA: {user_query}\n\n"
            "(No se encontró información relevante en la base de datos para esta consulta.)"
        )
 
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + trimmed_history
        + [{"role": "user", "content": user_content}]
    )
 
 
# ── Función pública ───────────────────────────────────────────────────────────
 
def generate(
    user_query: str,
    context:    str,
    history:    list[dict],
) -> tuple[str, bool]:
    """
    Genera la respuesta del LLM y verifica su grounding.
 
    Returns:
        (response_text, is_grounded_flag)
        El grafo usa is_grounded_flag para métricas futuras.
    """
    client   = _get_hf_client()
    messages = build_messages(user_query, context, history)
 
    try:
        completion = client.chat_completion(
            model=config.HF_MODEL,
            messages=messages,
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
        )
        raw_response = completion.choices[0].message.content.strip()
 
    except Exception:
        return FALLBACK_RESPONSE, False
 
    # Verificar grounding antes de devolver
    grounded = is_grounded(raw_response, context)
    if not grounded:
        return FALLBACK_RESPONSE, False
    return raw_response, True
