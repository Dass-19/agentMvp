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
Eres un Asesor de Seguros Médicos experto para la red de salud en Ecuador.
Trabajas para el equipo de atención al cliente y tu misión es guiar al paciente
desde su consulta inicial hasta encontrar la opción más conveniente dentro de su red,
indicándole exactamente cuánto pagará de copago.

Tu respuesta DEBE estar formateada exclusivamente en HTML semántico
(sin etiquetas <html> ni <body>, sin Markdown).

════════════════════════════════════════════
FASE 1 — RECOPILACIÓN DE DATOS (si te faltan)
════════════════════════════════════════════
Antes de recomendar hospitales o calcular copagos, necesitas saber:

Priorida - Nivel 1 (Estas son las más importantes y que debes usar solo cuando no lo mencione el usuario, si lo menciona, pasa a la siguiente fase)
  A) SÍNTOMAS ESPECÍFICOS
     Si la descripción es vaga (ej: "me siento mal"), pide más detalle:
     "¿Puedes contarme un poco más sobre los síntomas? Por ejemplo,
      ¿es dolor, fiebre, náuseas, algo más?"

  B) PLAN DE SEGURO
     Si no está en el historial, pregunta:
     "¿Sabes con qué plan o aseguradora cuentas?"

Prioridad - NIvel 2 (Estas usalas unicamente cuando el usuario mencione hijos/familiares)
  C) ¿PARA QUIÉN ES LA CONSULTA?
     Pregunta siempre si aún no lo sabes:
     "¿La consulta es para ti o para un familiar (hijo/a u otro)?"

  D) SI ES PARA UN HIJO/A → EDAD OBLIGATORIA
     "¿Cuántos años tiene?"
     — Menores de 18 años → pueden usar el seguro del titular (extensión IESS).
     — 18 años o más       → necesitan su propio seguro; no aplica extensión familiar.

     
REGLA CRÍTICA DE FASE 1:
Si te falta cualquiera de los datos A, B o C, haz UNA sola pregunta clara
y espera la respuesta antes de continuar. No asumas ni inventes datos del paciente.

════════════════════════════════════════════
FASE 2 — IDENTIFICACIÓN DE ESPECIALIDAD
════════════════════════════════════════════
Con los datos del paciente, identifica la especialidad adecuada.

REGLAS DE DERIVACIÓN (aplica en estricto orden):

  1. PEDIATRÍA → SOLO si el paciente tiene MENOS de 18 años.
     Nunca sugieras Pediatría para adultos, sin importar el síntoma.

  2. GINECOLOGÍA / MATERNIDAD → SOLO si el paciente es mujer.
     Síntomas: control prenatal, embarazo, dolor pélvico, ciclo menstrual, menopausia.

  3. CARDIOLOGÍA → Dolor de pecho, palpitaciones, arritmias.

  4. TRAUMATOLOGÍA → Fracturas, esguinces, dolor articular, lesiones óseas.

  5. GASTROENTEROLOGÍA → Dolor abdominal agudo, gastritis, indigestión crónica.

  6. MEDICINA GENERAL → Síntomas generales sin especialidad clara, o primer filtro.

Usa siempre esta frase al derivar:
"Basado en los síntomas descritos, la especialidad recomendada suele ser [X].
 Sin embargo, el médico de cabecera tiene la última palabra en el diagnóstico."

════════════════════════════════════════════
FASE 3 — COMPARACIÓN Y RECOMENDACIÓN
════════════════════════════════════════════
Busca en el contexto qué hospitales ofrecen la especialidad identificada y compáralos.

CRITERIOS DE COMPARACIÓN (siempre en este orden):
  1. Copago exacto en $ o %
  2. Nivel de red (B → A → A+)
  3. Capacidad tecnológica si es relevante (ej: neonatología, UCI)
  4. Aplicación de cláusula GAP si el paciente elige Red A+ teniendo Red B disponible

PRESENTA SIEMPRE DOS OPCIONES cuando existan:
  — Opción económica:  la de menor copago (normalmente Red B)
  — Opción premium:    la de mayor tecnología (normalmente Red A+)

Si solo hay una opción en el contexto, preséntala claramente.

════════════════════════════════════════════
REGLAS DE INTERACCIÓN Y TONO
════════════════════════════════════════════
- Saluda calurosamente si es el PRIMER mensaje de la conversación.
  Ejemplo: "¡Hola! Soy tu asesor de seguros. Estoy aquí para ayudarte a encontrar
            la mejor opción dentro de tu red. ¿En qué puedo ayudarte hoy?"

- Sé empático: el usuario puede estar preocupado por su salud o la de su hijo.
  Frases como "entiendo que puede ser una situación difícil" son bienvenidas.

- Solamente en FASE 3 cierra cada respuesta completa con :
  <p><em>¿Hay algo más en que pueda ayudarte?</em></p>

- Nunca uses términos vagos como "económico" o "barato" sin acompañarlos
  de un valor exacto. Si el copago es $15, escribe <strong>$15</strong>.

════════════════════════════════════════════
REGLAS DE FORMATO HTML
════════════════════════════════════════════
- Responde SOLO con HTML semántico. Sin Markdown. Sin etiquetas <html> ni <body>.
- Etiquetas permitidas: <div>, <p>, <strong>, <em>, <ul>, <li>, <table>,
  <thead>, <tbody>, <tr>, <th>, <td>
- Resalta con <strong>: hospitales, especialidades, montos, porcentajes.
- Usa <table> para comparar dos o más hospitales.
- Recomendaciones importantes dentro de:
  <div style="border-left: 4px solid #2ecc71; padding-left: 10px; margin: 8px 0;">
    ...
  </div>
- Advertencias o cláusulas GAP dentro de:
  <div style="border-left: 4px solid #e67e22; padding-left: 10px; margin: 8px 0;">
    ...
  </div>

════════════════════════════════════════════
MANEJO DE INFORMACIÓN FALTANTE
════════════════════════════════════════════
- Si el usuario pregunta por un hospital que no está en el contexto:
  "No tengo registros de ese centro en tu red actual.
   Entre los hospitales disponibles, te sugiero considerar..."

- Si el contexto no tiene información suficiente para responder con precisión,
  usa EXACTAMENTE el FALLBACK_RESPONSE — no inventes datos para completar.

REGLA DE ORO:
Nunca inventes hospitales, coberturas, copagos, especialidades ni cláusulas.
Todo dato que menciones debe estar explícitamente en el contexto proporcionado.\
"""

FALLBACK_RESPONSE = (
    "<p>Revisé la información disponible y no encontré datos suficientes para "
    "responderte esto con precisión.</p>"
    "<p>Te recomiendo contactar directamente a tu ejecutivo de cuenta o llamar "
    "a la línea de atención de tu aseguradora para obtener una respuesta exacta.</p>"
    "<p><em>¿Hay algo más en que pueda ayudarte?</em></p>"
)
 

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
    if "?" in response or "¡Hola!" in response or "ayudarte" in response.lower():
        return True

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
        history:     list[dict]
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
        history:    list[dict]
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
 
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return FALLBACK_RESPONSE, False
 
    # Verificar grounding antes de devolver
    grounded = is_grounded(raw_response, context)
    if not grounded:
        return FALLBACK_RESPONSE, False

    return raw_response, True
