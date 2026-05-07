"""
Script auxiliar para poblar Supabase con datos de prueba.
Genera embeddings locales con sentence-transformers e inserta
los registros en la tabla doc_segments.

Uso:
    python populate_db.py
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# Datos de prueba - hospitales de Guayaquil y cláusulas de póliza
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    # --- ESPECIALIDADES Y SÍNTOMAS ---
    {
        "content": (
            "Guía de Síntomas y Especialidades: Para dolores abdominales agudos, indigestión o gastritis, "
            "la especialidad correspondiente es Gastroenterología. Para fracturas, dolores articulares, "
            "esguinces o lesiones óseas, se debe acudir a Traumatología. Dolores de pecho o palpitaciones "
            "requieren Cardiología."
            "PEDIATRÍA: Fiebre en niños, tos persistente, erupciones cutáneas (sarpullido), "
            "falta de apetito en lactantes, controles de crecimiento y esquemas de vacunación. "
            "MATERNIDAD Y GINECOLOGÍA: Control prenatal, ausencia de periodo (amenorrea), "
            "dolor pélvico fuerte, ecografías obstétricas, asesoría en anticoncepción, "
            "cambios hormonales o síntomas de menopausia."
        ),
        "metadata": {"categoria": "mapeo_sintomas", "uso": "clasificacion_inicial"},
    },
    # --- REDES Y COPAGOS POR HOSPITAL ---
    {
        "content": (
            "Hospital Kennedy (Policentro y Samborondón): Pertenece a la Red Nivel A (Premium). "
            "Especialidades destacadas: Pediatría y Cardiología. El copago para consultas externas con "
            "especialistas es del 20%, con un valor mínimo de $25. Es la opción de mayor costo pero con "
            "la red de especialistas más amplia."
        ),
        "metadata": {"fuente": "Hospital Kennedy", "red": "Nivel A", "ciudad": "Guayaquil"},
    },
    {
        "content": (
            "Hospital Clínica San Francisco: Red Preferente (Nivel B). Especialidad en Medicina General "
            "y Traumatología. Copago fijo de $15 para consultas generales y 15% para especialistas. "
            "Recomendado como opción económica para consultas ambulatorias y rayos X."
        ),
        "metadata": {"fuente": "Clínica San Francisco", "red": "Nivel B", "ciudad": "Guayaquil"},
    },
    {
        "content": (
            "Clínica Alcívar: Centro especializado en Maternidad y Ginecología. El plan cubre el 100% "
            "de controles prenatales. Para partos, el copago es de $0 si es natural y un deducible "
            "fijo de $200 para cesáreas programadas en red preferente."
        ),
        "metadata": {"fuente": "Clínica Alcívar", "categoria": "maternidad", "ciudad": "Guayaquil"},
    },
    {
    "content": (
            "Omnihospital (Norte de Guayaquil): Red Nivel A+. Especializado en alta complejidad. "
            "En Ginecología y Maternidad, ofrece tecnología de punta con un copago del 25% ($40 mínimo). "
            "Para Pediatría, cuenta con emergencia 24/7 con copago de $30. Es la opción recomendada "
            "para casos que requieran hospitalización tecnológica o cuidados intensivos."
        ),
        "metadata": {"fuente": "Omnihospital", "red": "Nivel A+", "ciudad": "Guayaquil"},
    },
    {
        "content": (
            "Clínica Panamericana (Centro de Guayaquil): Red Nivel B. Excelente relación costo-beneficio. "
            "Cubre Ginecología con copago fijo de $18 para consultas preventivas. En Pediatría, "
            "las consultas ambulatorias tienen un copago del 15%. No cuenta con unidad de neonatología "
            "de alta complejidad, pero es ideal para controles de rutina y consultas externas."
        ),
        "metadata": {"fuente": "Clínica Panamericana", "red": "Nivel B", "ciudad": "Guayaquil"},
    },
    # --- DETALLES DE PÓLIZA ---
    {
        "content": (
            "Plan Salud Pro-Ecuador: Cobertura anual de $60,000. Deducible anual de $500. "
            "Medicamentos en farmacias de red (Fybeca/SanaSana) tienen descuento del 70%. "
            "Urgencias en hospitales de Red Nivel B no requieren pre-autorización."
        ),
        "metadata": {"fuente": "Póliza Pro-Ecuador", "categoria": "beneficios_generales"},
    },
    {
        "content": (
            "Cláusula de Emergencias: En caso de accidente o dolor súbito e intenso, el copago de "
            "emergencia en cualquier hospital de la red es del 10% del valor de la factura, "
            "independientemente del nivel de la red (A o B)."
        ),
        "metadata": {"fuente": "Contrato Art. 15", "categoria": "emergencias"},
    },
    # --- CLÁUSULAS DE SEGURO ---
    {
        "content": (
            "Póliza Salud Pro-Ecuador: El copago se calcula sobre el valor negociado con el hospital. "
            "Si el paciente elige un hospital de Red A+ teniendo disponible Red B para el mismo síntoma, "
            "el seguro cubrirá solo hasta el monto de la Red B, y el paciente asumirá la diferencia (Gap)."
        ),
        "metadata": {"fuente": "Manual del Asegurado", "categoria": "clausula_gap"},
    }
    ]

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"   # 384 dimensiones, local y rápido
TABLE_NAME = "doc_segments"


def main() -> None:
    # Conexión a Supabase
    url: str = os.environ["SUPABASE_URL"]
    key: str = os.environ["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)

    print(f"🔗 Conectado a Supabase: {url}")

    # Cargar modelo de embeddings
    print(f"📦 Cargando modelo '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Insertar registros
    print(f"\n📝 Insertando {len(KNOWLEDGE_BASE)} registros en '{TABLE_NAME}'...\n")
    for i, item in enumerate(KNOWLEDGE_BASE, start=1):
        embedding: list[float] = model.encode(item["content"]).tolist()
        record = {
            "content":   item["content"],
            "metadata":  item["metadata"],
            "embedding": embedding,
        }
        response = supabase.table(TABLE_NAME).insert(record).execute()
        fuente = item["metadata"].get("fuente", "—")
        print(f"  [{i}/{len(KNOWLEDGE_BASE)}] ✅ Insertado: {fuente}")

    print("\n🎉 Base de conocimiento poblada exitosamente.")


if __name__ == "__main__":
    main()
