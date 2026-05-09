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
# Datos de prueba - hospitales / clausulas / seguros
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    # --- 1. ESPECIALIDADES Y TRIAGE INICIAL (Lógica Global) ---
    {
        "content": (
            "Guía de Especialidades: Dolores abdominales o gastritis -> Gastroenterología. "
            "Fracturas o lesiones óseas -> Traumatología. Dolores de pecho o arritmias -> Cardiología. "
            "Control prenatal o temas ginecológicos -> Ginecología y Obstetricia. "
            "Niños menores de 18 años -> Pediatría (Obligastorio)."
        ),
        "metadata": {"categoria": "triage", "ambito": "general"}
    },

    # --- 2. RED QUITO (Nivel A+, A y B) ---
    {
        "content": (
            "Hospital Metropolitano (Quito, Av. Mariana de Jesús): Nivel A+. Excelencia en alta complejidad. "
            "Copago promedio con Saludsa/BMI: 20% a 30% ($45 mín). Especialidad destacada: Cardiología y Oncología."
        ),
        "metadata": {"hospital": "Metropolitano", "ciudad": "Quito", "red": "A+", "especialidades": ["Cardiología", "Oncología"]}
    },
    {
        "content": (
            "Hospital Vozandes (Quito, Villalengua): Nivel A. Copago fijo promedio para especialistas: $25 - $30. "
            "Reconocido por Medicina Interna y Traumatología. Convenio amplio con aseguradoras nacionales."
        ),
        "metadata": {"hospital": "Vozandes", "ciudad": "Quito", "red": "A", "especialidades": ["Traumatología", "Medicina Interna"]}
    },
    {
        "content": (
            "Hospital de los Valles (Quito, Cumbayá): Nivel A+. Instalaciones premium y alta tecnología. "
            "Copago promedio: 25% ($40 mín). Especialidades destacadas: Pediatría Neonatal y Cirugía Robótica."
        ),
        "metadata": {"hospital": "Hospital de los Valles", "ciudad": "Quito", "red": "A+", "especialidades": ["Pediatría", "Cirugía"]}
    },
    {
        "content": (
            "Clínica Pichincha (Quito, Centro-Norte): Nivel B. Excelente relación costo-beneficio. "
            "Copago fijo: $15 - $20. Especialidad destacada: Medicina General y Gastroenterología. "
            "Ideal para consultas de primera línea y exámenes de rutina sin gastar de más."
        ),
        "metadata": {"hospital": "Clínica Pichincha", "ciudad": "Quito", "red": "B", "especialidades": ["Medicina General", "Gastroenterología"]}
    },

    # --- 3. RED GUAYAQUIL (Nivel A+, A y B) ---
    {
        "content": (
            "Omnihospital (Guayaquil, Av. Juan Tanca Marengo): Nivel A+. Copago especialista: $40 mínimo (25%). "
            "Tecnología de punta en Ginecología y Cuidados Intensivos. Recomendado para casos críticos."
        ),
        "metadata": {"hospital": "Omnihospital", "ciudad": "Guayaquil", "red": "A+", "especialidades": ["Ginecología", "UCI"]}
    },
    {
        "content": (
            "Hospital Kennedy (Sedes Policentro/Samborondón): Nivel A. Copago especialista: $25 mínimo (20%). "
            "Referente en Pediatría y Cardiología en Guayaquil."
        ),
        "metadata": {"hospital": "Kennedy", "ciudad": "Guayaquil", "red": "A", "especialidades": ["Pediatría", "Cardiología"]}
    },
    {
        "content": (
            "Clínica Alcívar (Guayaquil, Sur): Nivel A. Referente nacional en Traumatología y Maternidad. "
            "Copago promedio: 20% ($25 mín). Cuenta con la unidad de trauma más completa y rápida del sur de la ciudad."
        ),
        "metadata": {"hospital": "Clínica Alcívar", "ciudad": "Guayaquil", "red": "A", "especialidades": ["Traumatología", "Ginecología"]}
    },
    {
        "content": (
            "Hospital Clínica San Francisco (Guayaquil, Norte): Nivel B. Opción económica y de rápido acceso. "
            "Copago fijo: $15 para consultas generales y especialistas básicos. Muy recomendado para "
            "Medicina General y triaje inicial."
        ),
        "metadata": {"hospital": "San Francisco", "ciudad": "Guayaquil", "red": "B", "especialidades": ["Medicina General", "Gastroenterología"]}
    },

    # --- 4. RED CUENCA (Nivel A+ y A) ---
    {
        "content": (
            "Hospital Santa Inés (Cuenca, Av. Daniel Córdova): Nivel A+. Líder en el Austro. "
            "Copago estimado: 20% ($35 mín). Especialidades: Traumatología y Neurocirugía."
        ),
        "metadata": {"hospital": "Santa Inés", "ciudad": "Cuenca", "red": "A+", "especialidades": ["Traumatología", "Neurocirugía"]}
    },
    {
        "content": (
            "Hospital Monte Sinaí (Cuenca, Miguel Cordero): Nivel A. Copago fijo: $20 - $25. "
            "Excelente red en Ginecología y Pediatría para la región sur."
        ),
        "metadata": {"hospital": "Monte Sinaí", "ciudad": "Cuenca", "red": "A", "especialidades": ["Ginecología", "Pediatría"]}
    },
    {
        "content": (
            "Hospital Universitario del Río (Cuenca, Autopista Cuenca-Azogues): Nivel A. "
            "Infraestructura moderna y amplia. Copago promedio: 20% ($25 mín). Especialidades destacadas: Cardiología "
            "y Gastroenterología."
        ),
        "metadata": {"hospital": "Hospital del Río", "ciudad": "Cuenca", "red": "A", "especialidades": ["Cardiología", "Gastroenterología"]}
    },
    {
        "content": (
            "Clínica Paucarbamba (Cuenca, Sector Estadio): Nivel B. Atención rápida y precios altamente accesibles. "
            "Copago fijo: $15. Es la opción más solicitada para Pediatría preventiva y Medicina General en el centro de la ciudad."
        ),
        "metadata": {"hospital": "Clínica Paucarbamba", "ciudad": "Cuenca", "red": "B", "especialidades": ["Pediatría", "Medicina General"]}
    },

    # --- 5. PLANES DE SEGURO (Lógica de Negocio) ---
    {
        "content": (
            "Plan Saludsa Pool/Elite: Cobertura en Red A+ al 80% (copago 20%). "
            "En Red B, el copago baja al 15%. Incluye medicina al 70% en Fybeca."
        ),
        "metadata": {"seguro": "Saludsa", "tipo": "Privado"}
    },
    {
        "content": (
            "Plan BMI Igual/Infinite: Cobertura internacional y nacional. "
            "Red A+ (Metropolitano/Omni) copago de $30 fijo en citas médicas. "
            "Resto de clínicas nivel A: copago $15."
        ),
        "metadata": {"seguro": "BMI", "tipo": "Privado"}
    },
    {
        "content": (
            "Convenio IESS (Derivación): Si el paciente es derivado del IESS, el copago es $0. "
            "Aplica solo en prestadores externos con convenio activo (ej: San Francisco en GYE, Vozandes en UIO)."
        ),
        "metadata": {"seguro": "IESS", "tipo": "Publico"}
    }
]

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
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
