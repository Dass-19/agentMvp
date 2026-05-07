"""
populate_db.py
==============
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
    {
        "content": (
            "Hospital Kennedy en Guayaquil ofrece cobertura del 80% en el área de "
            "Pediatría para asegurados de la red. El copago del paciente es de $15 por "
            "consulta ambulatoria y $30 por consulta con especialista pediátrico."
        ),
        "metadata": {"fuente": "Hospital Kennedy", "categoria": "cobertura_pediatria", "ciudad": "Guayaquil"},
    },
    {
        "content": (
            "Clínica Alcívar cubre el 90% de los gastos de Maternidad incluyendo parto "
            "normal y cesárea de emergencia. El parto normal no genera copago; la cesárea "
            "programada tiene un copago fijo de $200."
        ),
        "metadata": {"fuente": "Clínica Alcívar", "categoria": "cobertura_maternidad", "ciudad": "Guayaquil"},
    },
    {
        "content": (
            "Hospital Luis Vernaza es hospital público de tercer nivel en Guayaquil. "
            "Los afiliados al IESS tienen cobertura del 100% siempre que presenten "
            "autorización previa emitida por el IESS antes de cualquier procedimiento."
        ),
        "metadata": {"fuente": "Hospital Luis Vernaza", "categoria": "red_publica", "ciudad": "Guayaquil"},
    },
    {
        "content": (
            "Póliza Premium Salud EC cubre hasta $50,000 anuales por asegurado. "
            "Incluye odontología preventiva (dos limpiezas anuales sin copago), "
            "hospitalización ilimitada dentro de la red y medicamentos con 70% de descuento."
        ),
        "metadata": {"fuente": "Póliza Premium Salud EC", "categoria": "descripcion_poliza"},
    },
    {
        "content": (
            "Cláusula de Copago General (Art. 12): el asegurado asume el 20% del costo "
            "total de cada procedimiento quirúrgico programado. En procedimientos de "
            "emergencia el copago se reduce al 10%. El copago máximo anual es de $1,500."
        ),
        "metadata": {"fuente": "Contrato de Póliza", "categoria": "clausula_copago"},
    },
    {
        "content": (
            "Hospital Clínica San Francisco en Guayaquil pertenece a la red preferente "
            "de la póliza. Cubre consultas de Medicina General al 100%, laboratorio al "
            "80% e imagenología al 75%. No requiere referencia médica previa para urgencias."
        ),
        "metadata": {"fuente": "Clínica San Francisco", "categoria": "red_preferente", "ciudad": "Guayaquil"},
    },
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
