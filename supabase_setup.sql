-- ============================================================
--  Script SQL - Configuración Inicial de Supabase (Reto 3)
--  Ejecutar en el SQL Editor de Supabase
-- ============================================================

-- 1. Habilitar la extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Crear la tabla de conocimiento
CREATE TABLE IF NOT EXISTS doc_segments (
  id        BIGSERIAL PRIMARY KEY,
  content   TEXT,                   -- Texto de la póliza o info del hospital
  metadata  JSONB,                  -- Metadatos adicionales (fuente, categoría, etc.)
  embedding VECTOR(384)             -- Tamaño para 'all-MiniLM-L6-v2'
);

-- 3. Índice para búsqueda vectorial rápida (cosine distance)
CREATE INDEX IF NOT EXISTS doc_segments_embedding_idx
  ON doc_segments
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- 4. Función de búsqueda semántica para el RAG
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding  VECTOR(384),
  match_threshold  FLOAT,
  match_count      INT
)
RETURNS TABLE (
  id          BIGINT,
  content     TEXT,
  metadata    JSONB,
  similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    doc_segments.id,
    doc_segments.content,
    doc_segments.metadata,
    1 - (doc_segments.embedding <=> query_embedding) AS similarity
  FROM doc_segments
  WHERE 1 - (doc_segments.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;

-- ============================================================
--  Datos de Prueba - Hospitales de Guayaquil
--  (Insertar DESPUÉS de poblar embeddings con populate_db.py)
-- ============================================================
-- Los embeddings se generan con populate_db.py, no aquí.
-- Este bloque es sólo de referencia del contenido a insertar:

-- Hospital Kennedy: cobertura Pediatría 80%, copago $15 por consulta ambulatoria.
-- Clínica Alcívar: cobertura Maternidad 90%, sin copago en parto normal.
-- Hospital Luis Vernaza: red pública, cobertura 100% para afiliados IESS con autorización previa.
-- Póliza Premium Salud EC: cubre hasta $50,000 anuales, incluye odontología preventiva.
-- Cláusula de Copago General: el asegurado cubre el 20% del valor de cada procedimiento quirúrgico.
