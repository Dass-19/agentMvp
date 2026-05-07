-- ============================================================
--  Script SQL — Supabase Setup v2
--  Compatible con Qwen3-Embedding-0.6B (1024 dimensiones)
--  Ejecutar en el SQL Editor de Supabase
-- ============================================================
 
-- 1. Extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;
 
-- 2. Tabla principal de conocimiento
CREATE TABLE IF NOT EXISTS doc_segments (
  id        BIGSERIAL PRIMARY KEY,
  content   TEXT        NOT NULL,
  metadata  JSONB       NOT NULL DEFAULT '{}',
  embedding VECTOR(1024)           -- Qwen3-Embedding-0.6B: 1024 dims
);
 
-- 3. Índice HNSW (mejor recall que ivfflat para datasets pequeños/medianos)
--    ef_construction=128, m=16 son buenos defaults para RAG
CREATE INDEX IF NOT EXISTS doc_segments_embedding_hnsw_idx
  ON doc_segments
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 128);
 
-- 4. Función RPC de búsqueda semántica
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding  VECTOR(1024),
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
--  Referencia del contenido de prueba
--  (los embeddings se generan con populate_db.py, no aquí)
-- ============================================================
-- Hospital Kennedy:      cobertura Pediatría 80%, copago $15 ambulatorio
-- Clínica Alcívar:       cobertura Maternidad 90%, parto normal sin copago
-- Hospital Luis Vernaza: 100% afiliados IESS con autorización previa
-- Clínica San Francisco: red preferente, Medicina General 100%
-- Póliza Premium Salud:  hasta $50.000 anuales, odontología preventiva
-- Cláusula de Copago:    20% cirugía programada, 10% emergencia, máx $1.500/año