-- ============================================================
--  Script SQL — Supabase Setup v2
--  Compatible con nomic-embed-text-v1.5 (768 dimensiones)
--  Ejecutar en el SQL Editor de Supabase
-- ============================================================
 
-- 1. Extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;
 
-- 2. Tabla principal de conocimiento
CREATE TABLE IF NOT EXISTS doc_segments (
  id        BIGSERIAL PRIMARY KEY,
  content   TEXT        NOT NULL,
  metadata  JSONB       NOT NULL DEFAULT '{}',
  embedding VECTOR(768)           -- nomic-ai/nomic-embed-text-v1.5: 768 dims
);
 
-- 3. Índice HNSW (mejor recall que ivfflat para datasets pequeños/medianos)
CREATE INDEX IF NOT EXISTS doc_segments_embedding_hnsw_idx
  ON doc_segments
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 128);
 
-- 4. Función RPC de búsqueda semántica
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding  VECTOR(768),
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