-- Demo 2, Steps 5-6: chunk a file from Object Storage and vectorize the chunks.
-- Prereq: create a Pre-Authenticated Request (PAR) link for faq.txt (set an expiry) and paste it below.

-- 1) Chunk -> table AI_EXTRACTED_DATA (chunk_id, chunk_offset, chunk_length, chunk_data)
CREATE TABLE ai_extracted_data AS
SELECT j.chunk_id, j.chunk_offset, j.chunk_length, j.chunk_data
FROM (SELECT DBMS_CLOUD.GET_OBJECT(
               credential_name => NULL,
               object_uri      => 'PASTE_YOUR_PAR_LINK_HERE') AS blob_data FROM dual) t,
     TABLE(DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS(
             DBMS_VECTOR_CHAIN.UTL_TO_TEXT(t.blob_data),
             JSON('{"by":"words","max":"100","overlap":"10","split":"recursively","normalize":"all"}'))) c,
     JSON_TABLE(c.column_value, '$'
       COLUMNS (chunk_id     NUMBER PATH '$.chunk_id',
                chunk_offset NUMBER PATH '$.chunk_offset',
                chunk_length NUMBER PATH '$.chunk_length',
                chunk_data   CLOB   PATH '$.chunk_data')) j;

SELECT chunk_id, SUBSTR(chunk_data, 1, 80) AS preview FROM ai_extracted_data;

-- 2) Vector table. Required by Agents: DOCID, BODY, vector. VECTOR may be declared with or without dimensions.
CREATE TABLE ai_extracted_data_vector (
  docid    VARCHAR2(100) PRIMARY KEY,
  body     CLOB,
  text_vec VECTOR
);

-- 3) Embed every chunk (same model as the retrieval function!)
INSERT INTO ai_extracted_data_vector (docid, body, text_vec)
SELECT 'faq-' || chunk_id,
       chunk_data,
       DBMS_VECTOR.UTL_TO_EMBEDDING(
         chunk_data,
         JSON('{"provider":"OCIGenAI","credential_name":"OCI_GENAI_CRED",
                "url":"https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
                "model":"cohere.embed-v4.0"}'))
FROM ai_extracted_data;
COMMIT;

SELECT docid, SUBSTR(body, 1, 60) AS body_preview FROM ai_extracted_data_vector;
