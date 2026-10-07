-- Demo 2, Step 7: the retrieval function the Agents service calls (course name: retrieval_func_ai).
-- Contract required by Generative AI Agents:
--   inputs : p_query (the user question), top_k
--   returns: SYS_REFCURSOR with fields DOCID, BODY, SCORE (alias if your columns are named differently)
--   the embedding model here MUST match the one that produced text_vec
CREATE OR REPLACE FUNCTION retrieval_func_ai (
  p_query IN VARCHAR2,
  top_k   IN NUMBER
) RETURN SYS_REFCURSOR
IS
  v_results SYS_REFCURSOR;
  v_qvec    VECTOR;
BEGIN
  v_qvec := DBMS_VECTOR.UTL_TO_EMBEDDING(
              p_query,
              JSON('{"provider":"OCIGenAI","credential_name":"OCI_GENAI_CRED",
                     "url":"https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
                     "model":"cohere.embed-v4.0"}'));
  OPEN v_results FOR
    SELECT docid,
           body,
           1 - VECTOR_DISTANCE(text_vec, v_qvec, COSINE) AS score   -- higher = more similar
    FROM   ai_extracted_data_vector
    ORDER  BY score DESC
    FETCH FIRST top_k ROWS ONLY;
  RETURN v_results;
END;
/

-- Test: top 10 for the demo query (the demo confirmed 10 results came back)
DECLARE
  rc    SYS_REFCURSOR;
  v_id  VARCHAR2(100); v_body CLOB; v_score NUMBER;
BEGIN
  rc := retrieval_func_ai('Tell me about Oracle Free Tier Account', 10);
  LOOP
    FETCH rc INTO v_id, v_body, v_score;
    EXIT WHEN rc%NOTFOUND;
    DBMS_OUTPUT.PUT_LINE(v_id || '  ' || ROUND(v_score, 3));
  END LOOP;
  CLOSE rc;
END;
/
