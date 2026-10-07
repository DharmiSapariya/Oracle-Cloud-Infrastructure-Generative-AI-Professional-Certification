-- Demo 2, Step 4 (part 3): prove the credential works by embedding the word "Hello".
-- The embedding model used here MUST be the same one used later by the retrieval function.
-- The course used cohere.embed-multilingual-v3.0 (retired on-demand since); use a current model id.
SELECT DBMS_VECTOR.UTL_TO_EMBEDDING(
         'Hello',
         JSON('{
           "provider": "OCIGenAI",
           "credential_name": "OCI_GENAI_CRED",
           "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
           "model": "cohere.embed-v4.0"
         }')) AS embedding
FROM dual;
