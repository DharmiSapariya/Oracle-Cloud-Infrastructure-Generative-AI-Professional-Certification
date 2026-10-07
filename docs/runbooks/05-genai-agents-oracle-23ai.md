# Runbook 05 - Agent with an Oracle Database 23ai knowledge base (Demo 2)

Prerequisites (set up beforehand): IAM policies, a **VCN** with updated security rules, a **Vault** + key for database secrets.

## Database
- [ ] Create Autonomous Database: Data Warehouse, Serverless, **23ai**, admin password, network access **Private endpoint only** (pre-created VCN + subnet), **mTLS not required**, valid email. Wait for *Available*.
- [ ] Copy a **connection string** and the **private endpoint IP**.

## Database tools connection
- [ ] *Developer Services -> Database Tools -> Connections -> Create*: name, compartment, **Oracle Autonomous Database**, select the DB, user `admin`.
- [ ] Create the **password secret** (name, Vault, key, the admin password).
- [ ] Edit the connection string: retry count **20 -> 3**; replace host with the **private IP**; choose the private endpoint; wallet/SSL = none. **Create -> Validate**.

## Load vector data (SQL worksheet) - files in `part2-genai-professional/m4_genai_agents/sql/`
- [ ] `01_acl_and_credentials.sql` - ACL + credentials (fill in placeholders; never commit real values).
- [ ] `02_test_embedding.sql` - embed "Hello" to prove credentials and model work.
- [ ] Object Storage: create a **Pre-Authenticated Request** (with expiry) for `faq.txt`.
- [ ] `03_chunk_and_vectorize.sql` - paste the PAR link; creates `AI_EXTRACTED_DATA` and `AI_EXTRACTED_DATA_VECTOR`.
- [ ] `04_retrieval_function.sql` - creates `retrieval_func_ai(p_query, top_k)`; test with *"Tell me about Oracle Free Tier Account"*, top 10.

> These SQL files follow the demo but were not executed in this repo's CI. Verify package/function names against the current Oracle Database 23ai documentation, and make sure the embedding model in SQL is currently available.

## Agent
- [ ] *Knowledge bases -> Create*: data store type **Oracle AI Vector Search**, choose the database tools connection (test), choose the retrieval function, Create (Active).
- [ ] *Agents -> Create*: name, custom welcome message, knowledge base, auto-create endpoint, accept terms.
- [ ] *Chat* -> *"Tell me about Oracle Free Tier"* -> check citations and trace.
