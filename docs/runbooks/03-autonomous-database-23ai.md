# Runbook 03 - Autonomous Database 23ai for the RAG lab

- [ ] Console -> **Autonomous Database -> Create**: display + database name, compartment, workload **Data Warehouse**, deployment **Serverless**, version **23ai**.
- [ ] Set the **admin password**.
- [ ] Network access: **Secure access from allowed IPs** -> add your public IP. Leave **mTLS unchecked** (so `oracledb.connect(user, password, dsn)` works without a wallet).
- [ ] Create. When *Available*: **Database connection** -> copy a connection string -> `DB_DSN` in `.env`; `DB_USER=ADMIN`, `DB_PASSWORD=<admin password>`.
- [ ] Confirm the IP appears in the access control list.
- [ ] Run ingestion: `python part2-genai-professional/m3_rag_langchain/02_ingest_pdf_to_oracle_vs.py --file <your.pdf>` (add `--dry-run` first).
- [ ] Inspect the table in SQL Developer / Database Actions - 4 columns: primary key, text, metadata, embedding.
- [ ] Query: `python .../03_query_rag_chain.py --question "Tell us about Module 4 of AI Foundation Certification Course."`
- [ ] Chatbot: `python .../04_conversational_rag.py`

The same distance strategy (`DOT_PRODUCT` by default here) must be used at ingestion and query time, and the same embedding model.
