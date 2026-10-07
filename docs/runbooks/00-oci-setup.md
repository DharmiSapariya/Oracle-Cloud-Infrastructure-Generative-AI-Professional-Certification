# Runbook 00 - OCI account, API keys and local setup

Goal: be able to run every script in `part2-genai-professional/` against OCI Generative AI.

- [ ] **Account**: sign up for OCI (Free Tier is enough for exploring; GenAI inference is billed per use, dedicated clusters are billed per unit-hour - see the cost calculator).
- [ ] **Region**: Generative AI is only available in selected regions (the course used Frankfurt; the cluster demo used Chicago). Check availability in the docs first and set `OCI_GENAI_ENDPOINT` to `https://inference.generativeai.<region>.oci.oraclecloud.com`.
- [ ] **Compartment**: note its OCID -> `OCI_COMPARTMENT_ID`.
- [ ] **API key**: Console -> *Profile -> My profile -> API keys -> Add API key -> Generate API key pair*. Download the private key (PEM), then copy the **configuration file preview**.
- [ ] **Config file**: save the preview as `~/.oci/config` (template: `part2-genai-professional/m2_oci_genai_service/config/oci_config.example`) and point `key_file` at the downloaded PEM. `chmod 600` the key.
- [ ] **Python env**: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements-oci.txt`
- [ ] **.env**: `cp .env.example .env` and fill in compartment OCID, model IDs (see `docs/keeping-code-current.md`).
- [ ] **Verify**: `python part2-genai-professional/m2_oci_genai_service/00_check_setup.py --ping`

### Troubleshooting (from the course demo)
| Symptom | Cause | Fix |
|---|---|---|
| "provided key is not a private key or the provided passphrase is incorrect" | empty/corrupt key file | delete the API key, add a new one, replace config values and `key_file` |
| 401 / NotAuthenticated | wrong fingerprint, tenancy or user OCID | re-copy the config preview |
| 404 / NotAuthorizedOrNotFound | wrong compartment, model not available in region, missing IAM policy | check region + policies |
| Model not found | model retired | use a current model ID |

Never commit `.env`, `*.pem` or `~/.oci/config` (they are in `.gitignore`).
