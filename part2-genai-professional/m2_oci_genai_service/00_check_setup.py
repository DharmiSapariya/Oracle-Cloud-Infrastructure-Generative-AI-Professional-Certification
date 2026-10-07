"""Diagnose your OCI config *before* calling the Generative AI API.

The course deliberately broke the private key to show the failure mode:
    "the provided key is not a private key or the provided passphrase is incorrect"
and then fixed it via  Profile -> API keys -> delete old key -> Add API key -> new key pair (PEM)
-> copy user OCID / fingerprint / tenancy / region into the config -> point key_file at the new key.

    python part2-genai-professional/m2_oci_genai_service/00_check_setup.py          # static checks
    python part2-genai-professional/m2_oci_genai_service/00_check_setup.py --ping   # + one tiny chat call
"""
from __future__ import annotations

import argparse
import configparser
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.genai_common import chat, get_inference_client, load_settings  # noqa: E402

REQUIRED = ["user", "fingerprint", "tenancy", "region", "key_file"]


def check_config_file(path: str, profile: str) -> list[str]:
    problems: list[str] = []
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return [f"config file not found: {p} (copy config/oci_config.example there)"]
    cp = configparser.ConfigParser()
    cp.read(p)
    if profile not in cp:
        return [f"profile [{profile}] not found in {p} (found: {cp.sections() or 'none'})"]
    section = cp[profile]
    for key in REQUIRED:
        if not section.get(key):
            problems.append(f"missing '{key}' in profile [{profile}]")
        elif "replace_me" in section[key]:
            problems.append(f"'{key}' still has the placeholder value")
    key_file = Path(os.path.expanduser(section.get("key_file", "")))
    if section.get("key_file"):
        if not key_file.exists():
            problems.append(f"key_file does not exist: {key_file}")
        else:
            head = key_file.read_text(errors="ignore").strip()[:40]
            if not head.startswith("-----BEGIN"):
                problems.append(
                    f"key_file {key_file} is empty or not a PEM private key -> "
                    "this is exactly the 'provided key is not a private key' failure from the demo"
                )
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true", help="also send one tiny chat request")
    args = ap.parse_args()
    s = load_settings()
    print(f"config={s.config_file} profile={s.profile}\nendpoint={s.endpoint}\nchat model={s.chat_model_id}")
    issues = check_config_file(s.config_file, s.profile)
    if not s.compartment_id or "replace_me" in s.compartment_id:
        issues.append("OCI_COMPARTMENT_ID is not set in .env")
    if issues:
        print("\nProblems found:")
        for i in issues:
            print("  -", i)
        sys.exit(1)
    print("\nStatic checks passed.")
    if args.ping:
        client = get_inference_client(s)  # validates the config (fingerprint/key/tenancy)
        print("Ping response:", chat(client, s, "Reply with the single word: ready", max_tokens=20))
