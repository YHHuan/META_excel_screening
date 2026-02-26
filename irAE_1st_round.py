#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
irAE Paper Screening Tool — 1st Round Extraction
=================================================
Reads DOCX records, sends each one to an LLM via OpenRouter,
and extracts structured irAE data into a CSV file.

Usage:
    python irAE_1st_round.py

Requirements:
    pip install -r requirements.txt

Environment (.env file in same directory):
    OPENAI_API_KEY=sk-or-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    OPENAI_API_BASE=https://openrouter.ai/api/v1

Outputs:
    irAE_extraction.csv                # appended per batch
    .checkpoint.json                   # completed record_ids
    failed_batches/failed_batch_#.txt  # raw replies that failed parsing
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Optional, Union, Dict, Set

import pandas as pd
from dotenv import load_dotenv
from docx import Document
from pydantic import BaseModel, field_validator, ValidationError
from openai import OpenAI

try:
    import tiktoken
except ImportError:
    tiktoken = None

# ───────────────────────── Configurable Parameters ─────────────────────────

# START_INDEX: 0-based index to force-start from a specific record.
# Set to None to resume from checkpoint automatically.
START_INDEX: Optional[int] = None   # e.g. 9 = start at record #10; None = use checkpoint

# File paths
ROOT        = Path(__file__).parent   # directory containing this script
DOC_FILES   = ["data/records.docx", "data/records-2.docx"]
CSV_PATH    = ROOT / "results" / "irAE_extraction.csv"
CKPT_PATH   = ROOT / ".checkpoint.json"
FAILED_DIR  = ROOT / "failed_batches"
FAILED_DIR.mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

# Model & rate settings
MODEL_ID        = "google/gemini-2.5-flash-preview-05-20"
MAX_BATCH_TOK   = 12_000
RETRY_LIMIT     = 2
BATCH_SIZE_INIT = 1
SLEEP_BETWEEN   = 0.3
PRICE_PER_M     = 0.0   # set to 0 for free-tier models; update for paid models

# ───────────────────────── Token Estimation ─────────────────────────────────

def num_tokens(text: str) -> int:
    if tiktoken:
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    return max(1, len(text) // 4)

# ───────────────────────── Pydantic Schema ──────────────────────────────────

class ExtractionSchema(BaseModel):
    record_id: str

    # Article metadata
    name:    Optional[str]   = None   # first author
    DOI:     Optional[str]   = None   # trial ID or DOI (e.g. NCT...)
    year:    Optional[int]   = None   # publication year
    journal: Optional[str]  = None   # journal name

    # Core fields
    cancer_type:      Optional[str]       = None
    stage:            Optional[str]       = None
    comparison_arms:  Optional[List[str]] = None

    discusses_irAE:          str
    irAE_subgroup_analysis:  str
    irAE_grading_reported:   str
    irAE_organs_reported:    str

    reports_OS:  str
    reports_PFS: str

    study_design:                Optional[str]              = None
    sample_size:                 Optional[int]              = None
    immunotherapy_agent:         Optional[str]              = None
    followup_median_months:      Optional[Union[int,float]] = None
    irAE_incidence_any_grade:    Optional[Union[int,float]] = None
    irAE_incidence_grade3plus:   Optional[Union[int,float]] = None
    biomarker_reported:          Optional[str]              = None
    quality_score:               Optional[int]              = None

    eligibility_decision: str

    # Extended fields
    treatment_line:              Optional[str]                        = None
    combination_regimen:         Optional[str]                        = None
    prior_immunotherapy:         Optional[str]                        = None
    geographic_region:           Optional[str]                        = None
    specific_irAE_incidence:     Optional[Dict[str,Union[int,float]]] = None
    irAE_management_summary:     Optional[str]                        = None
    specific_biomarker_status:   Optional[str]                        = None
    study_type:                  Optional[str]                        = None
    ECOG_status:                 Optional[str]                        = None
    PROs_related_irAE:           Optional[str]                        = None

    @field_validator("*", mode="before")
    @classmethod
    def none_if_empty(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

# ───────────────────────── OpenRouter Client ────────────────────────────────

load_dotenv(ROOT / ".env")
API_KEY  = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
if not API_KEY:
    raise SystemExit("❌  OPENAI_API_KEY not found in .env — copy .env.example to .env and fill in your key.")

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# ───────────────────────── Load DOCX Records ────────────────────────────────

def load_records() -> List[Dict]:
    pattern = re.compile(r"(RECORD\s+\d+)", re.I)
    records = []
    for fname in DOC_FILES:
        path = ROOT / fname
        if not path.exists():
            print(f"⚠️  File not found: {fname} — skipping.")
            continue
        doc  = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        parts = pattern.split(text)[1:]   # [header, body, header, body ...]
        for hdr, body in zip(parts[0::2], parts[1::2]):
            records.append({"record_id": hdr.strip(), "raw": body.strip()})
    print(f"✔️  Loaded {len(records)} records total.")
    return records

# ───────────────────────── Checkpoint ───────────────────────────────────────

def load_checkpoint() -> Set[str]:
    if CKPT_PATH.exists():
        try:
            return set(json.loads(CKPT_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return set()

def save_checkpoint(done_ids: Set[str]):
    CKPT_PATH.write_text(
        json.dumps(sorted(done_ids), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ───────────────────────── Prompt ───────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an evidence-extraction agent for immune-related adverse events (irAE).\n"
    "RETURN **ONLY** a valid JSON *array* that exactly matches the schema below.\n"
    "No markdown fences, no comments, no additional keys, no prose.\n"
    "If a value is unknown, use null (for optional fields) or \"no\" (for yes/no fields).\n\n"
    "Schema:\n"
    + json.dumps(ExtractionSchema.model_json_schema(), indent=2)
    + "\n\n"
    "IMPORTANT: Each object MUST include all keys in this exact order:\n"
    "  1. record_id\n  2. name\n  3. DOI\n  4. year\n  5. journal\n"
    "  6. cancer_type\n  7. stage\n  8. comparison_arms\n"
    "  9. discusses_irAE\n 10. irAE_subgroup_analysis\n"
    " 11. irAE_grading_reported\n 12. irAE_organs_reported\n"
    " 13. reports_OS\n 14. reports_PFS\n 15. study_design\n"
    " 16. sample_size\n 17. immunotherapy_agent\n 18. followup_median_months\n"
    " 19. irAE_incidence_any_grade\n 20. irAE_incidence_grade3plus\n"
    " 21. biomarker_reported\n 22. quality_score\n 23. eligibility_decision\n"
    " 24. treatment_line\n 25. combination_regimen\n 26. prior_immunotherapy\n"
    " 27. geographic_region\n 28. specific_irAE_incidence\n"
    " 29. irAE_management_summary\n 30. specific_biomarker_status\n"
    " 31. study_type\n 32. ECOG_status\n 33. PROs_related_irAE\n"
    "\nBegin:\n"
)

# ───────────────────────── JSON Sanitizer ───────────────────────────────────

def sanitize(raw: str) -> str:
    """Strip markdown code fences and common LLM formatting noise."""
    txt = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` fences (handles multi-line)
    txt = re.sub(r"```(?:json)?\s*", "", txt, flags=re.I)
    txt = txt.replace("```", "")
    # Fall back: replace single-quotes only if no double-quotes present
    if "'" in txt and '"' not in txt:
        txt = txt.replace("'", '"')
    # Remove trailing comma before closing bracket
    txt = re.sub(r",\s*]", "]", txt)
    # Trim to first '[' … last ']'
    start = txt.find("[")
    end   = txt.rfind("]")
    if start != -1 and end != -1:
        txt = txt[start: end + 1]
    return txt

# ───────────────────────── LLM Call ─────────────────────────────────────────

def call_llm(batch_text: str, batch_idx: int, retry: int = 0) -> List[dict]:
    try:
        resp   = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": batch_text},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        finish = resp.choices[0].finish_reason
        reply  = resp.choices[0].message.content or ""

        if finish == "length" or not reply.strip():
            print("   ⚠️  Empty reply or truncated — retrying/shrinking batch.")
            if retry < RETRY_LIMIT:
                time.sleep(1.5 * (retry + 1))
                return call_llm(batch_text, batch_idx, retry + 1)
            return []

        try:
            data = json.loads(sanitize(reply))
        except Exception as e:
            print(f"   ⚠️  JSON parse error: {e}")
            snippet = reply[:400].replace("\n", " ") if reply else "<<empty>>"
            print(f"   ↳  First 400 chars: {snippet}…")
            if retry < RETRY_LIMIT:
                time.sleep(1.5 * (retry + 1))
                return call_llm(batch_text, batch_idx, retry + 1)
            fail_path = FAILED_DIR / f"failed_batch_{batch_idx}.txt"
            fail_path.write_text(reply, encoding="utf-8")
            print(f"   ❌  Saved raw reply to {fail_path}")
            return []

        # Validate via Pydantic and serialise
        validated = [ExtractionSchema(**item).model_dump() for item in data]
        usage = resp.usage.total_tokens if resp.usage else 0
        cost  = usage / 1_000_000 * PRICE_PER_M
        print(f"   ↳  tokens={usage}, est cost=${cost:.4f}")
        return validated

    except Exception as e:
        print(f"   ⚠️  API / network error: {e}")
        if retry < RETRY_LIMIT:
            time.sleep(1.5 * (retry + 1))
            return call_llm(batch_text, batch_idx, retry + 1)
        return []

# ───────────────────────── Main ─────────────────────────────────────────────

def main():
    records = load_records()

    if START_INDEX is not None:
        if not (0 <= START_INDEX < len(records)):
            raise ValueError(f"START_INDEX={START_INDEX} out of range (0–{len(records)-1})")
        pending  = records[START_INDEX:]
        done_ids: Set[str] = set()
        print(f"🎯  START_INDEX={START_INDEX}: processing records {START_INDEX+1}–{len(records)} ({len(pending)} total).")
    else:
        done_ids = load_checkpoint()
        pending  = [r for r in records if r["record_id"] not in done_ids]
        print(f"🎯  Checkpoint resume: {len(done_ids)} done, {len(pending)} remaining.")

    if not pending:
        print("🎉  All records already processed.")
        return

    # Write CSV header if file doesn't exist yet
    if not CSV_PATH.exists():
        columns = list(ExtractionSchema.model_fields.keys())
        pd.DataFrame(columns=columns).to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    i          = 0
    batch_size = BATCH_SIZE_INIT

    while i < len(pending):
        # Dynamically shrink batch to stay under token limit
        for size in [batch_size, 3, 2, 1]:
            subset     = pending[i: i + size]
            batch_text = "\n\n".join(f"{r['record_id']}\n{r['raw']}" for r in subset)
            if num_tokens(batch_text) <= MAX_BATCH_TOK or size == 1:
                batch_size = size
                break

        print(f"\n🚀  Batch starting at index {i}  (size={batch_size})")
        results = call_llm(batch_text, batch_idx=i)

        if not results:
            print(f"🔻  Batch failed — retrying each record individually…")
            for rec in subset:
                single_text = f"{rec['record_id']}\n{rec['raw']}"
                single_res  = call_llm(single_text, batch_idx=i)
                if single_res:
                    pd.DataFrame(single_res).to_csv(
                        CSV_PATH, mode="a", header=False, index=False, encoding="utf-8-sig"
                    )
                    done_ids.add(rec["record_id"])
                    print(f"   ✔️  {rec['record_id']} recovered individually.")
                else:
                    print(f"   ❌  {rec['record_id']} still failed — saving raw text.")
                    (FAILED_DIR / f"{rec['record_id']}.txt").write_text(rec["raw"], encoding="utf-8")
            save_checkpoint(done_ids)
            i += batch_size
            time.sleep(SLEEP_BETWEEN)
            continue

        pd.DataFrame(results).to_csv(CSV_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
        for r in subset:
            done_ids.add(r["record_id"])
        save_checkpoint(done_ids)
        print(f"   ✔️  Wrote {len(results)} rows  (total done: {len(done_ids)}/{len(records)})")

        i += batch_size
        time.sleep(SLEEP_BETWEEN)

    df = pd.read_csv(CSV_PATH)
    print("\n========== Done ==========")
    print(df["eligibility_decision"].value_counts(dropna=False))
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    main()
