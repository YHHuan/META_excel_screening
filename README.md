# META_excel_screening

An automated pipeline that uses large language models (LLMs) via the [OpenRouter](https://openrouter.ai) API to screen and extract structured data from immune-related adverse events (irAE) research papers stored as `.docx` records.

---

## Overview

Checkpoint-based batch processor that:
1. Reads DOCX files containing literature records
2. Sends each record to a configurable LLM (default: Gemini 2.5 Flash)
3. Extracts 33 structured fields per paper (cancer type, irAE incidence, study design, eligibility, etc.)
4. Appends results to a CSV file with automatic retry and fallback logic

---

## Project Structure

```
META_excel_screening/
├── irAE_1st_round.py          # Main extraction script
├── test_api.py                # API connectivity test
├── requirements.txt
├── .env.example               # Template — copy to .env and add your key
├── .gitignore
├── data/
│   ├── records.docx           # Input records (batch 1)
│   └── records-2.docx         # Input records (batch 2)
├── results/
│   ├── irAE_extraction_gemini.csv          # Gemini extraction output
│   ├── irAE_extraction_deepseek.csv        # DeepSeek extraction output
│   └── irAE_with_subgroup_analysis.csv     # Subgroup analysis output
└── failed_batches/            # Raw LLM replies that failed JSON parsing
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your real [OpenRouter API key](https://openrouter.ai/keys):

```
OPENAI_API_KEY=sk-or-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

### 3. Test the connection

```bash
python test_api.py
```

---

## Usage

### Run extraction

```bash
python irAE_1st_round.py
```

The script uses a checkpoint file (`.checkpoint.json`) so it can be interrupted and resumed safely.

### Key configuration (top of `irAE_1st_round.py`)

| Variable | Default | Description |
|---|---|---|
| `START_INDEX` | `None` | Force-start from record N (0-based). `None` = resume from checkpoint. |
| `MODEL_ID` | `google/gemini-2.5-flash-preview-05-20` | OpenRouter model ID |
| `MAX_BATCH_TOK` | `12000` | Max tokens per batch sent to the LLM |
| `RETRY_LIMIT` | `2` | Max retries per failed batch |
| `PRICE_PER_M` | `0.0` | Cost per 1M tokens (set for paid models) |

---

## Extracted Fields

| # | Field | Description |
|---|---|---|
| 1 | `record_id` | Record identifier from DOCX |
| 2 | `name` | First author |
| 3 | `DOI` | DOI or trial ID (e.g. NCT…) |
| 4 | `year` | Publication year |
| 5 | `journal` | Journal name |
| 6 | `cancer_type` | Cancer type |
| 7 | `stage` | Disease stage |
| 8 | `comparison_arms` | Treatment arms compared |
| 9 | `discusses_irAE` | yes/no |
| 10 | `irAE_subgroup_analysis` | yes/no |
| 11 | `irAE_grading_reported` | yes/no |
| 12 | `irAE_organs_reported` | yes/no |
| 13 | `reports_OS` | yes/no — reports overall survival |
| 14 | `reports_PFS` | yes/no — reports progression-free survival |
| 15 | `study_design` | RCT, cohort, etc. |
| 16 | `sample_size` | Number of patients |
| 17 | `immunotherapy_agent` | Agent(s) used |
| 18 | `followup_median_months` | Median follow-up (months) |
| 19 | `irAE_incidence_any_grade` | Any-grade irAE incidence (%) |
| 20 | `irAE_incidence_grade3plus` | Grade 3+ irAE incidence (%) |
| 21 | `biomarker_reported` | Biomarker(s) mentioned |
| 22 | `quality_score` | Study quality score |
| 23 | `eligibility_decision` | include/exclude/unclear |
| 24 | `treatment_line` | Neoadjuvant / Adjuvant / First-Line / etc. |
| 25 | `combination_regimen` | IO+Chemo, IO+Targeted, Dual ICIs, etc. |
| 26 | `prior_immunotherapy` | yes/no/not mentioned |
| 27 | `geographic_region` | Asia / North America / Europe / Global |
| 28 | `specific_irAE_incidence` | Per-organ incidence dict |
| 29 | `irAE_management_summary` | Steroid use, discontinuation rates, etc. |
| 30 | `specific_biomarker_status` | PD-L1 CPS, MSI, TMB status |
| 31 | `study_type` | RCT / Real-World Study |
| 32 | `ECOG_status` | Performance status (e.g. 0-1) |
| 33 | `PROs_related_irAE` | Patient-reported outcomes related to irAE |

---

## Results

Extraction results are stored in the `results/` folder:

- **`irAE_extraction_gemini.csv`** — Full extraction run using Gemini 2.5 Flash (~665 records)
- **`irAE_extraction_deepseek.csv`** — Extraction run using DeepSeek (~49 records)
- **`irAE_with_subgroup_analysis.csv`** — Records with irAE subgroup analysis flagged

---

## Error Handling

- Failed batches are automatically split and retried record-by-record
- Raw LLM replies that cannot be parsed are saved to `failed_batches/` for manual review
- Progress is checkpointed after every successful batch

---

## License

For academic / research use. See individual model licenses on [OpenRouter](https://openrouter.ai/models).
