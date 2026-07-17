"""
Generation pipeline — sequential, structured-CSV-aware, strict output keys.

Handles two primary input CSV formats:

  1. Medical_Data.csv  (15-column hospital quality data)
       → QA pairs   : [{"question": ..., "answer": ...}]

  2. Multi_Cuisine_Recipe_Dataset.csv  (name / area / category / ingredients / steps)
       → Summarization : [{"question": ..., "context": ..., "answer": ...}]

Both files are detected automatically from their column headers.

Strict output guarantee
-----------------------
  QA pairs      : ONLY keys  "question"  and  "answer"   — nothing else.
  Summarization : ONLY keys  "question", "context", "answer" — nothing else.
  Any extra keys the LLM returns are silently stripped.

Processing is fully sequential (one LLM call at a time) to stay within
Groq's API rate limits and avoid timeout / empty-response issues.
"""

import csv
import io
import json
import logging
import re as _re
from typing import Dict, List, Optional

from app.core.llm_client import LLMClient
from app.services.generation.prompts import (
    get_qa_prompt,
    get_summary_prompt,
    get_custom_prompt,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Column-schema signatures for the two known CSV types
# ═════════════════════════════════════════════════════════════════════════════

_MEDICAL_COLS = {
    "hospital_name", "city", "state", "ownership_type",
    "overall_star_rating", "performance_summary",
}

_RECIPE_COLS = {"name", "area", "category", "ingredients", "steps"}


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _extract_json_array(raw: str) -> list:
    """Robustly extract a JSON array from LLM output (handles markdown fences)."""
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    m = _re.search(r"\[.*\]", raw, _re.DOTALL)
    if m:
        raw = m.group(0)
    # Remove trailing commas that break json.loads
    raw = _re.sub(r",\s*([\}\]])", r"\1", raw)
    return json.loads(raw)


def _try_parse_csv(chunk: str):
    """
    Try to parse a text block as CSV.
    Returns (rows: list[dict], fieldnames: list[str]) or ([], []).
    """
    try:
        reader = csv.DictReader(io.StringIO(chunk.strip()))
        rows   = list(reader)
        if rows and reader.fieldnames:
            return rows, [f.strip() for f in reader.fieldnames]
    except Exception:
        pass
    return [], []


def _detect_schema(fieldnames: List[str]) -> str:
    """
    Returns one of: "medical" | "recipe" | "generic_structured" | "plain"
    based on which signature columns are present.
    """
    cols = {c.lower().strip() for c in fieldnames}

    if _MEDICAL_COLS.issubset(cols):
        return "medical"

    if _RECIPE_COLS.issubset(cols):
        return "recipe"

    flat_qa = {"question", "answer", "context"}
    if len(fieldnames) >= 4 or not cols.issubset(flat_qa):
        return "generic_structured"

    return "plain"


# ─────────────────────────────────────────────────────────────────────────────
# Prose builders — one per schema
# ─────────────────────────────────────────────────────────────────────────────

def _medical_row_to_prose(row: Dict[str, str]) -> str:
    """Build a readable English paragraph from a Medical_Data.csv row."""
    skip = {"unknown", "n/a", "none", "nan", ""}

    def v(col: str) -> str:
        return str(row.get(col, "")).strip()

    name       = v("hospital_name")
    city       = v("city")
    state      = v("state")
    ownership  = v("ownership_type")
    stars      = v("overall_star_rating")
    summary    = v("performance_summary")
    cost       = v("heart_attack_procedure_cost_usd")
    mortality  = v("mortality_national_comparison")
    safety     = v("safety_national_comparison")
    effective  = v("effectiveness_national_comparison")
    timeliness = v("timeliness_national_comparison")
    patient_ex = v("patient_experience_national_comparison")
    readmit    = v("readmission_rate")

    if not name:
        return ""

    parts = [
        f"{name} is a {ownership.lower()}-owned hospital located in {city}, {state}."
    ]

    if stars.lower() not in skip:
        parts.append(
            f"It has received an overall CMS star rating of {stars} out of 5."
        )

    comparisons = []
    if mortality.lower()  not in skip: comparisons.append(f"mortality ({mortality.lower()})")
    if safety.lower()     not in skip: comparisons.append(f"safety ({safety.lower()})")
    if effective.lower()  not in skip: comparisons.append(f"effectiveness ({effective.lower()})")
    if timeliness.lower() not in skip: comparisons.append(f"timeliness ({timeliness.lower()})")
    if patient_ex.lower() not in skip: comparisons.append(f"patient experience ({patient_ex.lower()})")
    if comparisons:
        parts.append(
            "Compared to the national average, the hospital performs at the following levels: "
            + ", ".join(comparisons) + "."
        )

    if cost.lower() not in skip:
        try:
            parts.append(
                f"The average cost for a heart attack procedure at this facility is "
                f"${int(float(cost)):,}."
            )
        except ValueError:
            parts.append(f"The heart attack procedure cost is {cost}.")

    if readmit.lower() not in skip:
        parts.append(
            f"The hospital's readmission rate is considered {readmit.lower()} "
            f"compared to national peers."
        )

    if summary and summary.lower() not in skip:
        parts.append(summary)

    return " ".join(parts)


def _recipe_row_to_prose(row: Dict[str, str]) -> str:
    """Build a readable English paragraph from a Multi_Cuisine_Recipe_Dataset.csv row."""
    skip = {"unknown", "n/a", "none", "nan", ""}

    name        = str(row.get("name",        "")).strip()
    area        = str(row.get("area",        "")).strip()
    category    = str(row.get("category",    "")).strip()
    ingredients = str(row.get("ingredients", "")).strip()
    steps       = str(row.get("steps",       "")).strip()

    if not name:
        return ""

    # Keep ingredient list concise to avoid oversized prompts
    ing_short = ingredients[:350].rstrip(",").strip()
    if len(ingredients) > 350:
        ing_short += "…"

    step_short = steps[:500].strip()
    if len(steps) > 500:
        step_short += "…"

    parts = [
        f"{name} is a traditional {area} dish classified under the {category} category."
    ]
    if ing_short and ing_short.lower() not in skip:
        parts.append(f"Key ingredients include: {ing_short}.")
    if step_short and step_short.lower() not in skip:
        parts.append(f"The preparation method begins as follows: {step_short}")

    return " ".join(parts)


def _generic_row_to_prose(row: Dict[str, str]) -> str:
    """Fallback prose builder for any other multi-column structured CSV."""
    skip  = {"unknown", "n/a", "none", "nan", ""}
    parts = []
    for key, val in row.items():
        val_str = str(val).strip()
        if not val_str or val_str.lower() in skip:
            continue
        label = key.replace("_", " ").title()
        key_lower = key.lower()
        if "cost_usd" in key_lower or "cost" in key_lower:
            try:
                val_str = f"${int(float(val_str)):,}"
            except (ValueError, TypeError):
                pass
        elif "minutes" in key_lower or "_time" in key_lower:
            try:
                val_str = f"{int(float(val_str))} minutes"
            except (ValueError, TypeError):
                pass
        parts.append(f"{label}: {val_str}")
    return ". ".join(parts) + "." if parts else ""


def _build_prose(row: Dict[str, str], schema: str) -> str:
    """Select the correct prose builder for the detected schema."""
    if schema == "medical":
        return _medical_row_to_prose(row)
    if schema == "recipe":
        return _recipe_row_to_prose(row)
    return _generic_row_to_prose(row)


# ─────────────────────────────────────────────────────────────────────────────
# Strict output key enforcement
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_qa_keys(record: dict) -> Optional[Dict[str, str]]:
    """Return ONLY {question, answer} — strip all other keys."""
    q = str(record.get("question", "")).strip()
    a = str(record.get("answer",   "")).strip()
    if not q or not a or len(a.split()) < 5:
        return None
    if not q.endswith("?"):
        q = q.rstrip(".!") + "?"
    return {"question": q, "answer": a}


def _enforce_summarization_keys(record: dict) -> Optional[Dict[str, str]]:
    """Return ONLY {question, context, answer} — strip all other keys."""
    q   = str(record.get("question", "")).strip()
    ctx = str(record.get("context",  "")).strip()
    a   = str(record.get("answer",   "")).strip()
    if not q or not ctx or not a:
        return None
    if len(ctx.split()) < 10 or len(a.split()) < 5:
        return None
    if not q.endswith("?"):
        q = q.rstrip(".!") + "?"
    return {"question": q, "context": ctx, "answer": a}


# ═════════════════════════════════════════════════════════════════════════════
# GenerationPipeline
# ═════════════════════════════════════════════════════════════════════════════

class GenerationPipeline:
    """
    Sequential LLM-based dataset generation pipeline.

    Public API
    ----------
    pipeline.generate_qa_pairs(context)    → List[{question, answer}]
    pipeline.generate_summary(context)     → List[{question, context, answer}] | None
    pipeline.generate_custom(context, ...) → str | None
    pipeline.batch_generate_qa(chunks)     → List[{question, answer}]
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    # ──────────────────────────────────────────────────────────────────────────
    # QA generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_qa_pairs(
        self,
        context:           str,
        num_pairs:         int = None,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate {question, answer} pairs from a text chunk or CSV block.
        Each CSV row is processed individually (one LLM call per row).
        Only {question, answer} keys are kept — all others are stripped.
        """
        num_pairs = num_pairs or settings.max_qa_pairs_per_chunk

        if not context or not context.strip():
            return []

        rows, fieldnames = _try_parse_csv(context)
        schema = _detect_schema(fieldnames) if fieldnames else "plain"

        if schema in ("medical", "recipe", "generic_structured") and rows:
            logger.info(
                f"Structured CSV detected ({schema}): "
                f"{len(rows)} rows × {len(fieldnames)} cols — processing sequentially"
            )
            all_pairs = []
            for idx, row in enumerate(rows):
                prose = _build_prose(row, schema)
                if not prose or len(prose.split()) < 10:
                    continue
                logger.info(f"  Row {idx + 1}/{len(rows)}: calling LLM…")
                pairs = self._call_llm_qa(prose, 1, few_shot_examples)
                all_pairs.extend(pairs)
            logger.info(f"CSV→QA complete: {len(all_pairs)} pairs from {len(rows)} rows.")
            return all_pairs

        # Plain text — single LLM call
        return self._call_llm_qa(context, num_pairs, few_shot_examples)

    # ──────────────────────────────────────────────────────────────────────────
    # Summarization generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_summary(
        self,
        context:           str,
        num_pairs:         int = 3,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> Optional[List[Dict[str, str]]]:
        """
        Generate {question, context, answer} records from a text chunk or CSV block.
        Each CSV row is processed individually (one LLM call per row).
        Only {question, context, answer} keys are kept — all others are stripped.
        """
        if not context or not context.strip():
            return None

        rows, fieldnames = _try_parse_csv(context)
        schema = _detect_schema(fieldnames) if fieldnames else "plain"

        if schema in ("medical", "recipe", "generic_structured") and rows:
            logger.info(
                f"Structured CSV detected ({schema}) for summarization: "
                f"{len(rows)} rows × {len(fieldnames)} cols — processing sequentially"
            )
            all_records = []
            for idx, row in enumerate(rows):
                prose = _build_prose(row, schema)
                if not prose or len(prose.split()) < 10:
                    continue
                logger.info(f"  Row {idx + 1}/{len(rows)}: calling LLM…")
                recs = self._call_llm_sum(prose, 1, few_shot_examples)
                if recs:
                    all_records.extend(recs)
            logger.info(
                f"CSV→Summarization complete: {len(all_records)} records from {len(rows)} rows."
            )
            return all_records or None

        # Plain text — single LLM call
        return self._call_llm_sum(context, num_pairs, few_shot_examples)

    # ──────────────────────────────────────────────────────────────────────────
    # Custom generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_custom(
        self,
        context:           str,
        instruction:       str,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> Optional[str]:
        if not context or not context.strip():
            return None
        try:
            prompt = get_custom_prompt(context, instruction, few_shot_examples)
            return self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"Custom generation failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Batch QA helper (Celery-compatible)
    # ──────────────────────────────────────────────────────────────────────────

    def batch_generate_qa(
        self,
        chunks:               List[str],
        num_pairs_per_chunk:  int = None,
        few_shot_examples:    Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Process a list of chunks sequentially.
        Returns [{question, answer}] — no extra keys.
        """
        results = []
        skipped = 0

        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)}")
            if not chunk or not chunk.strip():
                skipped += 1
                continue
            pairs = self.generate_qa_pairs(chunk, num_pairs_per_chunk, few_shot_examples)
            results.extend(pairs)

        logger.info(
            f"Batch complete: {len(results)} QA pairs from "
            f"{len(chunks) - skipped}/{len(chunks)} chunks."
        )
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Internal LLM callers (sequential, strict key enforcement)
    # ──────────────────────────────────────────────────────────────────────────

    def _call_llm_qa(
        self,
        prose:             str,
        num_pairs:         int,
        few_shot_examples: Optional[List[Dict]],
    ) -> List[Dict[str, str]]:
        """One sequential LLM call → QA pairs with only {question, answer}."""
        try:
            prompt   = get_qa_prompt(prose, num_pairs, few_shot_examples)
            response = self.llm_client.generate(prompt)
            records  = _extract_json_array(response)
            valid    = []
            for rec in (records if isinstance(records, list) else []):
                enforced = _enforce_qa_keys(rec)
                if enforced:
                    valid.append(enforced)
            return valid
        except json.JSONDecodeError as e:
            logger.error(f"QA JSON parse failed: {e}")
            return []
        except Exception as e:
            logger.error(f"QA generation failed: {e}")
            return []

    def _call_llm_sum(
        self,
        prose:             str,
        num_pairs:         int,
        few_shot_examples: Optional[List[Dict]],
    ) -> Optional[List[Dict[str, str]]]:
        """One sequential LLM call → Summarization records with only {question, context, answer}."""
        try:
            prompt   = get_summary_prompt(prose, num_pairs, few_shot_examples)
            response = self.llm_client.generate(prompt)
            records  = _extract_json_array(response)
            valid    = []
            for rec in (records if isinstance(records, list) else []):
                enforced = _enforce_summarization_keys(rec)
                if enforced:
                    valid.append(enforced)
            return valid or None
        except json.JSONDecodeError as e:
            logger.error(f"Summarization JSON parse failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Summarization generation failed: {e}")
            return None
