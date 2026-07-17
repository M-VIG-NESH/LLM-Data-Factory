"""
Pipeline stages 3-14: data cleaning, text normalization, content filtering,
schema design, transformation, tokenization, balancing, annotation,
validation, deduplication, versioning, and export.
"""

import re
import os
import json
import csv
import hashlib
import unicodedata
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 3 – Data Cleaning (Structural)
# ---------------------------------------------------------------------------

class DataCleaner:
    """Fix technical inconsistencies in tabular / textual data."""

    @staticmethod
    def handle_missing_values(records: List[Dict], strategy: str = "mark_null") -> Dict:
        """
        Handle missing values across a list of record dicts.

        strategy: 'remove' | 'mark_null' | 'impute_empty'
        """
        total = len(records)
        cleaned = []
        removed = 0
        fixed = 0

        for rec in records:
            has_missing = any(v is None or v == "" for v in rec.values())
            if has_missing:
                if strategy == "remove":
                    removed += 1
                    continue
                elif strategy == "mark_null":
                    rec = {k: ("<NULL>" if (v is None or v == "") else v) for k, v in rec.items()}
                    fixed += 1
                elif strategy == "impute_empty":
                    rec = {k: ("N/A" if (v is None or v == "") else v) for k, v in rec.items()}
                    fixed += 1
            cleaned.append(rec)

        return {
            "records": cleaned,
            "stats": {
                "total": total,
                "removed": removed,
                "fixed": fixed,
                "remaining": len(cleaned),
            }
        }

    @staticmethod
    def remove_exact_duplicates(records: List[Dict]) -> Dict:
        seen = set()
        unique = []
        dupes = 0
        for rec in records:
            key = hashlib.md5(json.dumps(rec, sort_keys=True).encode()).hexdigest()
            if key in seen:
                dupes += 1
            else:
                seen.add(key)
                unique.append(rec)
        return {"records": unique, "stats": {"removed_duplicates": dupes, "remaining": len(unique)}}

    @staticmethod
    def normalize_dates(records: List[Dict], date_fields: List[str]) -> Dict:
        """Attempt ISO-8601 normalization for known date columns."""
        import re as _re
        fixed = 0
        for rec in records:
            for field in date_fields:
                val = rec.get(field, "")
                if not val:
                    continue
                # Try common patterns
                for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y%m%d"):
                    try:
                        dt = datetime.strptime(str(val), fmt)
                        rec[field] = dt.strftime("%Y-%m-%d")
                        fixed += 1
                        break
                    except Exception:
                        pass
        return {"records": records, "stats": {"date_fields_normalized": fixed}}

    @staticmethod
    def fix_encoding(records: List[Dict]) -> Dict:
        fixed = 0
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, str):
                    cleaned = v.encode("utf-8", errors="ignore").decode("utf-8")
                    cleaned = cleaned.replace("\x00", "").replace("\ufffd", "")
                    if cleaned != v:
                        rec[k] = cleaned
                        fixed += 1
        return {"records": records, "stats": {"encoding_fixes": fixed}}

    @staticmethod
    def run_all(records: List[Dict], missing_strategy: str = "mark_null",
                date_fields: Optional[List[str]] = None) -> Dict:
        date_fields = date_fields or []
        steps = {}

        r1 = DataCleaner.remove_exact_duplicates(records)
        steps["duplicate_removal"] = r1["stats"]
        records = r1["records"]

        r2 = DataCleaner.handle_missing_values(records, missing_strategy)
        steps["missing_values"] = r2["stats"]
        records = r2["records"]

        if date_fields:
            r3 = DataCleaner.normalize_dates(records, date_fields)
            steps["date_normalization"] = r3["stats"]
            records = r3["records"]

        r4 = DataCleaner.fix_encoding(records)
        steps["encoding_fix"] = r4["stats"]
        records = r4["records"]

        return {"records": records, "steps": steps}


# ---------------------------------------------------------------------------
# Stage 4 – Text Normalization (Linguistic Cleaning)
# ---------------------------------------------------------------------------

class TextNormalizer:
    HTML_PATTERN = re.compile(r"<[^>]+>")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

    @staticmethod
    def remove_html(text: str) -> str:
        return TextNormalizer.HTML_PATTERN.sub(" ", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        return TextNormalizer.WHITESPACE_PATTERN.sub(" ", text).strip()

    @staticmethod
    def remove_urls(text: str) -> str:
        return TextNormalizer.URL_PATTERN.sub("", text)

    @staticmethod
    def remove_boilerplate(text: str) -> str:
        patterns = [
            r"(?i)(all rights reserved|copyright \d{4}|terms of service|privacy policy|"
            r"disclaimer|footer|header|page \d+ of \d+)"
        ]
        for p in patterns:
            text = re.sub(p, "", text)
        return text

    @staticmethod
    def standardize_punctuation(text: str) -> str:
        # Normalize quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        # Normalize dashes
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        return text

    @staticmethod
    def normalize_text(
        text: str,
        remove_html: bool = True,
        remove_urls: bool = False,
        lowercase: bool = False,
        remove_boilerplate_: bool = True,
    ) -> Dict:
        original_len = len(text)
        steps_applied = []

        if remove_html:
            text = TextNormalizer.remove_html(text)
            steps_applied.append("remove_html")
        if remove_urls:
            text = TextNormalizer.remove_urls(text)
            steps_applied.append("remove_urls")
        if remove_boilerplate_:
            text = TextNormalizer.remove_boilerplate(text)
            steps_applied.append("remove_boilerplate")

        text = TextNormalizer.standardize_punctuation(text)
        text = TextNormalizer.normalize_whitespace(text)
        steps_applied.extend(["standardize_punctuation", "normalize_whitespace"])

        if lowercase:
            text = text.lower()
            steps_applied.append("lowercase")

        return {
            "text": text,
            "stats": {
                "original_length": original_len,
                "cleaned_length": len(text),
                "reduction_pct": round((1 - len(text) / max(original_len, 1)) * 100, 1),
                "steps_applied": steps_applied,
            }
        }

    @staticmethod
    def normalize_records(records: List[Dict], text_fields: List[str], **kwargs) -> Dict:
        all_stats = []
        for rec in records:
            for field in text_fields:
                if field in rec and isinstance(rec[field], str):
                    result = TextNormalizer.normalize_text(rec[field], **kwargs)
                    rec[field] = result["text"]
                    all_stats.append(result["stats"])
        avg_reduction = (
            sum(s["reduction_pct"] for s in all_stats) / len(all_stats) if all_stats else 0
        )
        return {
            "records": records,
            "stats": {"fields_processed": len(all_stats), "avg_reduction_pct": avg_reduction},
        }


# ---------------------------------------------------------------------------
# Stage 5 – Content Filtering & Quality Control
# ---------------------------------------------------------------------------

TOXIC_KEYWORDS = [
    "hate", "kill", "racist", "slur", "obscene", "explicit",
    "violence", "abuse", "harassment",
]


class ContentFilter:
    @staticmethod
    def is_too_short(text: str, min_words: int = 10) -> bool:
        return len(text.split()) < min_words

    @staticmethod
    def is_toxic(text: str) -> bool:
        t = text.lower()
        return any(kw in t for kw in TOXIC_KEYWORDS)

    @staticmethod
    def is_spam(text: str) -> bool:
        """Heuristic: many uppercase letters, many repeated characters."""
        if len(text) == 0:
            return False
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
        words = text.split()
        if not words:
            return False
        unique_ratio = len(set(words)) / len(words)
        return upper_ratio > 0.5 or unique_ratio < 0.2

    @staticmethod
    def filter_records(
        records: List[Dict],
        text_fields: List[str],
        min_words: int = 10,
        remove_toxic: bool = True,
        remove_spam: bool = True,
    ) -> Dict:
        kept = []
        filtered_log = []
        for rec in records:
            reasons = []
            text_sample = " ".join(
                str(rec.get(f, "")) for f in text_fields if rec.get(f)
            )
            if ContentFilter.is_too_short(text_sample, min_words):
                reasons.append("too_short")
            if remove_toxic and ContentFilter.is_toxic(text_sample):
                reasons.append("toxic")
            if remove_spam and ContentFilter.is_spam(text_sample):
                reasons.append("spam")

            if reasons:
                filtered_log.append({"record_preview": text_sample[:80], "reasons": reasons})
            else:
                kept.append(rec)

        stats = Counter()
        for entry in filtered_log:
            for r in entry["reasons"]:
                stats[r] += 1

        return {
            "records": kept,
            "filtered": filtered_log,
            "stats": {
                "original": len(records),
                "kept": len(kept),
                "removed": len(filtered_log),
                "by_reason": dict(stats),
            },
        }


# ---------------------------------------------------------------------------
# Stage 6 – Schema Design (Restructuring for LLM Use)
# ---------------------------------------------------------------------------

class SchemaDesigner:
    FORMATS = ["instruction_tuning", "chat_format", "retrieval_doc", "qa_pairs"]

    @staticmethod
    def to_instruction_tuning(records: List[Dict],
                               instruction_field: str,
                               input_field: str,
                               output_field: str) -> List[Dict]:
        result = []
        for rec in records:
            result.append({
                "instruction": str(rec.get(instruction_field, "")),
                "input": str(rec.get(input_field, "")),
                "output": str(rec.get(output_field, "")),
            })
        return result

    @staticmethod
    def to_chat_format(records: List[Dict],
                        user_field: str,
                        assistant_field: str,
                        system_prompt: str = "You are a helpful assistant.") -> List[Dict]:
        result = []
        for rec in records:
            result.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": str(rec.get(user_field, ""))},
                    {"role": "assistant", "content": str(rec.get(assistant_field, ""))},
                ]
            })
        return result

    @staticmethod
    def to_retrieval_doc(records: List[Dict],
                          text_field: str,
                          metadata_fields: Optional[List[str]] = None) -> List[Dict]:
        result = []
        for rec in records:
            meta = {k: rec[k] for k in (metadata_fields or []) if k in rec}
            result.append({
                "text": str(rec.get(text_field, "")),
                "metadata": meta,
            })
        return result

    @staticmethod
    def to_qa_pairs(records: List[Dict],
                     question_field: str,
                     answer_field: str,
                     context_field: Optional[str] = None) -> List[Dict]:
        result = []
        for rec in records:
            entry = {
                "question": str(rec.get(question_field, "")),
                "answer": str(rec.get(answer_field, "")),
            }
            if context_field and context_field in rec:
                entry["context"] = str(rec[context_field])
            result.append(entry)
        return result

    @staticmethod
    def convert(records: List[Dict], target_format: str, field_map: Dict) -> Dict:
        """
        field_map keys depend on target_format:
          instruction_tuning: instruction, input, output
          chat_format: user, assistant, (system_prompt optional)
          retrieval_doc: text, (metadata_fields list optional)
          qa_pairs: question, answer, (context optional)
        """
        if target_format == "instruction_tuning":
            out = SchemaDesigner.to_instruction_tuning(
                records,
                field_map.get("instruction", "question"),
                field_map.get("input", "context"),
                field_map.get("output", "answer"),
            )
        elif target_format == "chat_format":
            out = SchemaDesigner.to_chat_format(
                records,
                field_map.get("user", "question"),
                field_map.get("assistant", "answer"),
                field_map.get("system_prompt", "You are a helpful assistant."),
            )
        elif target_format == "retrieval_doc":
            out = SchemaDesigner.to_retrieval_doc(
                records,
                field_map.get("text", "answer"),
                field_map.get("metadata_fields"),
            )
        elif target_format == "qa_pairs":
            out = SchemaDesigner.to_qa_pairs(
                records,
                field_map.get("question", "question"),
                field_map.get("answer", "answer"),
                field_map.get("context"),
            )
        else:
            out = records

        return {"records": out, "format": target_format, "count": len(out)}


# ---------------------------------------------------------------------------
# Stage 7 – Data Transformation / Feature Engineering
# ---------------------------------------------------------------------------

class DataTransformer:
    @staticmethod
    def combine_columns_to_text(records: List[Dict],
                                  columns: List[str],
                                  output_field: str = "combined_text",
                                  template: Optional[str] = None) -> List[Dict]:
        for rec in records:
            if template:
                try:
                    rec[output_field] = template.format(**rec)
                except KeyError:
                    rec[output_field] = " | ".join(str(rec.get(c, "")) for c in columns)
            else:
                rec[output_field] = " ".join(str(rec.get(c, "")) for c in columns)
        return records

    @staticmethod
    def row_to_narrative(record: Dict, excluded: Optional[List[str]] = None) -> str:
        excluded = excluded or []
        parts = []
        for k, v in record.items():
            if k not in excluded and v not in (None, ""):
                parts.append(f"{k.replace('_', ' ').title()} is {v}")
        return ". ".join(parts) + "."

    @staticmethod
    def add_metadata_tags(records: List[Dict],
                           tags: Dict[str, str]) -> List[Dict]:
        for rec in records:
            rec.update(tags)
        return records

    @staticmethod
    def run_transformations(records: List[Dict],
                             combine_cols: Optional[List[str]] = None,
                             template: Optional[str] = None,
                             to_narrative: bool = False,
                             tags: Optional[Dict] = None,
                             output_field: str = "transformed_text") -> Dict:
        ops = []
        if combine_cols:
            records = DataTransformer.combine_columns_to_text(
                records, combine_cols, output_field, template
            )
            ops.append(f"combined_columns: {combine_cols}")

        if to_narrative:
            for rec in records:
                rec["narrative"] = DataTransformer.row_to_narrative(rec)
            ops.append("row_to_narrative")

        if tags:
            records = DataTransformer.add_metadata_tags(records, tags)
            ops.append(f"added_tags: {list(tags.keys())}")

        return {"records": records, "ops_applied": ops, "count": len(records)}


# ---------------------------------------------------------------------------
# Stage 8 – Tokenization & Length Control
# ---------------------------------------------------------------------------

class TokenizationController:
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.enc = None

    def count_tokens(self, text: str) -> int:
        if self.enc:
            return len(self.enc.encode(text))
        return len(text.split())

    def chunk_text(self, text: str) -> List[Dict]:
        tokens = self.enc.encode(text) if self.enc else text.split()
        chunks = []
        start = 0
        idx = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = (
                self.enc.decode(chunk_tokens) if self.enc
                else " ".join(chunk_tokens)
            )
            chunks.append({
                "chunk_id": idx,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end,
            })
            start += self.chunk_size - self.overlap
            idx += 1
        return chunks

    def process_records(self, records: List[Dict], text_field: str) -> Dict:
        all_chunks = []
        oversized = 0
        for rec in records:
            text = str(rec.get(text_field, ""))
            token_count = self.count_tokens(text)
            if token_count > self.chunk_size:
                oversized += 1
                for ch in self.chunk_text(text):
                    new_rec = dict(rec)
                    new_rec[text_field] = ch["text"]
                    new_rec["_token_count"] = ch["token_count"]
                    new_rec["_chunk_id"] = ch["chunk_id"]
                    all_chunks.append(new_rec)
            else:
                rec["_token_count"] = token_count
                rec["_chunk_id"] = 0
                all_chunks.append(rec)

        return {
            "records": all_chunks,
            "stats": {
                "original_records": len(records),
                "output_records": len(all_chunks),
                "oversized_chunked": oversized,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
            }
        }


# ---------------------------------------------------------------------------
# Stage 9 – Dataset Balancing
# ---------------------------------------------------------------------------

class DatasetBalancer:
    @staticmethod
    def get_distribution(records: List[Dict], field: str) -> Dict[str, int]:
        counts: Counter = Counter()
        for rec in records:
            val = str(rec.get(field, "UNKNOWN"))
            counts[val] += 1
        return dict(counts)

    @staticmethod
    def downsample(records: List[Dict], field: str, target: int) -> List[Dict]:
        from random import sample
        groups: Dict[str, list] = {}
        for rec in records:
            key = str(rec.get(field, "UNKNOWN"))
            groups.setdefault(key, []).append(rec)
        result = []
        for group in groups.values():
            result.extend(sample(group, min(target, len(group))))
        return result

    @staticmethod
    def oversample(records: List[Dict], field: str, target: int) -> List[Dict]:
        from random import choices
        groups: Dict[str, list] = {}
        for rec in records:
            key = str(rec.get(field, "UNKNOWN"))
            groups.setdefault(key, []).append(rec)
        result = []
        for group in groups.values():
            if len(group) < target:
                result.extend(choices(group, k=target))
            else:
                result.extend(group)
        return result

    @staticmethod
    def run_balancing(records: List[Dict], field: str,
                       strategy: str = "none") -> Dict:
        original_dist = DatasetBalancer.get_distribution(records, field)
        if not original_dist:
            return {"records": records, "stats": {}}

        if strategy == "downsample":
            target = min(original_dist.values())
            records = DatasetBalancer.downsample(records, field, target)
        elif strategy == "oversample":
            target = max(original_dist.values())
            records = DatasetBalancer.oversample(records, field, target)
        # "none" or "weighted_sampling" — just report distribution

        new_dist = DatasetBalancer.get_distribution(records, field)
        return {
            "records": records,
            "stats": {
                "strategy": strategy,
                "original_distribution": original_dist,
                "new_distribution": new_dist,
                "original_count": sum(original_dist.values()),
                "new_count": len(records),
            }
        }


# ---------------------------------------------------------------------------
# Stage 10 – Annotation / Labeling
# ---------------------------------------------------------------------------

class Annotator:
    @staticmethod
    def auto_label_by_keyword(records: List[Dict],
                               text_field: str,
                               label_map: Dict[str, List[str]],
                               label_output_field: str = "label") -> Dict:
        labeled = 0
        for rec in records:
            text = str(rec.get(text_field, "")).lower()
            assigned = "unlabeled"
            for label, keywords in label_map.items():
                if any(kw.lower() in text for kw in keywords):
                    assigned = label
                    labeled += 1
                    break
            rec[label_output_field] = assigned

        return {
            "records": records,
            "stats": {
                "total": len(records),
                "labeled": labeled,
                "unlabeled": len(records) - labeled,
                "label_distribution": dict(Counter(r[label_output_field] for r in records)),
            }
        }

    @staticmethod
    def validate_labels(records: List[Dict],
                         label_field: str,
                         valid_labels: List[str]) -> Dict:
        invalid = []
        for rec in records:
            if rec.get(label_field) not in valid_labels:
                invalid.append(rec.get(label_field))

        return {
            "is_valid": len(invalid) == 0,
            "invalid_labels": invalid,
            "stats": {
                "total": len(records),
                "invalid_count": len(invalid),
                "valid_labels": valid_labels,
            }
        }


# ---------------------------------------------------------------------------
# Stage 11 – Validation & Testing
# ---------------------------------------------------------------------------

class DataValidator:
    @staticmethod
    def validate_schema(records: List[Dict], required_fields: List[str]) -> Dict:
        errors = []
        for idx, rec in enumerate(records):
            missing = [f for f in required_fields if f not in rec or rec[f] in (None, "")]
            if missing:
                errors.append({"record_index": idx, "missing_fields": missing})
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "stats": {"total": len(records), "invalid": len(errors), "valid": len(records) - len(errors)},
        }

    @staticmethod
    def validate_json_format(records: List[Dict]) -> Dict:
        errors = []
        for idx, rec in enumerate(records):
            try:
                json.dumps(rec)
            except (TypeError, ValueError) as e:
                errors.append({"record_index": idx, "error": str(e)})
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "stats": {"total": len(records), "invalid_json": len(errors)},
        }

    @staticmethod
    def train_val_test_split(records: List[Dict],
                              train_ratio: float = 0.8,
                              val_ratio: float = 0.1) -> Dict:
        from random import shuffle, seed
        seed(42)
        data = list(records)
        shuffle(data)
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        return {
            "train": data[:train_end],
            "validation": data[train_end:val_end],
            "test": data[val_end:],
            "stats": {
                "total": n,
                "train": train_end,
                "validation": val_end - train_end,
                "test": n - val_end,
            }
        }

    @staticmethod
    def run_full_validation(records: List[Dict],
                             required_fields: List[str],
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1) -> Dict:
        schema_result = DataValidator.validate_schema(records, required_fields)
        json_result = DataValidator.validate_json_format(records)
        split_result = DataValidator.train_val_test_split(records, train_ratio, val_ratio)

        return {
            "schema_validation": schema_result,
            "json_validation": json_result,
            "split": split_result,
            "overall_valid": schema_result["is_valid"] and json_result["is_valid"],
        }


# ---------------------------------------------------------------------------
# Stage 12 – Deduplication (Final Pass)
# ---------------------------------------------------------------------------

class FinalDeduplicator:
    @staticmethod
    def exact_hash_dedup(records: List[Dict], fields: List[str]) -> Dict:
        seen = set()
        unique = []
        dupes = 0
        for rec in records:
            key = hashlib.md5(
                "|".join(str(rec.get(f, "")) for f in fields).encode()
            ).hexdigest()
            if key in seen:
                dupes += 1
            else:
                seen.add(key)
                unique.append(rec)
        return {
            "records": unique,
            "stats": {"original": len(records) + dupes, "removed": dupes, "remaining": len(unique)},
        }

    @staticmethod
    def near_duplicate_dedup(records: List[Dict], text_field: str,
                              similarity_threshold: float = 0.9) -> Dict:
        """
        Simple shingling-based near-duplicate detection.
        """
        def shingle(text: str, n: int = 5) -> set:
            words = text.lower().split()
            return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

        unique = []
        shingle_sets = []
        dupes = 0

        for rec in records:
            text = str(rec.get(text_field, ""))
            sh = shingle(text)
            is_dup = False
            for existing_sh in shingle_sets:
                if not existing_sh or not sh:
                    continue
                jaccard = len(sh & existing_sh) / len(sh | existing_sh)
                if jaccard >= similarity_threshold:
                    is_dup = True
                    dupes += 1
                    break
            if not is_dup:
                unique.append(rec)
                shingle_sets.append(sh)

        return {
            "records": unique,
            "stats": {
                "original": len(records) + dupes,
                "removed": dupes,
                "remaining": len(unique),
                "threshold": similarity_threshold,
            }
        }


# ---------------------------------------------------------------------------
# Stage 13 – Versioning & Documentation
# ---------------------------------------------------------------------------

class VersionManager:
    VERSION_DIR = "./data/versions"

    @staticmethod
    def save_version(records: List[Dict],
                      metadata: Dict,
                      version_tag: Optional[str] = None) -> Dict:
        os.makedirs(VersionManager.VERSION_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = version_tag or f"v_{ts}"
        filename = os.path.join(VersionManager.VERSION_DIR, f"dataset_{tag}.jsonl")

        with open(filename, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        meta_file = os.path.join(VersionManager.VERSION_DIR, f"metadata_{tag}.json")
        meta = {
            "version_tag": tag,
            "timestamp": ts,
            "record_count": len(records),
            "file": filename,
            **metadata,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return {"version_tag": tag, "file": filename, "metadata_file": meta_file, "count": len(records)}

    @staticmethod
    def list_versions() -> List[Dict]:
        os.makedirs(VersionManager.VERSION_DIR, exist_ok=True)
        versions = []
        for fname in os.listdir(VersionManager.VERSION_DIR):
            if fname.startswith("metadata_") and fname.endswith(".json"):
                fpath = os.path.join(VersionManager.VERSION_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        versions.append(json.load(f))
                except Exception:
                    pass
        return sorted(versions, key=lambda x: x.get("timestamp", ""), reverse=True)


# ---------------------------------------------------------------------------
# Stage 14 – Export to LLM-Ready Format
# ---------------------------------------------------------------------------

class FinalExporter:
    """
    Export the final working_data to disk.

    Key guarantee
    -------------
    Every record is cleaned to ONLY the allowed keys before writing:
      QA pair       -> {"question": ..., "answer": ...}
      Summarization -> {"question": ..., "context": ..., "answer": ...}

    All other fields (transformed_text, narrative, _chunk_id, _token_count,
    label, summary, complexity_score, is_valid, etc.) are silently dropped.
    """

    @staticmethod
    def _clean(records: List[Dict]) -> List[Dict]:
        """
        Strip every record to the strict allowed key set.
        Records missing both question AND answer are dropped entirely.
        Detection: if context is non-empty -> Summarization; else -> QA.
        """
        cleaned = []
        for rec in records:
            q   = str(rec.get("question", "") or "").strip()
            a   = str(rec.get("answer",   "") or "").strip()
            ctx = str(rec.get("context",  "") or "").strip()
            if not q or not a:
                continue
            if ctx:
                cleaned.append({"question": q, "context": ctx, "answer": a})
            else:
                cleaned.append({"question": q, "answer": a})
        return cleaned

    @staticmethod
    def to_jsonl(records: List[Dict], output_path: str) -> str:
        clean = FinalExporter._clean(records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in clean:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return output_path

    @staticmethod
    def to_csv(records: List[Dict], output_path: str) -> str:
        clean = FinalExporter._clean(records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not clean:
            return output_path
        df = pd.DataFrame(clean)
        df.to_csv(output_path, index=False, encoding="utf-8")
        return output_path

    @staticmethod
    def to_json(records: List[Dict], output_path: str) -> str:
        clean = FinalExporter._clean(records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        return output_path

    @staticmethod
    def to_parquet(records: List[Dict], output_path: str) -> str:
        clean = FinalExporter._clean(records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(clean)
        df.to_parquet(output_path, index=False)
        return output_path

    @staticmethod
    def export(records: List[Dict], fmt: str, base_name: str = "final_export") -> Dict:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./data/exports/{base_name}_{ts}.{fmt}"
        os.makedirs("./data/exports", exist_ok=True)

        if fmt == "jsonl":
            FinalExporter.to_jsonl(records, filename)
        elif fmt == "csv":
            FinalExporter.to_csv(records, filename)
        elif fmt == "json":
            FinalExporter.to_json(records, filename)
        elif fmt == "parquet":
            FinalExporter.to_parquet(records, filename)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        clean_count = len(FinalExporter._clean(records))
        size_kb = os.path.getsize(filename) / 1024
        return {
            "file": filename,
            "format": fmt,
            "record_count": clean_count,
            "size_kb": round(size_kb, 2),
        }
