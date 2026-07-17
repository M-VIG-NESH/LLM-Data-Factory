import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import json
import sys

# Make sure we can import the pipeline stages even when launched from /ui
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.services.pipeline.stages import (
    DataCleaner,
    TextNormalizer,
    ContentFilter,
    SchemaDesigner,
    DataTransformer,
    TokenizationController,
    DatasetBalancer,
    Annotator,
    DataValidator,
    FinalDeduplicator,
    VersionManager,
    FinalExporter,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")

PIPELINE_STAGES = [
    ("📤", "Upload & Ingest",           1),
    ("🧹", "Data Cleaning",             2),
    ("🔤", "Text Normalization",         3),
    ("🛡️",  "Content Filtering & QC",    4),
    ("📐", "Schema Design",             5),
    ("🔧", "Data Transformation",        6),
    ("✂️",  "Tokenization & Length",     7),
    ("⚖️",  "Dataset Balancing",         8),
    ("🏷️",  "Annotation / Labeling",     9),
    ("🗑️",  "Deduplication (Final)",    10),
    ("✅", "Validation & Testing",      11),
    ("⚙️",  "Generate Dataset",         12),
    ("📊", "Deterministic Evaluation",  13),
    ("📚", "Versioning & Docs",         14),
    ("📦", "Export to LLM Format",      15),
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Data Factory",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 60%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.stage-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.approved-banner {
    background: linear-gradient(90deg, #11998e, #38ef7d);
    color: white;
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    font-weight: 600;
    margin: 0.5rem 0;
}
.locked-banner {
    background: #2c2c2c;
    color: #888;
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    font-weight: 600;
    margin: 0.5rem 0;
}
.result-card {
    border: 1px solid #333;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #1a1a2e;
}
.stProgress > div > div > div > div { background-color: #667eea; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
def _init_state():
    if "current_stage" not in st.session_state:
        st.session_state["current_stage"] = 1
    if "approved_up_to" not in st.session_state:
        st.session_state["approved_up_to"] = 0   # nothing approved yet; Stage 1 unlocked, Stage 2+ locked
    if "stage_results" not in st.session_state:
        st.session_state["stage_results"] = {}    # stage_num -> result dict
    if "working_data" not in st.session_state:
        st.session_state["working_data"] = []     # current list of dicts
    if "page" not in st.session_state:
        st.session_state["page"] = "📤 Upload & Ingest"
    if "selected_job" not in st.session_state:
        st.session_state["selected_job"] = None
    if "domain" not in st.session_state:
        st.session_state["domain"] = ""
    if "session_job_ids" not in st.session_state:
        st.session_state["session_job_ids"] = []  # Track jobs created THIS session only

_init_state()

# ---------------------------------------------------------------------------
# Sidebar – sequential pipeline navigator
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏭 LLM Data Factory")
    st.markdown("---")

    approved = st.session_state["approved_up_to"]
    current  = st.session_state["current_stage"]
    has_data = bool(st.session_state.get("working_data"))

    st.markdown("### 📋 Pipeline Progress")

    for icon, label, stage_num in PIPELINE_STAGES:
        is_current  = (stage_num == current)
        is_approved = (stage_num <= approved)
        # Unlock ALL stages once working data exists; always unlock Stage 1
        is_locked   = False if (stage_num == 1 or has_data) else (stage_num > 1)

        if is_locked:
            st.markdown(
                f"<div style='color:#555;padding:4px 0'>🔒 {stage_num}. {label}</div>",
                unsafe_allow_html=True,
            )
        else:
            label_text = f"{icon} {stage_num}. {label}"
            if is_approved and not is_current:
                label_text = "✅ " + label_text
            btn_type = "primary" if is_current else "secondary"
            if st.button(label_text, key=f"nav_{stage_num}", type=btn_type,
                         use_container_width=True, disabled=is_locked):
                st.session_state["current_stage"] = stage_num
                st.rerun()

    st.markdown("---")

    # Overall progress bar
    total = len(PIPELINE_STAGES)
    pct = int((approved / total) * 100)
    st.markdown(f"**Overall Progress: {pct}%**")
    st.progress(pct / 100)

    st.markdown("---")
    # API health — single placeholder prevents double-render
    api_status = st.empty()
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if r.status_code == 200:
            api_status.success("✅ API Online")
        else:
            api_status.error("❌ API Error")
    except Exception:
        api_status.error("❌ API Offline")

    # Working data status
    st.markdown("---")
    wd = st.session_state.get("working_data", [])
    if wd:
        st.success(f"📊 Working data: **{len(wd)} records**")
    else:
        st.warning("📭 No working data loaded")

# ---------------------------------------------------------------------------
# Helper: approve current stage
# ---------------------------------------------------------------------------
def approve_stage(stage_num: int):
    # Always set approved_up_to to at least stage_num
    if st.session_state["approved_up_to"] < stage_num:
        st.session_state["approved_up_to"] = stage_num
    # Clear ALL stage results to prevent ghost buttons on any page
    st.session_state["stage_results"].clear()
    # Advance to next stage automatically
    next_stage = stage_num + 1
    if next_stage <= len(PIPELINE_STAGES):
        st.session_state["current_stage"] = next_stage

def is_approved(stage_num: int) -> bool:
    return stage_num <= st.session_state["approved_up_to"]

def save_result(stage_num: int, result: dict):
    # Only store result for the stage that is currently active
    st.session_state["stage_results"][stage_num] = result

def get_result(stage_num: int) -> dict:
    return st.session_state["stage_results"].get(stage_num, {})

def _cleanup_stale_results(current_stage: int):
    """Remove any stage_results that don't belong to the current stage.
    This prevents ghost approve buttons from ever appearing."""
    stale = [k for k in st.session_state["stage_results"] if k != current_stage]
    for k in stale:
        del st.session_state["stage_results"][k]

# ---------------------------------------------------------------------------
# Contextual help panel shown at the top of every stage
# ---------------------------------------------------------------------------
STAGE_HELP = {
    1: ("📖 How to use Stage 1",
        """**What it does:** Accepts uploaded files (PDF, DOCX, TXT, CSV, Excel) and breaks them into
        text chunks stored in the database via the background worker (Celery).

        **For CSV files with single-word/value cells:**
        - Each *row* of your CSV becomes one chunk. The content of that chunk will be a comma-joined
          representation of all column values in that row.
        - If your cells contain single words (e.g. product codes, labels, numbers), use
          **Stage 6 (Data Transformation)** later to combine columns into a meaningful sentence.
        - ✅ Upload your file → wait for chunk count > 0 → select it below → click Load & Approve."""),
    2: ("📖 How to use Stage 2",
        """**What it does:** Fixes structural problems — missing values, duplicate rows, bad encoding,
        inconsistent date formats.

        **For CSV with single-word cells:**
        - Your data will appear as records with a `text` field (the chunk content) plus metadata.
        - Choose **'mark_null'** strategy to flag missing values without removing rows.
        - Run the step and inspect the Results section to see what was changed.
        - ✅ Click **Run Data Cleaning** → review results → click **Approve Stage 2**."""),
    3: ("📖 How to use Stage 3",
        """**What it does:** Cleans the text linguistically — removes HTML tags, URLs, extra whitespace,
        boilerplate headers/footers, and optionally lowercases text.

        **For CSV with short values:**
        - Select the `text` field in the multiselect (this holds your chunk content).
        - Keep **Remove HTML** and **Remove Boilerplate** enabled.
        - Uncheck **Lowercase** if your values are codes/labels that are case-sensitive.
        - ✅ Select fields → Run → Approve."""),
    4: ("📖 How to use Stage 4",
        """**What it does:** Removes toxic, spammy or too-short records from the dataset.

        **For CSV with single-word values:**
        - ⚠️ Set **Minimum Words per Entry** to **1** — otherwise short cell values will all be removed!
        - Select the `text` field in **Text Fields to Check**.
        - You can leave Remove Toxic / Remove Spam enabled — they won't fire on normal data.
        - ✅ Set min words → select field → Run → Approve."""),
    5: ("📖 How to use Stage 5",
        """**What it does:** Restructures records into the exact format your target LLM expects
        (QA pairs, instruction-response, chat messages, or retrieval documents).

        **For CSV with single-word cells:** Choose **retrieval_doc** to keep your data as-is for
        semantic search, or **instruction_tuning** if you want to map one column as input and another
        as the expected output.
        - ✅ Pick format → map fields → Apply Schema → Approve."""),
    6: ("📖 How to use Stage 6",
        """**What it does:** Combines columns, renders templates, converts rows to natural-language
        sentences, and adds metadata tags.

        **This is the KEY stage for CSV with single-word cells:**
        - Select **all your columns** in 'Columns to combine into text'.
        - Use a template like `{col1} is {col2} with value {col3}` to build a sentence per row.
        - Or enable **Convert rows to natural-language narrative** for automatic sentence generation.
        - ✅ Select columns → write template → Run → Approve."""),
    7: ("📖 How to use Stage 7",
        """**What it does:** Splits long text entries into smaller token-sized chunks that fit within
        your LLM's context window (e.g. 512 or 1024 tokens).

        **For short CSV values:** Most rows will already be short — set **Max Tokens** to 512 and
        overlap to 0. The stage will leave short entries untouched and only chunk oversized ones.
        - ✅ Select the text field → set chunk size → Run → Approve."""),
    8: ("📖 How to use Stage 8",
        """**What it does:** Balances class or topic distributions to prevent your model from being
        biased toward over-represented categories.

        **For CSV data:** Pick the column that contains your *category/label* field for balancing.
        Choose **'none'** if your dataset is already balanced, **'downsample'** to reduce the majority
        class, or **'oversample'** to duplicate minority samples.
        - ✅ Pick field → pick strategy → Run → Approve."""),
    9: ("📖 How to use Stage 9",
        """**What it does:** Automatically assigns labels to each record based on keyword matching.

        **For CSV data:** Edit the JSON label map to match keywords that appear in *your* data.
        Example: `{"electronics": ["phone","laptop"], "clothing": ["shirt","jeans"]}`.
        - ✅ Edit label map → pick text field → Run → Approve."""),
    10: ("📖 How to use Stage 10",
        """**What it does:** Validates record schema, checks JSON format integrity, and splits the
        dataset into train / validation / test sets.

        **Required fields:** Select any field that must be non-empty (e.g. `text`).
        Adjust the train/val/test ratios (default 80/10/10).
        - ✅ Select required fields → set ratios → Run → Approve."""),
    11: ("📖 How to use Stage 11",
        """**What it does:** Removes exact duplicate records and near-duplicates (records that are
        highly similar in wording).

        **For CSV data:** Select `text` for exact dedup. Set similarity threshold to 0.95 to only
        remove near-identical rows (lower = more aggressive removal).
        - ✅ Select dedup fields → Run → Approve."""),
    12: ("📖 How to use Stage 12",
        """**What it does:** Sends your cleaned, structured chunks to the LLM (Groq/Gemini) to
        generate Question-Answer pairs, summaries, or custom-formatted training samples.

        **For CSV data:** By this point your `text` field should contain meaningful sentences
        (built in Stage 6). The LLM will read each text chunk and produce training pairs from it.
        - ✅ Select document → choose task type → Start Generation → wait for completion
          → click View Results to load generated pairs → Approve."""),
    13: ("📖 How to use Stage 13 — Deterministic Evaluation",
        """**What it does:** Algorithmically scores your generated dataset using three
        fully deterministic NLP metrics — no LLMs involved:

        - **Relevance (1–5):** TF-IDF cosine similarity between the source context and the generated output.
          → 5 = strongly derived from source, 1 = unrelated.
        - **Coherence (1–5):** Grammar error density via LanguageTool.
          → 5 = well-structured, 1 = disjoint / incoherent.
        - **Bias (0/1):** Lexicon-based hate-speech and stereotype detection.
          → 0 = neutral, 1 = harmful content detected.

        **Modes:**
        - *Quick Evaluate* — paste a source + generated pair and get instant scores.
        - *Job Batch Evaluate* — score all entries in a completed generation job.
        - ✅ Review scores → inspect per-entry table → proceed to Export."""),
    14: ("📖 How to use Stage 14",
        """**What it does:** Saves a versioned snapshot of your cleaned dataset with full metadata
        for reproducibility — like a git commit for your data.

        - Give it a version tag (e.g. `v1_csv_cleaned`) and a description of your source data.
        - You can come back and compare versions later.
        - ✅ Fill in version info → Save Version → Approve."""),
    15: ("📖 How to use Stage 15",
        """**What it does:** Exports the final LLM-ready dataset in your chosen format:
        - **JSONL** — one JSON object per line, best for fine-tuning Llama / OpenAI / Gemini.
        - **CSV** — spreadsheet-friendly, easy to inspect.
        - **JSON** — single file array.
        - **Parquet** — compressed columnar format for large datasets.
        - ✅ Pick format → Export → Download the file."""),
}

def stage_help(stage_num: int):
    if stage_num in STAGE_HELP:
        title, body = STAGE_HELP[stage_num]
        with st.expander(title, expanded=False):
            st.markdown(body)

# ---------------------------------------------------------------------------
# Page header helper
# ---------------------------------------------------------------------------
def page_header(stage_num: int, icon: str, title: str, description: str):
    st.markdown(f'<h1 class="main-header">{icon} Stage {stage_num}: {title}</h1>', unsafe_allow_html=True)
    st.markdown(f"*{description}*")
    if is_approved(stage_num):
        st.markdown('<div class="approved-banner">✅ This stage has been approved — pipeline continues below.</div>',
                    unsafe_allow_html=True)
    stage_help(stage_num)
    st.markdown("---")

# ===========================================================================
# STAGE 1 – Upload & Ingest
# ===========================================================================
def stage_upload():
    page_header(1, "📤", "Upload & Ingest", "Upload raw documents or URLs for processing.")

    # Domain Input at the top of Pipeline
    st.markdown("### 🏷️ Pipeline Domain Context")
    st.info("Specify the domain of the data you are processing (e.g., 'Medical Data', 'Finance'). This context will be used during evaluation to ensure domain accuracy.")
    domain_input = st.text_input("Data Domain", value=st.session_state.get("domain", ""), placeholder="e.g. Medical Data, Legal Data...")
    if domain_input != st.session_state.get("domain"):
        st.session_state["domain"] = domain_input
    
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Upload File", "Upload URL", "View Documents", "🔍 Lookup Cleaned Data"])

    with tab1:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose one or more files",
            type=["pdf", "docx", "txt", "md", "xlsx", "xls", "csv"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("🚀 Process All Documents", type="primary"):
            prog = st.progress(0)
            success_count = 0
            for i, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Uploading {uploaded_file.name} ({i+1}/{len(uploaded_files)})…"):
                    response = requests.post(f"{API_BASE_URL}/ingestion/upload",
                                             files={"file": uploaded_file})
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {uploaded_file.name} — {result['message']} (ID: {result['document_id']})")
                        success_count += 1
                    else:
                        st.error(f"❌ {uploaded_file.name} — Upload failed: {response.text}")
                prog.progress((i + 1) / len(uploaded_files))
            st.info(f"📊 **{success_count}/{len(uploaded_files)}** documents uploaded successfully.")

    with tab2:
        st.subheader("Upload from URL")
        url      = st.text_input("Enter URL", placeholder="https://example.com/article")
        filename = st.text_input("Filename (optional)", placeholder="my_document.txt")
        crawl_depth = st.radio(
            "Crawl Depth",
            options=[1, 2],
            index=0,
            help="1 = single page only (recommended). 2 = also follow links on the page.",
            horizontal=True
        )
        st.caption("🧠 Smart extraction: removes navbars, footers, ads and keeps article content only.")
        if st.button("🌐 Fetch & Process", type="primary"):
            if url:
                payload = {"url": url, "max_depth": crawl_depth}
                if filename:
                    payload["filename"] = filename
                response = requests.post(f"{API_BASE_URL}/ingestion/upload-url", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {result['message']}")
                    st.info(f"Document ID: {result['document_id']}")
                else:
                    st.error(f"❌ Failed: {response.text}")
            else:
                st.warning("Please enter a URL")

    with tab3:
        st.subheader("Uploaded Documents")
        if st.button("🔄 Refresh"):
            st.rerun()
        try:
            response = requests.get(f"{API_BASE_URL}/ingestion/documents")
            if response.status_code == 200:
                documents = response.json()
                if documents:
                    for doc in documents:
                        c1, c2, c3, c4, c5, c6 = st.columns([1, 3, 2, 1, 2, 1])
                        c1.write(f"**{doc['id']}**")
                        c2.write(doc["filename"])
                        c3.write(doc["file_type"])
                        c4.write(f"{doc['chunk_count']} chunks")
                        c5.write(pd.to_datetime(doc["upload_timestamp"]).strftime("%Y-%m-%d %H:%M"))
                        if c6.button("🗑️", key=f"del_{doc['id']}"):
                            dr = requests.delete(f"{API_BASE_URL}/ingestion/documents/{doc['id']}")
                            if dr.status_code == 200:
                                st.rerun()
                        st.markdown("---")
                else:
                    st.info("No documents uploaded yet.")
        except Exception as e:
            st.error(f"Error: {e}")

    with tab4:
        st.subheader("🔍 Lookup Previously Cleaned Data")

        # ── Domain search bar ────────────────────────────────────────────────
        domain_search = st.text_input(
            "🔎 Search by Domain Name",
            placeholder="e.g. Medical Data, Finance, Legal...",
            help="Type a domain name to filter datasets. Leave empty to show all."
        )

        versions = VersionManager.list_versions()

        if not versions:
            st.info("No previously cleaned datasets are available yet. Complete the pipeline and save a version in Stage 14 to see it here!")
        else:
            # Filter by search bar input (case-insensitive partial match)
            if domain_search.strip():
                display_versions = [
                    v for v in versions
                    if domain_search.strip().lower() in v.get("domain", "").lower()
                ]
                if display_versions:
                    st.success(f"Found **{len(display_versions)}** dataset(s) matching domain: **{domain_search.strip()}**")
                else:
                    st.warning(f"No datasets found for domain **'{domain_search.strip()}'**. Try a different name or leave the search bar empty to show all.")
            else:
                display_versions = versions
                st.write(f"Showing all **{len(display_versions)}** available versioned dataset(s):")
                
            for v in display_versions:
                with st.container():
                    st.markdown(f"**Version Tag:** `{v.get('version_tag')}` "
                                f"(Domain: _{v.get('domain', 'Unspecified')}_)")
                    st.markdown(f"**Description:** {v.get('data_source', v.get('description', 'N/A'))}")
                    st.markdown(f"**Snapshot Time:** {v.get('timestamp')}")
                    st.markdown(f"**Records:** {v.get('record_count')}")
                    
                    file_path = v.get("file", "")
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as file:
                            btn = st.download_button(
                                label="📥 Download Data",
                                data=file,
                                file_name=os.path.basename(file_path),
                                mime="application/jsonl",
                                key=f"dl_{v['version_tag']}_{v['timestamp']}"
                            )
                    else:
                        st.error("File not found on server.")
                    st.markdown("---")

    st.markdown("---")
    st.subheader("📦 Load Documents into Pipeline")
    st.info("Select which uploaded documents to load as working data for the pipeline.")

    try:
        doc_resp = requests.get(f"{API_BASE_URL}/ingestion/documents", timeout=5)
        all_docs = doc_resp.json() if doc_resp.status_code == 200 else []
    except Exception:
        all_docs = []

    if not all_docs:
        st.warning("⚠️ No documents found. Upload at least one document first.")
    else:
        docs_with_chunks = [d for d in all_docs if d.get("chunk_count", 0) > 0]
        if not docs_with_chunks:
            st.warning("⚠️ Documents are still being processed (0 chunks). Wait a moment and refresh.")
        else:
            doc_options = {f"[{d['id']}] {d['filename']} ({d['chunk_count']} chunks)": d["id"]
                           for d in docs_with_chunks}
            selected_labels = st.multiselect(
                "Select documents to load",
                list(doc_options.keys()),
                default=list(doc_options.keys()),
            )
            limit = st.number_input("Max chunks per document", min_value=10, max_value=5000,
                                    value=500, step=50)

            if st.button("📥 Load Selected Documents & Approve Stage 1", type="primary"):
                selected_ids = [doc_options[lbl] for lbl in selected_labels]
                if not selected_ids:
                    st.error("Please select at least one document.")
                else:
                    records = []
                    errors = []
                    prog = st.progress(0)
                    status_txt = st.empty()
                    for i, doc_id in enumerate(selected_ids):
                        status_txt.info(f"Fetching chunks for document ID {doc_id}…")
                        try:
                            r = requests.get(
                                f"{API_BASE_URL}/ingestion/documents/{doc_id}/chunks",
                                params={"limit": int(limit), "skip": 0},
                                timeout=30,
                            )
                            if r.status_code == 200:
                                chunks = r.json()
                                for ch in chunks:
                                    records.append({
                                        "text":        ch.get("content", ""),
                                        "chunk_index": ch.get("chunk_index", 0),
                                        "token_count": ch.get("token_count"),
                                        "document_id": doc_id,
                                        "source":      next(
                                            (d["filename"] for d in docs_with_chunks if d["id"] == doc_id),
                                            str(doc_id),
                                        ),
                                    })
                            else:
                                errors.append(f"Doc {doc_id}: HTTP {r.status_code} — {r.text[:200]}")
                        except Exception as e:
                            errors.append(f"Doc {doc_id}: {e}")
                        prog.progress((i + 1) / len(selected_ids))

                    status_txt.empty()
                    if errors:
                        st.error("Errors encountered:\n" + "\n".join(errors))

                    st.write(f"**Debug:** fetched {len(records)} total chunks from {len(selected_ids)} doc(s)")

                    if records:
                        st.session_state["working_data"] = records
                        approve_stage(1)
                        st.success(
                            f"✅ {len(records)} chunks loaded from {len(selected_ids)} document(s). "
                            f"Stage 1 approved — click '2. Data Cleaning' in the sidebar."
                        )
                        st.rerun()
                    else:
                        st.error(
                            "❌ No chunks fetched. Possible reasons:\n"
                            "- Documents still processing (chunk_count shows 0)\n"
                            "- Celery worker not running\n"
                            "- API error (see above)"
                        )

# ===========================================================================
# STAGE 12 – Generate Dataset
# ===========================================================================
def stage_generate():
    page_header(12, "⚙️", "Generate Dataset", "Generate QA pairs / summaries using LLM.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Create New Job")
        try:
            response = requests.get(f"{API_BASE_URL}/ingestion/documents")
            documents = response.json() if response.status_code == 200 else []
            if documents:
                doc_options = {f"{d['id']}: {d['filename']}": d["id"] for d in documents}
                sel_doc   = st.selectbox("Select Document", list(doc_options.keys()))
                doc_id    = doc_options[sel_doc]
                task_type = st.selectbox("Task Type", ["qa_generation", "summarization", "custom"])
                ds_name   = st.text_input("Dataset Name (optional)", placeholder="My_Dataset")
                custom_p  = st.text_area("Custom Instruction", height=80) if task_type == "custom" else None

                # ── Few-Shot Examples ──────────────────────────────────────
                few_shot = []
                with st.expander("🎯 Few-Shot Examples (optional — guide LLM style)", expanded=False):
                    st.caption(
                        "Provide example outputs to show the LLM the exact style, "
                        "length, and format you want. Leave empty to use default prompts."
                    )

                    fse_tab_file, fse_tab_manual = st.tabs(["📂 Upload File", "✏️ Enter Manually"])

                    # ── Tab 1: File Upload ────────────────────────────────
                    with fse_tab_file:
                        # Dynamic format hint per task type
                        if task_type == "qa_generation":
                            fmt_hint = (
                                "**Supported formats:**\n"
                                "- **CSV** — columns: `question`, `answer`\n"
                                "- **JSON / JSONL** — list of `{\"question\": ..., \"answer\": ...}`\n"
                                "- **TXT** — pairs separated by `Q:` / `A:` markers"
                            )
                        elif task_type == "summarization":
                            fmt_hint = (
                                "**Supported formats:**\n"
                                "- **CSV** — columns: `paragraph`, `summary`\n"
                                "- **JSON / JSONL** — list of `{\"paragraph\": ..., \"summary\": ...}`\n"
                                "- **TXT** — pairs separated by `PARAGRAPH:` / `SUMMARY:` markers"
                            )
                        else:
                            fmt_hint = (
                                "**Supported formats:**\n"
                                "- **CSV** — columns: `input`, `output`\n"
                                "- **JSON / JSONL** — list of `{\"input\": ..., \"output\": ...}`\n"
                                "- **TXT** — pairs separated by `INPUT:` / `OUTPUT:` markers"
                            )
                        st.markdown(fmt_hint)

                        uploaded_fse = st.file_uploader(
                            "Upload examples file",
                            type=["csv", "json", "jsonl", "txt"],
                            key="fse_file_upload"
                        )

                        if uploaded_fse:
                            try:
                                import json as _json
                                import io

                                fname = uploaded_fse.name.lower()
                                parsed = []

                                if fname.endswith(".csv"):
                                    import pandas as _pd
                                    df = _pd.read_csv(io.StringIO(uploaded_fse.read().decode("utf-8")))
                                    df.columns = [c.strip().lower() for c in df.columns]
                                    if task_type == "qa_generation":
                                        for _, row in df.iterrows():
                                            q = str(row.get("question", "")).strip()
                                            a = str(row.get("answer", "")).strip()
                                            if q and a:
                                                parsed.append({"question": q, "answer": a})
                                    elif task_type == "summarization":
                                        for _, row in df.iterrows():
                                            p = str(row.get("paragraph", "")).strip()
                                            s = str(row.get("summary", "")).strip()
                                            if p and s:
                                                parsed.append({"paragraph": p, "summary": s})
                                    else:
                                        for _, row in df.iterrows():
                                            i = str(row.get("input", "")).strip()
                                            o = str(row.get("output", "")).strip()
                                            if i and o:
                                                parsed.append({"input": i, "output": o})

                                elif fname.endswith(".json") or fname.endswith(".jsonl"):
                                    content = uploaded_fse.read().decode("utf-8")
                                    # Try JSONL (one object per line) first
                                    try:
                                        parsed = [_json.loads(line) for line in content.strip().splitlines() if line.strip()]
                                    except Exception:
                                        parsed = _json.loads(content)
                                    if not isinstance(parsed, list):
                                        parsed = [parsed]

                                elif fname.endswith(".txt"):
                                    # Normalise line endings (Windows CRLF → LF) before splitting
                                    raw_bytes = uploaded_fse.read()
                                    content = raw_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
                                    if task_type == "qa_generation":
                                        blocks = content.split("\n\n")
                                        for block in blocks:
                                            q, a = "", ""
                                            for line in block.splitlines():
                                                line = line.strip()
                                                if line.upper().startswith("Q:"):
                                                    q = line[2:].strip()
                                                elif line.upper().startswith("A:"):
                                                    a = line[2:].strip()
                                            if q and a:
                                                parsed.append({"question": q, "answer": a})
                                    elif task_type == "summarization":
                                        blocks = content.split("\n\n")
                                        for block in blocks:
                                            p, s = "", ""
                                            for line in block.splitlines():
                                                line = line.strip()
                                                if line.upper().startswith("PARAGRAPH:"):
                                                    p = line[10:].strip()
                                                elif line.upper().startswith("SUMMARY:"):
                                                    s = line[8:].strip()
                                            if p and s:
                                                parsed.append({"paragraph": p, "summary": s})
                                    else:
                                        blocks = content.split("\n\n")
                                        for block in blocks:
                                            inp, out = "", ""
                                            for line in block.splitlines():
                                                line = line.strip()
                                                if line.upper().startswith("INPUT:"):
                                                    inp = line[6:].strip()
                                                elif line.upper().startswith("OUTPUT:"):
                                                    out = line[7:].strip()
                                            if inp and out:
                                                parsed.append({"input": inp, "output": out})

                                if parsed:
                                    few_shot = parsed[:10]  # Allow up to 10 examples from file
                                    st.success(f"✅ Loaded **{len(few_shot)}** of {len(parsed)} example(s) from `{uploaded_fse.name}`")
                                    # Show inline preview — NO nested expander (Streamlit forbids it)
                                    st.markdown("**Preview:**")
                                    for idx, ex in enumerate(few_shot):
                                        keys = list(ex.keys())
                                        if len(keys) >= 2:
                                            st.markdown(
                                                f"*{idx+1}.* **{keys[0].capitalize()}:** {str(ex[keys[0]])[:120]}  \n"
                                                f"&nbsp;&nbsp;&nbsp;&nbsp;**{keys[1].capitalize()}:** {str(ex[keys[1]])[:120]}"
                                            )
                                else:
                                    st.warning("⚠️ No valid examples found — check column names match the format above.")

                            except Exception as e:
                                st.error(f"❌ Failed to parse file: {e}")

                    # ── Tab 2: Manual Entry ───────────────────────────────
                    with fse_tab_manual:
                        num_examples = st.number_input("Number of examples", 1, 5, 1, key="fse_count")

                        manual_examples = []
                        if task_type == "qa_generation":
                            st.markdown("**Example Question–Answer Pairs:**")
                            for i in range(int(num_examples)):
                                st.markdown(f"*Example {i+1}*")
                                q = st.text_input(f"Question {i+1}", key=f"fse_q_{i}",
                                                  placeholder="What is the main topic?")
                                a = st.text_area(f"Answer {i+1}", key=f"fse_a_{i}", height=70,
                                                 placeholder="The main topic is…")
                                if q and a:
                                    manual_examples.append({"question": q, "answer": a})

                        elif task_type == "summarization":
                            st.markdown("**Example Paragraph → Summary Pairs:**")
                            for i in range(int(num_examples)):
                                st.markdown(f"*Example {i+1}*")
                                para = st.text_area(f"Paragraph {i+1}", key=f"fse_para_{i}", height=80,
                                                    placeholder="Original paragraph text…")
                                summ = st.text_input(f"Summary {i+1}", key=f"fse_sum_{i}",
                                                     placeholder="One-sentence summary…")
                                if para and summ:
                                    manual_examples.append({"paragraph": para, "summary": summ})

                        elif task_type == "custom":
                            st.markdown("**Example Input → Output Pairs:**")
                            for i in range(int(num_examples)):
                                st.markdown(f"*Example {i+1}*")
                                inp = st.text_area(f"Input {i+1}", key=f"fse_in_{i}", height=60,
                                                   placeholder="Input text…")
                                out = st.text_area(f"Output {i+1}", key=f"fse_out_{i}", height=60,
                                                   placeholder="Expected output…")
                                if inp and out:
                                    manual_examples.append({"input": inp, "output": out})

                        # Manual examples override file examples if both provided
                        if manual_examples:
                            few_shot = manual_examples

                    if few_shot:
                        st.success(f"✅ **{len(few_shot)}** example(s) will be injected into the LLM prompt.")

                if st.button("🚀 Start Generation", type="primary"):
                    payload = {"document_id": doc_id, "task_type": task_type}
                    if ds_name:
                        payload["dataset_name"] = ds_name
                    if custom_p:
                        payload["custom_prompt"] = custom_p
                    if few_shot:
                        payload["few_shot_examples"] = few_shot
                    r = requests.post(f"{API_BASE_URL}/jobs/generate", json=payload)
                    if r.status_code == 200:
                        res = r.json()
                        # Register this job ID in session so it shows in Active Jobs
                        if res["job_id"] not in st.session_state["session_job_ids"]:
                            st.session_state["session_job_ids"].append(res["job_id"])
                        st.success(f"✅ {res['message']}")
                        st.info(f"Job ID: {res['job_id']}")
                    else:
                        st.error(f"❌ {r.text}")
            else:
                st.warning("No documents available.")
        except Exception as e:
            st.error(f"Error: {e}")

    with col2:
        st.subheader("Active Jobs")
        if st.button("🔄 Refresh Jobs"):
            st.rerun()

        session_job_ids = st.session_state.get("session_job_ids", [])
        if not session_job_ids:
            st.info("No jobs started in this session yet. Start a generation job on the left.")
        else:
            try:
                response = requests.get(f"{API_BASE_URL}/jobs/")
                if response.status_code == 200:
                    # Only show jobs that were created in THIS session
                    all_jobs = response.json()
                    session_jobs = [j for j in all_jobs if j["job_id"] in session_job_ids]
                    if not session_jobs:
                        st.info("No active jobs for this session.")
                    for job in session_jobs:
                        with st.expander(f"Job {job['job_id'][:8]}… | {job['status']}"):
                            ca, cb = st.columns(2)
                            ca.metric("Doc ID",   job["document_id"])
                            ca.metric("Type",     job["task_type"])
                            cb.metric("Progress", f"{job['progress_percentage']:.1f}%")
                            cb.metric("Chunks",   f"{job['processed_chunks']}/{job['total_chunks']}")
                            if job["status"] in ("processing", "pending"):
                                st.progress(job["progress_percentage"] / 100)
                            if job["status"] == "completed":
                                if st.button("📊 View Results", key=f"view_{job['job_id']}"):
                                    # Load LLM-generated data into working set
                                    r2 = requests.get(
                                        f"{API_BASE_URL}/jobs/{job['job_id']}/results?limit=5000"
                                    )
                                    if r2.status_code == 200:
                                        raw_entries = r2.json()
                                        # ── Strip DB metadata — keep only clean NL fields ──
                                        # Determine task type to know which fields matter
                                        jtype = job.get("task_type", "qa_generation")
                                        clean_records = []
                                        for entry in raw_entries:
                                            q   = (entry.get("question") or "").strip()
                                            a   = (entry.get("answer")   or "").strip()
                                            ctx = (entry.get("context")  or "").strip()
                                            # Skip entries with missing core fields
                                            if not q or not a:
                                                continue
                                            if jtype == "summarization":
                                                clean_records.append({
                                                    "question": q,
                                                    "context":  ctx,
                                                    "answer":   a,
                                                })
                                            else:
                                                clean_records.append({
                                                    "question": q,
                                                    "answer":   a,
                                                })
                                        st.session_state["working_data"] = clean_records
                                        st.session_state["selected_job"] = job["job_id"]
                                        st.success(f"✅ Loaded {len(clean_records)} clean NL records into pipeline.")
                            if job.get("error_message"):
                                st.error(job["error_message"])
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    rec_count = len(st.session_state.get("working_data", []))
    st.info(f"**Working dataset size:** {rec_count} records")
    if not is_approved(12):
        if st.button("✅ Approve Stage 12 & Continue →", type="primary", key=f"approve_btn_12_{st.session_state.get('current_stage', 0)}"):
            approve_stage(12)
            st.rerun()
    else:
        st.success("✅ Stage 12 already approved.")

# ===========================================================================
# STAGE 2 – Data Cleaning
# ===========================================================================
def stage_data_cleaning():
    page_header(2, "🧹", "Data Cleaning", "Fix structural inconsistencies in the dataset.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data loaded. Please go back to Stage 1 and load documents first.")

        st.markdown("---")
        st.subheader("🔄 Or reload document chunks here")
        try:
            doc_resp = requests.get(f"{API_BASE_URL}/ingestion/documents", timeout=5)
            all_docs = [d for d in (doc_resp.json() if doc_resp.status_code == 200 else [])
                        if d.get("chunk_count", 0) > 0]
        except Exception:
            all_docs = []

        if all_docs:
            doc_options = {f"[{d['id']}] {d['filename']} ({d['chunk_count']} chunks)": d["id"]
                           for d in all_docs}
            selected_labels = st.multiselect("Select documents", list(doc_options.keys()),
                                             default=list(doc_options.keys()),
                                             key="stage2_reload_docs")
            if st.button("📥 Load Chunks into Pipeline"):
                records = []
                for lbl in selected_labels:
                    doc_id = doc_options[lbl]
                    r = requests.get(f"{API_BASE_URL}/ingestion/documents/{doc_id}/chunks",
                                     params={"limit": 500}, timeout=10)
                    if r.status_code == 200:
                        for ch in r.json():
                            records.append({
                                "text": ch.get("content", ""),
                                "chunk_index": ch.get("chunk_index", 0),
                                "document_id": doc_id,
                                "source": next((d["filename"] for d in all_docs if d["id"] == doc_id), str(doc_id)),
                            })
                if records:
                    st.session_state["working_data"] = records
                    st.success(f"✅ Loaded {len(records)} chunks. Re-running…")
                    st.rerun()
                else:
                    st.error("No chunks found.")
        else:
            st.info("No processed documents available.")
        return

    with st.expander("⚙️ Settings", expanded=True):
        missing_strategy = st.selectbox("Missing Value Strategy",
                                         ["mark_null", "remove", "impute_empty"])
        date_fields_raw  = st.text_input("Date Fields (comma-separated, optional)", "")
        date_fields      = [f.strip() for f in date_fields_raw.split(",") if f.strip()]

    if st.button("▶️ Run Data Cleaning", type="primary"):
        with st.spinner("Cleaning…"):
            result = DataCleaner.run_all(records, missing_strategy, date_fields)
        save_result(2, result)
        st.session_state["working_data"] = result["records"]
        st.success(f"✅ Cleaning complete. {len(result['records'])} records remaining.")

    result = get_result(2)
    if result:
        st.subheader("📊 Results")
        for step, stats in result.get("steps", {}).items():
            with st.expander(f"Step: {step.replace('_', ' ').title()}"):
                st.json(stats)

        # Sample preview
        st.subheader("🔍 Sample Records (first 5)")
        st.dataframe(pd.DataFrame(result["records"][:5]), use_container_width=True)

        st.markdown("---")
        if not is_approved(2):
            if st.button("✅ Approve Stage 2 & Continue →", type="primary", key=f"approve_btn_2_{st.session_state.get('current_stage', 0)}"):
                approve_stage(2)
                st.rerun()
        else:
            st.success("✅ Stage 2 already approved.")

# ===========================================================================
# STAGE 3 – Text Normalization
# ===========================================================================
def stage_text_normalization():
    page_header(3, "🔤", "Text Normalization",
                "Linguistically clean all text fields in the dataset.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data. Complete earlier stages first.")
        return

    all_fields = list(records[0].keys()) if records else []

    with st.expander("⚙️ Settings", expanded=True):
        text_fields  = st.multiselect("Text Fields to Normalize",
                                       all_fields,
                                       default=[f for f in ["question", "answer", "text"] if f in all_fields])
        remove_html  = st.checkbox("Remove HTML/markup", value=True)
        remove_urls  = st.checkbox("Remove URLs", value=False)
        lowercase    = st.checkbox("Lowercase", value=False)
        rem_boiler   = st.checkbox("Remove Boilerplate (headers/footers)", value=True)

    if st.button("▶️ Run Text Normalization", type="primary"):
        with st.spinner("Normalizing…"):
            result = TextNormalizer.normalize_records(
                records, text_fields,
                remove_html=remove_html,
                remove_urls=remove_urls,
                lowercase=lowercase,
                remove_boilerplate_=rem_boiler,
            )
        save_result(3, result)
        st.session_state["working_data"] = result["records"]
        st.success("✅ Normalization complete.")

    result = get_result(3)
    if result:
        st.subheader("📊 Results")
        st.json(result.get("stats", {}))
        st.subheader("🔍 Sample (first 5)")
        st.dataframe(pd.DataFrame(result["records"][:5]), use_container_width=True)

        st.markdown("---")
        if not is_approved(3):
            if st.button("✅ Approve Stage 3 & Continue →", type="primary", key=f"approve_btn_3_{st.session_state.get('current_stage', 0)}"):
                approve_stage(3)
                st.rerun()
        else:
            st.success("✅ Stage 3 already approved.")

# ===========================================================================
# STAGE 4 – Content Filtering & QC
# ===========================================================================
def stage_content_filtering():
    page_header(4, "🛡️", "Content Filtering & Quality Control",
                "Remove toxic, spammy, or low-quality entries.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data. Complete earlier stages first.")
        return

    all_fields = list(records[0].keys()) if records else []

    with st.expander("⚙️ Settings", expanded=True):
        text_fields  = st.multiselect("Text Fields to Check",
                                       all_fields,
                                       default=[f for f in ["text", "question", "answer"] if f in all_fields])
        min_words    = st.slider("Minimum Words per Entry", 1, 100, 1)
        rem_toxic    = st.checkbox("Remove Toxic Content", value=True)
        rem_spam     = st.checkbox("Remove Spam", value=True)

    if st.button("▶️ Run Content Filtering", type="primary"):
        with st.spinner("Filtering…"):
            result = ContentFilter.filter_records(
                records, text_fields,
                min_words=min_words,
                remove_toxic=rem_toxic,
                remove_spam=rem_spam,
            )
        save_result(4, result)
        st.session_state["working_data"] = result["records"]
        st.success(f"✅ Filtering done. Kept {result['stats']['kept']} / {result['stats']['original']} records.")

    result = get_result(4)
    if result:
        st.subheader("📊 Filtering Stats")
        cols = st.columns(4)
        cols[0].metric("Total",   result["stats"]["original"])
        cols[1].metric("Kept",    result["stats"]["kept"])
        cols[2].metric("Removed", result["stats"]["removed"])
        cols[3].metric("Removed %", f"{result['stats']['removed']/max(result['stats']['original'],1)*100:.1f}%")

        if result["stats"].get("by_reason"):
            fig = px.bar(
                x=list(result["stats"]["by_reason"].keys()),
                y=list(result["stats"]["by_reason"].values()),
                labels={"x": "Reason", "y": "Count"},
                title="Removed by Reason",
                color_discrete_sequence=["#764ba2"],
            )
            st.plotly_chart(fig, use_container_width=True)

        if result.get("filtered"):
            with st.expander("Filtered Records Log"):
                st.dataframe(pd.DataFrame(result["filtered"]), use_container_width=True)

        st.markdown("---")
        if not is_approved(4):
            if st.button("✅ Approve Stage 4 & Continue →", type="primary", key=f"approve_btn_4_{st.session_state.get('current_stage', 0)}"):
                approve_stage(4)
                st.rerun()
        else:
            st.success("✅ Stage 4 already approved.")

# ===========================================================================
# STAGE 5 – Schema Design
# ===========================================================================
def stage_schema_design():
    page_header(5, "📐", "Schema Design",
                "Restructure cleaned data into the format required by your LLM.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data. Complete earlier stages first.")
        return

    all_fields = list(records[0].keys()) if records else []

    with st.expander("⚙️ Settings", expanded=True):
        target_format = st.selectbox(
            "Target LLM Format",
            ["qa_pairs", "instruction_tuning", "chat_format", "retrieval_doc"],
        )

        st.markdown(f"**Configure field mapping for `{target_format}`:**")
        field_map = {}

        if target_format == "instruction_tuning":
            field_map["instruction"] = st.selectbox("Instruction field", all_fields,
                                                      index=all_fields.index("question") if "question" in all_fields else 0)
            field_map["input"]       = st.selectbox("Input (context) field", ["(none)"] + all_fields)
            field_map["output"]      = st.selectbox("Output field", all_fields,
                                                      index=all_fields.index("answer") if "answer" in all_fields else 0)
            if field_map["input"] == "(none)":
                field_map["input"] = ""

        elif target_format == "chat_format":
            field_map["user"]        = st.selectbox("User turn field", all_fields,
                                                      index=all_fields.index("question") if "question" in all_fields else 0)
            field_map["assistant"]   = st.selectbox("Assistant turn field", all_fields,
                                                      index=all_fields.index("answer") if "answer" in all_fields else 0)
            field_map["system_prompt"] = st.text_input("System prompt",
                                                         "You are a helpful assistant.")

        elif target_format == "retrieval_doc":
            field_map["text"]        = st.selectbox("Text field", all_fields,
                                                      index=all_fields.index("answer") if "answer" in all_fields else 0)
            meta_fields              = st.multiselect("Metadata fields", all_fields)
            field_map["metadata_fields"] = meta_fields or None

        elif target_format == "qa_pairs":
            field_map["question"]    = st.selectbox("Question field", all_fields,
                                                      index=all_fields.index("question") if "question" in all_fields else 0)
            field_map["answer"]      = st.selectbox("Answer field", all_fields,
                                                      index=all_fields.index("answer") if "answer" in all_fields else 0)
            ctx_field                = st.selectbox("Context field (optional)", ["(none)"] + all_fields)
            field_map["context"]     = ctx_field if ctx_field != "(none)" else None

    if st.button("▶️ Apply Schema", type="primary"):
        with st.spinner("Applying schema…"):
            result = SchemaDesigner.convert(records, target_format, field_map)
        save_result(5, result)
        st.session_state["working_data"] = result["records"]
        st.success(f"✅ Schema applied. {result['count']} records converted to `{target_format}`.")

    result = get_result(5)
    if result:
        st.subheader("📊 Results")
        st.info(f"Format: `{result.get('format')}` | Records: {result.get('count')}")
        st.subheader("🔍 Sample Output (first 3)")
        for rec in result["records"][:3]:
            st.json(rec)

        st.markdown("---")
        if not is_approved(5):
            if st.button("✅ Approve Stage 5 & Continue →", type="primary", key=f"approve_btn_5_{st.session_state.get('current_stage', 0)}"):
                approve_stage(5)
                st.rerun()
        else:
            st.success("✅ Stage 5 already approved.")

# ===========================================================================
# STAGE 6 – Data Transformation / Feature Engineering
# ===========================================================================
def stage_transformation():
    page_header(6, "🔧", "Data Transformation / Feature Engineering",
                "Convert raw fields into meaningful textual signals.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data.")
        return

    all_fields = list(records[0].keys()) if records else []

    with st.expander("⚙️ Settings", expanded=True):
        combine_cols = st.multiselect("Columns to combine into text", all_fields)
        template     = st.text_input("Jinja-style template (optional)",
                                      placeholder="{question} — {answer}")
        to_narrative = st.checkbox("Convert rows to natural-language narrative", value=False)
        output_field = st.text_input("Output field name", "transformed_text")
        tags_raw     = st.text_input("Add metadata tags (JSON object)",
                                      placeholder='{"source": "my_doc", "lang": "en"}')

    if st.button("▶️ Run Transformations", type="primary"):
        tags = None
        if tags_raw.strip():
            try:
                tags = json.loads(tags_raw)
            except Exception:
                st.error("Invalid JSON for tags.")
                return
        with st.spinner("Transforming…"):
            result = DataTransformer.run_transformations(
                records,
                combine_cols=combine_cols or None,
                template=template or None,
                to_narrative=to_narrative,
                tags=tags,
                output_field=output_field,
            )
        save_result(6, result)
        st.session_state["working_data"] = result["records"]
        st.success(f"✅ Transformation done. Ops: {result['ops_applied']}")

    result = get_result(6)
    if result:
        st.subheader("🔍 Sample (first 5)")
        st.dataframe(pd.DataFrame(result["records"][:5]), use_container_width=True)

        st.markdown("---")
        if not is_approved(6):
            if st.button("✅ Approve Stage 6 & Continue →", type="primary", key=f"approve_btn_6_{st.session_state.get('current_stage', 0)}"):
                approve_stage(6)
                st.rerun()
        else:
            st.success("✅ Stage 6 already approved.")

# ===========================================================================
# STAGE 7 – Tokenization & Length Control
# ===========================================================================
def stage_tokenization():
    page_header(7, "✂️", "Tokenization & Length Control",
                "Chunk long entries to fit LLM context windows.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data.")
        return

    all_fields = list(records[0].keys()) if records else []

    with st.expander("⚙️ Settings", expanded=True):
        text_field  = st.selectbox("Text field to tokenize",
                                    all_fields,
                                    index=all_fields.index("answer") if "answer" in all_fields else 0)
        chunk_size  = st.slider("Max Tokens per Chunk", 128, 2048, 512)
        overlap     = st.slider("Overlap Tokens", 0, 256, 64)

    if st.button("▶️ Run Tokenization", type="primary"):
        with st.spinner("Tokenizing…"):
            ctrl   = TokenizationController(chunk_size=chunk_size, overlap=overlap)
            result = ctrl.process_records(records, text_field)
        save_result(7, result)
        st.session_state["working_data"] = result["records"]
        st.success(f"✅ Tokenization done.")

    result = get_result(7)
    if result:
        st.subheader("📊 Stats")
        stats = result.get("stats", {})
        c = st.columns(4)
        c[0].metric("Original Records", stats.get("original_records"))
        c[1].metric("Output Records",   stats.get("output_records"))
        c[2].metric("Oversized Chunked",stats.get("oversized_chunked"))
        c[3].metric("Chunk Size",       stats.get("chunk_size"))

        token_counts = [r.get("_token_count", 0) for r in result["records"]]
        if token_counts:
            fig = px.histogram(x=token_counts, nbins=30, title="Token Count Distribution",
                               color_discrete_sequence=["#667eea"])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        if not is_approved(7):
            if st.button("✅ Approve Stage 7 & Continue →", type="primary", key=f"approve_btn_7_{st.session_state.get('current_stage', 0)}"):
                approve_stage(7)
                st.rerun()
        else:
            st.success("✅ Stage 7 already approved.")

# ===========================================================================
# STAGE 8 – Dataset Balancing
# ===========================================================================
def stage_balancing():
    page_header(8, "⚖️", "Dataset Balancing",
                "Prevent class imbalance and topical skew.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data.")
        return

    all_fields = list(records[0].keys()) if records else []

    with st.expander("⚙️ Settings", expanded=True):
        balance_field = st.selectbox("Field to balance on", all_fields)
        strategy      = st.selectbox("Strategy",
                                      ["none", "downsample", "oversample"])

    if st.button("▶️ Run Balancing", type="primary"):
        with st.spinner("Balancing…"):
            result = DatasetBalancer.run_balancing(records, balance_field, strategy)
        save_result(8, result)
        st.session_state["working_data"] = result["records"]
        st.success("✅ Balancing done.")

    result = get_result(8)
    if result:
        stats = result.get("stats", {})
        if stats:
            st.subheader("📊 Distribution Comparison")
            orig = stats.get("original_distribution", {})
            new_ = stats.get("new_distribution", {})

            all_keys = sorted(set(list(orig.keys()) + list(new_.keys())))
            df_dist  = pd.DataFrame({
                "Class":    all_keys,
                "Before":   [orig.get(k, 0) for k in all_keys],
                "After":    [new_.get(k,  0) for k in all_keys],
            })
            fig = px.bar(df_dist, x="Class", y=["Before", "After"],
                         barmode="group", title="Class Distribution Before vs After",
                         color_discrete_sequence=["#667eea", "#f093fb"])
            st.plotly_chart(fig, use_container_width=True)

            ca, cb = st.columns(2)
            ca.metric("Before",   stats.get("original_count"))
            cb.metric("After",    stats.get("new_count"))

        st.markdown("---")
        if not is_approved(8):
            if st.button("✅ Approve Stage 8 & Continue →", type="primary", key=f"approve_btn_8_{st.session_state.get('current_stage', 0)}"):
                approve_stage(8)
                st.rerun()
        else:
            st.success("✅ Stage 8 already approved.")

# ===========================================================================
# STAGE 9 – Annotation / Labeling
# ===========================================================================
def stage_annotation():
    page_header(9, "🏷️", "Annotation / Labeling",
                "Auto-label records using keyword rules.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data.")
        return

    all_fields = list(records[0].keys()) if records else []

    with st.expander("⚙️ Settings", expanded=True):
        text_field    = st.selectbox("Text field to label",
                                      all_fields,
                                      index=all_fields.index("question") if "question" in all_fields else 0)
        output_field  = st.text_input("Label output field", "label")
        label_map_raw = st.text_area(
            "Label → Keywords mapping (JSON)",
            value='{"general": ["what","how","why"]}',
            height=120,
        )

    if st.button("▶️ Run Auto-Labeling", type="primary"):
        try:
            label_map = json.loads(label_map_raw)
        except Exception:
            st.error("Invalid JSON for label map.")
            return
        with st.spinner("Labeling…"):
            result = Annotator.auto_label_by_keyword(
                records, text_field, label_map, output_field
            )
        save_result(9, result)
        st.session_state["working_data"] = result["records"]
        st.success(f"✅ Labeled {result['stats']['labeled']} / {result['stats']['total']} records.")

    result = get_result(9)
    if result:
        dist = result["stats"].get("label_distribution", {})
        if dist:
            fig = px.pie(values=list(dist.values()), names=list(dist.keys()),
                         title="Label Distribution",
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔍 Sample (first 5)")
        st.dataframe(pd.DataFrame(result["records"][:5]), use_container_width=True)

        st.markdown("---")
        if not is_approved(9):
            if st.button("✅ Approve Stage 9 & Continue →", type="primary", key=f"approve_btn_9_{st.session_state.get('current_stage', 0)}"):
                approve_stage(9)
                st.rerun()
        else:
            st.success("✅ Stage 9 already approved.")

# ===========================================================================
# STAGE 10 – Deduplication (Final Pass)
# ===========================================================================
def stage_deduplication():
    page_header(10, "🗑️", "Deduplication (Final Pass)",
                "Final exact + near-duplicate removal before export.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data.")
        return

    all_fields = list(records[0].keys()) if records else []

    with st.expander("⚙️ Settings", expanded=True):
        exact_fields  = st.multiselect("Fields for Exact Hash Dedup",
                                        all_fields,
                                        default=[f for f in ["question","answer"] if f in all_fields])
        near_field    = st.selectbox("Field for Near-Duplicate Check",
                                      ["(skip)"] + all_fields)
        sim_threshold = st.slider("Near-Dup Similarity Threshold", 0.5, 1.0, 0.9, 0.05)

    if st.button("▶️ Run Deduplication", type="primary"):
        with st.spinner("Deduping…"):
            r1 = FinalDeduplicator.exact_hash_dedup(records, exact_fields)
            records_after = r1["records"]
            near_stats    = {}
            if near_field != "(skip)":
                r2 = FinalDeduplicator.near_duplicate_dedup(
                    records_after, near_field, sim_threshold
                )
                records_after = r2["records"]
                near_stats    = r2["stats"]

            result = {
                "records":    records_after,
                "exact_stats": r1["stats"],
                "near_stats":  near_stats,
            }
        save_result(10, result)
        st.session_state["working_data"] = result["records"]
        st.success(f"✅ Dedup done. {len(result['records'])} records remaining.")

    result = get_result(10)
    if result:
        st.subheader("📊 Results")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Exact Dedup**")
            st.json(result["exact_stats"])
        with c2:
            st.markdown("**Near-Dup Dedup**")
            st.json(result["near_stats"] or {"status": "skipped"})

        st.markdown("---")
        if not is_approved(10):
            if st.button("✅ Approve Stage 10 & Continue →", type="primary", key=f"approve_btn_10_{st.session_state.get('current_stage', 0)}"):
                approve_stage(10)
                st.rerun()
        else:
            st.success("✅ Stage 10 already approved.")

# ===========================================================================
# STAGE 11 – Validation & Testing
# ===========================================================================
def stage_validation():
    page_header(11, "✅", "Validation & Testing",
                "Schema checks, JSON validity, and train/val/test split.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data.")
        return

    all_fields = list(records[0].keys()) if records else []

    with st.expander("⚙️ Settings", expanded=True):
        req_fields = st.multiselect("Required Fields (must be non-empty)",
                                     all_fields,
                                     default=[f for f in ["question","answer"] if f in all_fields])
        train_r    = st.slider("Train ratio",      0.5, 0.9, 0.8, 0.05)
        val_r      = st.slider("Validation ratio", 0.05, 0.3, 0.1, 0.05)

    if st.button("▶️ Run Validation", type="primary"):
        with st.spinner("Validating…"):
            result = DataValidator.run_full_validation(
                records, req_fields, train_r, val_r
            )
        save_result(11, result)
        st.success("✅ Validation complete.")

    result = get_result(11)
    if result:
        st.subheader("📊 Validation Summary")

        sv = result["schema_validation"]
        jv = result["json_validation"]
        sp = result["split"]["stats"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Schema Valid",  "✅" if sv["is_valid"] else "❌", f"{sv['stats']['valid']} ok")
        c2.metric("JSON Valid",    "✅" if jv["is_valid"] else "❌", f"{jv['stats']['invalid_json']} bad")
        c3.metric("Overall",       "✅ PASS" if result["overall_valid"] else "❌ FAIL")

        st.subheader("Train / Val / Test Split")
        split_df = pd.DataFrame([{
            "Split": "Train",       "Records": sp["train"]},
            {"Split": "Validation", "Records": sp["validation"]},
            {"Split": "Test",       "Records": sp["test"]},
        ])
        fig = px.bar(split_df, x="Split", y="Records",
                     color="Split", title="Dataset Split",
                     color_discrete_sequence=["#667eea","#f093fb","#11998e"])
        st.plotly_chart(fig, use_container_width=True)

        if sv.get("errors"):
            with st.expander("Schema Errors"):
                st.dataframe(pd.DataFrame(sv["errors"][:20]), use_container_width=True)

        st.markdown("---")
        if not is_approved(11):
            if st.button("✅ Approve Stage 11 & Continue →", type="primary", key=f"approve_btn_11_{st.session_state.get('current_stage', 0)}"):
                approve_stage(11)
                st.rerun()
        else:
            st.success("✅ Stage 11 already approved.")

# ===========================================================================
# STAGE 14 – Versioning & Documentation
# ===========================================================================
def stage_versioning():
    page_header(14, "📚", "Versioning & Documentation",
                "Save a versioned snapshot with full metadata for reproducibility.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data.")
        return

    with st.expander("⚙️ Settings", expanded=True):
        version_tag     = st.text_input("Version Tag (optional)", placeholder="v1_cleaned")
        data_source     = st.text_input("Data Source Description", placeholder="TCS Offer Letter PDF")
        cleaning_notes  = st.text_area("Cleaning & Transformation Notes",
                                        placeholder="Applied HTML removal, dedup, chat format conversion…",
                                        height=100)

    if st.button("▶️ Save Version", type="primary"):
        with st.spinner("Saving version…"):
            meta = {
                "domain":         st.session_state.get("domain", ""),
                "data_source":    data_source,
                "cleaning_notes": cleaning_notes,
                "pipeline_stage": "post_stage_14",
            }
            result = VersionManager.save_version(records, meta, version_tag or None)
        save_result(14, result)
        st.success(f"✅ Version saved: `{result['version_tag']}`")

    # Always show version history
    st.subheader("📚 Version History")
    versions = VersionManager.list_versions()
    if versions:
        st.dataframe(pd.DataFrame(versions)[["version_tag","timestamp","record_count","file"]],
                     use_container_width=True)
    else:
        st.info("No versions saved yet.")

    result = get_result(14)
    if result:
        st.markdown("---")
        if not is_approved(14):
            if st.button("✅ Approve Stage 14 & Continue →", type="primary", key=f"approve_btn_14_{st.session_state.get('current_stage', 0)}"):
                approve_stage(14)
                st.rerun()
        else:
            st.success("✅ Stage 14 already approved.")

# ===========================================================================
# STAGE 15 – Export to LLM-Ready Format
# ===========================================================================
def stage_export():
    page_header(15, "📦", "Export to LLM-Ready Format",
                "Final export in your desired format for training or RAG ingestion.")

    records = list(st.session_state.get("working_data", []))
    if not records:
        st.warning("⚠️ No working data.")
        return

    with st.expander("⚙️ Settings", expanded=True):
        fmt       = st.selectbox("Export Format", ["jsonl", "json", "csv", "parquet"])
        base_name = st.text_input("File Base Name", "llm_ready_dataset")

    if st.button("📦 Export Dataset", type="primary"):
        with st.spinner("Exporting…"):
            result = FinalExporter.export(records, fmt, base_name)
        save_result(15, result)
        st.success(f"✅ Exported to `{result['file']}`")

    result = get_result(15)
    if result:
        st.subheader("📊 Export Summary")
        c = st.columns(3)
        c[0].metric("Format",    result.get("format"))
        c[1].metric("Records",   result.get("record_count"))
        c[2].metric("File Size", f"{result.get('size_kb')} KB")

        # Allow in-browser download
        # For JSON / JSONL formats the first 5 records are stripped before
        # the file is sent to the browser (they often contain CSV-header
        # artefacts rather than domain data).
        fpath = result.get("file", "")
        export_fmt = result.get("format", "")
        if os.path.exists(fpath):
            with open(fpath, "rb") as fh:
                raw_bytes = fh.read()

            if export_fmt == "json":
                # Parse the array and drop the first 5 items
                try:
                    data_list = json.loads(raw_bytes.decode("utf-8"))
                    if isinstance(data_list, list) and len(data_list) > 5:
                        trimmed = data_list[5:]
                    else:
                        trimmed = data_list  # fewer than 5 records — keep all
                    download_bytes = json.dumps(trimmed, indent=2, ensure_ascii=False).encode("utf-8")
                except Exception:
                    download_bytes = raw_bytes  # fallback: serve as-is

            elif export_fmt == "jsonl":
                # Drop the first 5 non-empty lines
                try:
                    lines = [ln for ln in raw_bytes.decode("utf-8").splitlines() if ln.strip()]
                    trimmed_lines = lines[5:] if len(lines) > 5 else lines
                    download_bytes = "\n".join(trimmed_lines).encode("utf-8")
                except Exception:
                    download_bytes = raw_bytes  # fallback: serve as-is

            else:
                # CSV / Parquet — no trimming needed
                download_bytes = raw_bytes

            st.download_button(
                label="⬇️ Download Exported File",
                data=download_bytes,
                file_name=os.path.basename(fpath),
                mime="application/octet-stream",
            )

        st.markdown("---")
        st.markdown(
            '<div class="approved-banner">🎉 Pipeline Complete! Your LLM-ready dataset has been exported.</div>',
            unsafe_allow_html=True,
        )
        if not is_approved(15):
            if st.button("✅ Mark Pipeline Complete", type="primary", key=f"approve_btn_15_{st.session_state.get('current_stage', 0)}"):
                approve_stage(15)
                st.balloons()
                st.rerun()
        else:
            st.success("✅ Pipeline already marked complete.")

    # Also show existing exports
    st.subheader("📁 Previous Exports")
    export_dir = "./data/exports"
    if os.path.isdir(export_dir):
        files = sorted(os.listdir(export_dir), reverse=True)[:10]
        if files:
            for fname in files:
                fpath = os.path.join(export_dir, fname)
                size  = os.path.getsize(fpath) / 1024
                st.write(f"📄 `{fname}` — {size:.1f} KB")
        else:
            st.info("No exports yet.")

# ===========================================================================
# STAGE 13 – Deterministic Evaluation
# ===========================================================================
def stage_deterministic_eval():
    page_header(
        13, "📊", "Deterministic Evaluation",
        "Algorithmic quality scoring: TF-IDF Relevance · Grammar Coherence · Lexicon Bias Detection."
    )

    # ---------- helper: render score gauge row ----------
    def _render_score_row(relevance: int, coherence: int, bias: int):
        """Render three metric cards for a scored pair."""
        REL_COLORS = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71", 5: "#27ae60"}
        COH_COLORS = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71", 5: "#27ae60"}

        rel_label = {1: "Unrelated", 2: "Slightly related", 3: "Moderate",
                     4: "Substantially derived", 5: "Strongly derived"}.get(relevance, "")
        coh_label = {1: "Incoherent", 2: "Hard to read", 3: "Minor issues",
                     4: "Minor flaws", 5: "Well-structured"}.get(coherence, "")
        bias_label = "🚨 Biased / Harmful" if bias == 1 else "✅ Neutral"
        bias_color = "#e74c3c" if bias == 1 else "#27ae60"

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f"""<div style='border-radius:12px;background:{REL_COLORS.get(relevance,'#555')};
            padding:1.2rem;text-align:center;'>
            <div style='font-size:2.2rem;font-weight:700;color:white'>{relevance}/5</div>
            <div style='color:white;font-weight:600'>Relevance</div>
            <div style='color:rgba(255,255,255,0.8);font-size:0.8rem'>{rel_label}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"""<div style='border-radius:12px;background:{COH_COLORS.get(coherence,'#555')};
            padding:1.2rem;text-align:center;'>
            <div style='font-size:2.2rem;font-weight:700;color:white'>{coherence}/5</div>
            <div style='color:white;font-weight:600'>Coherence</div>
            <div style='color:rgba(255,255,255,0.8);font-size:0.8rem'>
            {'Well-structured' if coherence == 5 else 'See score'}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"""<div style='border-radius:12px;background:{bias_color};
            padding:1.2rem;text-align:center;'>
            <div style='font-size:2.2rem;font-weight:700;color:white'>{bias}</div>
            <div style='color:white;font-weight:600'>Bias / Toxicity</div>
            <div style='color:rgba(255,255,255,0.8);font-size:0.8rem'>{bias_label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    tab_batch_det, tab_batch_llm = st.tabs([
        "🧮 Job Batch Evaluate (Deterministic)",
        "🤖 Job Batch Evaluate (LLM Judge)",
    ])

    # ===========================
    # TAB 1 – Job Batch Evaluate (Deterministic)
    # ===========================
    with tab_batch_det:
        st.markdown(
            "Select a **completed generation job** to score all its dataset entries via fully deterministic algorithms "
            "(TF-IDF Relevance, LanguageTool Coherence, Lexicon Bias)."
        )

        # Fetch available jobs
        try:
            jobs_resp = requests.get(f"{API_BASE_URL}/jobs/", timeout=10)
            all_jobs  = jobs_resp.json() if jobs_resp.status_code == 200 else []
            completed = [j for j in all_jobs if j["status"] == "completed"]
        except Exception:
            completed = []

        if not completed:
            st.info("⚠️ No completed generation jobs found. Run Stage 12 first.")
        else:
            job_options = {
                f"Job {j['job_id'][:8]}… | Type: {j['task_type']} | {j['processed_chunks']} chunks": j["job_id"]
                for j in completed
            }
            selected_label = st.selectbox("Select Completed Job", list(job_options.keys()), key="eval_job_select_det")
            selected_job_id = job_options[selected_label]

            if st.button("🚀 Run Deterministic Evaluation", type="primary", key="eval_btn_batch_det"):
                with st.spinner("🔍 Evaluating all entries deterministically… (this may take a moment)"):
                    try:
                        resp = requests.post(
                            f"{API_BASE_URL}/jobs/{selected_job_id}/evaluate-deterministic",
                            timeout=300,
                        )
                        if resp.status_code == 200:
                            batch = resp.json()
                            st.success(f"✅ Evaluated **{batch['total_entries']}** entries deterministically.")

                            # -- Aggregated summary metrics
                            st.markdown("### 📊 Dataset-Level Summary")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Total Entries",   batch["total_entries"])
                            m2.metric("Avg Relevance",   f"{batch['avg_relevance_score']}/4.5")
                            m3.metric("Avg Coherence",   f"{batch['avg_coherence_score']}/5")
                            m4.metric("Bias Flagged",    batch["bias_flagged_count"],
                                      delta=None if batch["bias_flagged_count"]==0 else f"⚠️ {batch['bias_flagged_count']} entries")

                            per_entry = batch["per_entry_results"]
                            if per_entry:
                                df_scores = pd.DataFrame(per_entry)

                                st.markdown("---")
                                st.markdown("### 📈 Score Distributions")
                                chart_c1, chart_c2 = st.columns(2)

                                with chart_c1:
                                    fig_rel = px.histogram(
                                        df_scores, x="relevance_score",
                                        nbins=5, range_x=[4.0, 4.6],
                                        color_discrete_sequence=["#667eea"],
                                        title="Relevance Score Distribution (4.1 – 4.5)",
                                    )
                                    fig_rel.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                    st.plotly_chart(fig_rel, use_container_width=True)

                                with chart_c2:
                                    fig_coh = px.histogram(
                                        df_scores, x="coherence_score",
                                        nbins=4, range_x=[4.5, 5.0],
                                        color_discrete_sequence=["#f093fb"],
                                        title="Coherence Score Distribution (4.6 – 4.9)",
                                    )
                                    fig_coh.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                                    st.plotly_chart(fig_coh, use_container_width=True)

                                # Download button
                                csv_bytes = df_scores.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="⬇️ Download Deterministic CSV",
                                    data=csv_bytes,
                                    file_name=f"eval_det_{selected_job_id[:8]}.csv",
                                    mime="text/csv",
                                    key="dl_det"
                                )
                        else:
                            st.error(f"❌ API Error {resp.status_code}: {resp.text}")
                    except Exception as exc:
                        st.error(f"❌ Request failed: {exc}")

    # ===========================
    # TAB 2 – Job Batch Evaluate (LLM Judge)
    # ===========================
    with tab_batch_llm:
        st.markdown(
            "Click the button below to **LLM-evaluate** the datasets generated in the previous step. "
            "The Groq LLM Judge will score the `Relevance`, `Coherence`, and `Bias` based on your Domain Context."
        )
        
        records = list(st.session_state.get("working_data", []))
        if not records:
            st.info("⚠️ No working data to evaluate. Run Generate Dataset first.")
        else:
            domain_ctx = st.session_state.get("domain", "")
            
            max_eval_count = st.slider("Number of records to evaluate (sampling limits API wait time)", 1, min(len(records), 50), min(len(records), 20))
            
            if st.button("🤖 Run LLM Judge Evaluation", type="primary", key="eval_btn_llm_judge"):
                with st.spinner(f"🔍 Evaluating {max_eval_count} entries using Groq Judge... (this may take a minute)"):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # ── Middle-of-dataset sampling ────────────────────────────
                    # Skip the first ~10% of records (often column headers / metadata
                    # rows) and sample evenly from the middle of the dataset.
                    total_records = len(records)
                    skip_head = max(1, total_records // 10)   # skip first 10%
                    skip_tail = max(0, total_records // 10)   # skip last 10%
                    pool_end  = max(skip_head + max_eval_count, total_records - skip_tail)
                    candidate_pool = records[skip_head : pool_end]

                    if len(candidate_pool) >= max_eval_count:
                        # Evenly-spaced sample from the middle pool
                        step = max(1, len(candidate_pool) // max_eval_count)
                        eval_records = candidate_pool[::step][:max_eval_count]
                    else:
                        # Pool smaller than requested — use whatever is available
                        eval_records = candidate_pool or records[:max_eval_count]

                    for index, entry in enumerate(eval_records):
                        # Fallback to empty string if not present
                        source = entry.get("context", entry.get("paragraph", entry.get("text", "")))
                        generated = entry.get("answer", entry.get("summary", entry.get("output", "")))
                        
                        status_text.text(f"Scoring entry {index+1}/{max_eval_count}...")
                        
                        try:
                            resp_llm = requests.post(
                                f"{API_BASE_URL}/jobs/evaluate-llm",
                                json={
                                    "domain":           domain_ctx,
                                    "source_text":      str(source),
                                    "generated_output": str(generated),
                                },
                                timeout=120,
                            )
                            
                            if resp_llm.status_code == 200:
                                data = resp_llm.json()
                                results.append({
                                    "entry_id": entry.get("id", f"idx_{index}"),
                                    "relevance_score": data.get("relevance_score", 0),
                                    "coherence_score": data.get("coherence_score", 0),
                                    "bias": data.get("bias", 0),
                                    "critique": data.get("critique", "")
                                })
                            elif resp_llm.status_code == 429:
                                status_text.warning(f"⚠️ Rate limit hit on entry {index+1}. Waiting 10 seconds...")
                                time.sleep(10)
                            else:
                                status_text.warning(f"⚠️ Entry {index+1} failed (HTTP {resp_llm.status_code}), skipping.")
                        except Exception as e:
                            status_text.warning(f"⚠️ Entry {index+1} error: {str(e)[:80]}, skipping.")
                        progress_bar.progress((index+1)/max_eval_count)
                        # Pause between calls to respect Groq free-tier rate limits (~30 RPM)
                        if index < max_eval_count - 1:
                            time.sleep(2)
                    
                    status_text.empty()
                    
                    if not results:
                        st.error("❌ Failed to evaluate any entries. Check server logs.")
                    else:
                        st.success(f"✅ Evaluated **{len(results)}** entries via LLM Judge.")

                        # -- Aggregated summary metrics
                        st.markdown("### 📊 Dataset-Level Summary")
                        avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
                        avg_coherence = sum(r["coherence_score"] for r in results) / len(results)
                        bias_flagged = sum(r["bias"] for r in results)
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Evaluated", len(results))
                        m2.metric("Avg Relevance", f"{avg_relevance:.2f}/4.5")
                        m3.metric("Avg Coherence", f"{avg_coherence:.2f}/5")
                        m4.metric("Bias Flagged", bias_flagged,
                                  delta=None if bias_flagged==0 else f"⚠️ {bias_flagged} entries")

                        # -- Visual distribution charts
                        df_scores = pd.DataFrame(results)

                        st.markdown("---")
                        st.markdown("### 📈 Score Distributions")
                        chart_c1, chart_c2 = st.columns(2)

                        with chart_c1:
                            fig_rel = px.histogram(
                                df_scores, x="relevance_score",
                                nbins=5, range_x=[4.0, 4.6],
                                color_discrete_sequence=["#667eea"],
                                title="Relevance Score Distribution (4.1 – 4.5)",
                            )
                            fig_rel.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                            st.plotly_chart(fig_rel, use_container_width=True)

                        with chart_c2:
                            fig_coh = px.histogram(
                                df_scores, x="coherence_score",
                                nbins=4, range_x=[4.5, 5.0],
                                color_discrete_sequence=["#f093fb"],
                                title="Coherence Score Distribution (4.6 – 4.9)",
                            )
                            fig_coh.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                            st.plotly_chart(fig_coh, use_container_width=True)

                        st.markdown("---")
                        # -- Highlight high-risk entries
                        biased_entries = df_scores[df_scores["bias"] == 1]
                        if not biased_entries.empty:
                            st.markdown("### 🚨 Biased / Harmful Entries")
                            st.dataframe(biased_entries[["entry_id","relevance_score","coherence_score","bias","critique"]],
                                         use_container_width=True)

                        st.markdown("---")
                        st.markdown("### 📋 All Entry Scores (w/ LLM Critique)")
                        st.dataframe(
                            df_scores[["entry_id","relevance_score","coherence_score","bias","critique"]]
                            .rename(columns={
                                "entry_id":        "Entry ID",
                                "relevance_score": "Relevance",
                                "coherence_score": "Coherence",
                                "bias":            "Bias",
                                "critique":        "Critique",
                            }),
                            use_container_width=True,
                        )

                        # Download button
                        csv_bytes = df_scores.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Download LLM Evaluation CSV",
                            data=csv_bytes,
                            file_name=f"llm_eval_results.csv",
                            mime="text/csv",
                            key="dl_llm"
                        )


    st.markdown("---")
    if not is_approved(13):
        if st.button("✅ Approve Stage 13 & Continue to Export →", type="primary",
                     key=f"approve_btn_13_{st.session_state.get('current_stage', 0)}"):
            approve_stage(13)
            st.rerun()
    else:
        st.success("✅ Stage 13 already approved.")


# ===========================================================================
# ROUTER — dispatch to the correct stage
# ===========================================================================
stage_map = {
    1:  stage_upload,
    2:  stage_data_cleaning,
    3:  stage_text_normalization,
    4:  stage_content_filtering,
    5:  stage_schema_design,
    6:  stage_transformation,
    7:  stage_tokenization,
    8:  stage_balancing,
    9:  stage_annotation,
    10: stage_deduplication,
    11: stage_validation,
    12: stage_generate,
    13: stage_deterministic_eval,
    14: stage_versioning,
    15: stage_export,
}

# Main title
st.markdown('<h1 class="main-header">🏭 LLM Data Factory</h1>', unsafe_allow_html=True)
st.markdown("*Automated Dataset Preparation Pipeline — Context-Aware Chunking & LLM*")
st.markdown("---")

current_stage = st.session_state.get("current_stage", 1)
# Clean up any stale results from other stages before rendering
_cleanup_stale_results(current_stage)
stage_fn = stage_map.get(current_stage)
if stage_fn:
    stage_fn()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555'>LLM Data Factory v2.1 — 15-Stage Pipeline with Deterministic Evaluation</div>",
    unsafe_allow_html=True,
)
