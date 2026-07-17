"""
Jinja2 prompt templates for various generation tasks.
Supports few-shot examples to guide LLM output style and format.
"""

from jinja2 import Template
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# QA Generation Template
# ─────────────────────────────────────────────────────────────────────────────
QA_GENERATION_TEMPLATE = Template("""\
You are an expert dataset creator for AI model training.

Your input below (SOURCE DATA) may be in ANY format — raw CSV, tabular rows, hospital records,
food recipes, medical data, financial tables, or plain text. It does NOT matter.
You MUST read and understand it, extract the meaningful information, and produce
high-quality English Question-Answer pairs.

Generate exactly {{ num_pairs }} question-answer pairs.

━━━ ABSOLUTE RULES ━━━
1. "question" — A complete, natural English question ending with "?".
   - Must sound like something a curious, intelligent person would ask.
   - Must be specific to facts in the source data.
   - NEVER copy raw tokens like "above", "nan", "same", "lower", or numeric codes.
   - NEVER start with a number, code, abbreviation, or raw data value.

2. "answer" — A complete, fluent English answer in full prose sentences.
   - Must directly and fully answer its question.
   - Must be self-contained — readable without seeing the original data.
   - Must be at least 20 words.
   - NEVER contain raw data tokens, CSV values, codes, or abbreviations.
   - If the source is tabular data about hospitals, ratings, scores, etc.,
     INTERPRET and EXPLAIN the values in plain English.

3. Each question must focus on a DIFFERENT fact or aspect.
4. Output ONLY a valid JSON array. No markdown, no explanation, no extra text.
5. Even if the input looks like garbled data, you MUST produce clean English output.
{% if examples %}

━━━ EXAMPLES ━━━
{% for ex in examples %}
Example {{ loop.index }}:
  Question: {{ ex.question }}
  Answer:   {{ ex.answer }}
{% endfor %}
{% endif %}

━━━ SOURCE DATA ━━━
{{ context }}

━━━ OUTPUT (JSON array only) ━━━
[
  {
    "question": "<a specific, natural English question ending in ?>",
    "answer": "<a complete, well-written answer in full prose sentences>"
  }
]

Respond with ONLY the JSON array starting with [ and ending with ]. No other text.""")


# ─────────────────────────────────────────────────────────────────────────────
# Summarization Template  (Question / Context / Answer format)
# Context is an INTERPRETED prose paragraph — NOT a verbatim copy
# ─────────────────────────────────────────────────────────────────────────────
SUMMARIZATION_TEMPLATE = Template("""\
You are an expert dataset creator for AI model training.

Your input below (SOURCE DATA) may be in ANY format — raw CSV, tabular rows, hospital records,
food recipes, medical data, financial tables, or plain text. It does NOT matter.
You MUST read and understand it, extract the meaningful information, and produce
high-quality English Summarization training records.

Generate exactly {{ num_pairs }} summarization records.

━━━ ABSOLUTE RULES ━━━
1. Every record MUST contain exactly three fields: "question", "context", "answer".

2. "question" — A natural, specific English question asking for a summary of the information.
   - Must end with "?".
   - Must be specific to the data (name entities, topics, or aspects directly).
   - NEVER use raw data tokens, codes, or abbreviations.
   Examples:
     "What are the key performance metrics of Sequoia Hospital in Redwood City?"
     "Can you summarise the preparation method and ingredients for Hyderabad Soy Biryani?"

3. "context" — A rich, well-written PROSE PARAGRAPH of at least 80 words that YOU write
   by interpreting and organising the key facts from SOURCE DATA into readable English.
   - Write in clear, fluent English sentences.
   - If the source is tabular/CSV data, TRANSLATE it into a human-readable narrative.
   - Include names, numbers, comparisons, and key details in natural language.
   - NEVER copy raw data tokens like "above", "nan", "same", "lower", "average".
   - Must read like a news article or encyclopedia paragraph.

4. "answer" — A concise 2-4 sentence summary of the context paragraph.
   - Fully grounded in the context (no hallucinations).
   - Written in fluent English prose.

5. Cover different aspects across each record.
6. Output ONLY a valid JSON array. No markdown, no explanation, no extra text.
7. Even if the input looks like garbled data, you MUST produce clean English output.
{% if examples %}

━━━ EXAMPLES ━━━
{% for ex in examples %}
Example {{ loop.index }}:
  Question: {{ ex.question }}
  Context:  {{ ex.paragraph[:300] }}...
  Answer:   {{ ex.summary }}
{% endfor %}
{% endif %}

━━━ SOURCE DATA ━━━
{{ context }}

━━━ OUTPUT (JSON array only) ━━━
[
  {
    "question": "<a specific summarisation question ending in ?>",
    "context":  "<a clean prose paragraph of 80+ words interpreting the key facts>",
    "answer":   "<a concise 2-4 sentence summary of the context>"
  }
]

Respond with ONLY the JSON array starting with [ and ending with ]. No other text.""")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Task Template
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_TASK_TEMPLATE = Template("""\
{{ instruction }}
{% if examples %}

━━━ EXAMPLES (follow this exact style) ━━━
{% for ex in examples %}
Example {{ loop.index }}:
  Input: {{ ex.input }}
  Output: {{ ex.output }}
{% endfor %}
{% endif %}

━━━ CONTEXT ━━━
{{ context }}

━━━ OUTPUT ━━━""")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Template
# ─────────────────────────────────────────────────────────────────────────────
EVALUATION_TEMPLATE = Template("""\
Evaluate if the following answer is accurate and complete based on the context.

**Context:**
{{ context }}

**Question:**
{{ question }}

**Answer:**
{{ answer }}

**Evaluation:**
Rate the answer on a scale of 1-10 for:
1. Accuracy (is it factually correct?)
2. Completeness (does it fully answer the question?)
3. Relevance (is it on-topic?)

Output format:
{
  "accuracy": <score>,
  "completeness": <score>,
  "relevance": <score>,
  "overall": <average_score>
}

JSON Output:""")


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def get_qa_prompt(
    context: str,
    num_pairs: int = 3,
    few_shot_examples: Optional[List[Dict]] = None
) -> str:
    """Generate QA generation prompt with optional few-shot examples."""
    return QA_GENERATION_TEMPLATE.render(
        context=context,
        num_pairs=num_pairs,
        examples=few_shot_examples or []
    )


def get_summary_prompt(
    context: str,
    num_pairs: int = 3,
    few_shot_examples: Optional[List[Dict]] = None
) -> str:
    """Generate summarization prompt — outputs Question/Context/Answer JSON array."""
    return SUMMARIZATION_TEMPLATE.render(
        context=context,
        num_pairs=num_pairs,
        examples=few_shot_examples or []
    )



def get_custom_prompt(
    context: str,
    instruction: str,
    few_shot_examples: Optional[List[Dict]] = None
) -> str:
    """Generate custom task prompt with optional few-shot examples."""
    return CUSTOM_TASK_TEMPLATE.render(
        context=context,
        instruction=instruction,
        examples=few_shot_examples or []
    )


def get_evaluation_prompt(context: str, question: str, answer: str) -> str:
    """Generate evaluation prompt."""
    return EVALUATION_TEMPLATE.render(
        context=context,
        question=question,
        answer=answer
    )
