"""
Data formatting utilities for FinQA pipelines.

Converts raw FinQA samples (with pre_text, table, post_text) into
prompt-ready strings that can be fed to LLMs.
"""

from typing import List, Dict, Any


def table_to_markdown(table: List[List[str]]) -> str:
    """Convert a 2D table to a markdown-style string.
    
    Args:
        table: 2D list, e.g. [["", "2018", "2017"], ["Revenue", "100", "90"]]
    
    Returns:
        Markdown-formatted string with rows separated by newlines.
    """
    if not table or not table[0]:
        return ""
    lines = []
    for row in table:
        # Clean each cell: strip whitespace, replace empty strings
        cells = [str(cell).strip() if cell else "-" for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_context(sample: Dict[str, Any]) -> str:
    """Build a single context string from a FinQA sample.
    
    Concatenates pre_text + table (markdown) + post_text into one document.
    """
    pre = " ".join(sample.get('pre_text', []))
    table_md = table_to_markdown(sample.get('table', []))
    post = " ".join(sample.get('post_text', []))
    
    parts = []
    if pre:
        parts.append(pre)
    if table_md:
        parts.append("Table:\n" + table_md)
    if post:
        parts.append(post)
    
    return "\n\n".join(parts)


def build_prompt(sample: Dict[str, Any], 
                 max_context_words: int = 350) -> str:
    """Build a zero-shot prompt for Flan-T5.
    
    Truncates context to max_context_words to fit Flan-T5's 512-token limit.
    (~350 words leaves room for question + prompt template + answer.)
    
    Args:
        sample: FinQA sample dict
        max_context_words: word-level truncation limit
    
    Returns:
        Complete prompt string ready for tokenization.
    """
    context = build_context(sample)
    
    # Word-level truncation (rough but safe)
    words = context.split()
    if len(words) > max_context_words:
        context = " ".join(words[:max_context_words]) + " [...]"
    
    question = sample.get('qa', {}).get('question', '')
    
    prompt = (
        "Read the following financial document and answer the question. "
        "Give a short, direct answer (a number, percentage, or yes/no).\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt


def get_gold_answer(sample: Dict[str, Any]) -> str:
    """Extract the ground-truth answer string."""
    return str(sample.get('qa', {}).get('answer', '')).strip()
