# utils/text_utils.py
import re
import nltk

def safe_normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_words(text: str) -> list:
    return safe_normalize_text(text).split()

def remove_repeated_sentences(text: str) -> str:
    if not text:
        return ""
    try:
        sentences = nltk.tokenize.sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[\.\?\!])\s+", text)
    seen = set(); out = []
    for s in sentences:
        s_norm = s.strip()
        if s_norm and s_norm not in seen:
            out.append(s_norm); seen.add(s_norm)
    return ". ".join(out).strip()

def collapse_adjacent_duplicate_phrases(text: str) -> str:
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s{2,}", " ", text)
    pattern = re.compile(r"(.{10,200}?)\s+\1", flags=re.IGNORECASE|re.DOTALL)
    for _ in range(3):
        text = pattern.sub(r"\1", text)
    return text.strip()

def clean_model_output(text: str) -> str:
    text = text or ""
    text = remove_repeated_sentences(text)
    text = collapse_adjacent_duplicate_phrases(text)
    return text.strip()
