# utils/eval_utils.py
"""
Evaluation helpers.

- evaluate_response_detailed(pred, gold) -> token-overlap metrics
- evaluate_with_llm(llm_pipeline, answer, context) -> parse JSON output from evaluator LLM
"""

import re
import json
from typing import Dict, Any, Optional

# relative imports from utils package
from .text_utils import tokenize_words, clean_model_output

# Prompt template for LLM-based evaluation (kept from your original script)
PROMPT_TEMPLATE_EVAL = """You are an expert evaluator. Using ONLY the CONTEXT below, judge whether the claims made in the MODEL_ANSWER are supported by the CONTEXT.

Return ONLY a valid JSON object wrapped inside a markdown json code block (i.e. ```json ... ```). The JSON must contain the keys:
 - "precision": float (0.0-1.0)
 - "recall": float (0.0-1.0)
 - "f1": float
 - "supported_claims": int
 - "total_claims": int
 - "missing_details": list[str]
 - "rationale": str

IMPORTANT: Do NOT output any prose outside the ```json``` block. If you cannot compute a metric, use null for that value.

CONTEXT:
{context}

MODEL_ANSWER:
{answer}

OUTPUT:
"""

def evaluate_response_detailed(pred: str, gold: str) -> Dict[str, Any]:
    """
    Simple token-overlap metrics (fallback).
    Returns TP/FP/FN counts and precision/recall/f1 floats.
    """
    pred_tokens = set(tokenize_words(pred or ""))
    gold_tokens = set(tokenize_words(gold or ""))
    tp = len(pred_tokens & gold_tokens)
    fp = len(pred_tokens - gold_tokens)
    fn = len(gold_tokens - pred_tokens)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def _try_parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Try multiple heuristics to extract a JSON object from LLM text."""
    if not text:
        return None

    # 1) Look for a fenced JSON block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        js = m.group(1)
        try:
            return json.loads(js)
        except Exception:
            pass

    # 2) Extract first {...} block
    m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m2:
        js = m2.group(1)
        try:
            return json.loads(js)
        except Exception:
            # attempt minor fixes
            fixed = js.replace("'", '"')
            fixed = re.sub(r",\s*\}", "}", fixed)
            fixed = re.sub(r",\s*\]", "]", fixed)
            try:
                return json.loads(fixed)
            except Exception:
                pass

    # 3) Try to parse numeric fields heuristically
    numeric = {}
    m_prec = re.search(r"precision\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    m_rec = re.search(r"recall\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    m_f1 = re.search(r"f1\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    m_sup = re.search(r"supported[_\s]claims\s*[:=]\s*([0-9]+)", text, flags=re.IGNORECASE)
    m_tot = re.search(r"total[_\s]claims\s*[:=]\s*([0-9]+)", text, flags=re.IGNORECASE)

    if m_prec:
        try: numeric['precision'] = float(m_prec.group(1))
        except: pass
    if m_rec:
        try: numeric['recall'] = float(m_rec.group(1))
        except: pass
    if m_f1:
        try: numeric['f1'] = float(m_f1.group(1))
        except: pass
    if m_sup:
        try: numeric['supported_claims'] = int(m_sup.group(1))
        except: pass
    if m_tot:
        try: numeric['total_claims'] = int(m_tot.group(1))
        except: pass

    if numeric:
        # fill defaults for missing keys
        result = {
            'precision': numeric.get('precision'),
            'recall': numeric.get('recall'),
            'f1': numeric.get('f1'),
            'supported_claims': numeric.get('supported_claims'),
            'total_claims': numeric.get('total_claims'),
            'missing_details': [],
            'rationale': None
        }
        return result

    return None

def evaluate_with_llm(llm_pipe, answer: str, context: str) -> Dict[str, Any]:
    """
    Use the supplied LLM pipeline (HuggingFace pipeline or similar) to evaluate the answer.
    llm_pipe should be callable like: llm_pipe(prompt, num_return_sequences=1, max_length=...)
    Returns parsed JSON-like dict if possible, otherwise returns {'parsing_failed': True, 'raw': raw_text}
    """
    prompt = PROMPT_TEMPLATE_EVAL.format(context=context, answer=answer)
    try:
        outputs = llm_pipe(prompt, num_return_sequences=1, max_length=512)
        if isinstance(outputs, list) and outputs:
            raw = outputs[0].get("generated_text", "") if isinstance(outputs[0], dict) else str(outputs[0])
        elif isinstance(outputs, dict):
            raw = outputs.get("generated_text", "")
        else:
            raw = str(outputs)
        parsed = _try_parse_json_from_text(raw)
        if parsed:
            # ensure floats where applicable
            for k in ["precision", "recall", "f1"]:
                if k in parsed:
                    try:
                        parsed[k] = float(parsed[k]) if parsed[k] is not None else None
                    except Exception:
                        parsed[k] = None
            parsed["raw"] = raw
            return parsed
        else:
            return {"parsing_failed": True, "raw": raw}
    except Exception as e:
        return {"error": str(e)}
