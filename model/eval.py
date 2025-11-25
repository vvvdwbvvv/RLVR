import json
import random
import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_ehr_eval_examples(path="data/ehr_structured.jsonl", limit=None):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            soap = record.get("soap") or {}
            s = (soap.get("S") or "").strip()
            o = (soap.get("O") or "").strip()
            if not (s or o):
                continue
            example = {
                "id": record.get("id"),
                "input": f"Subjective:\n{s}\n\nObjective:\n{o}",
                "label": (soap.get("A") or "").strip(),
            }
            examples.append(example)
            if limit and len(examples) >= limit:
                break
    return examples


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def get_tokens(text: str) -> List[str]:
    return normalize_text(text).split()


def token_f1(pred: str, gold: str) -> float:
    """
    超簡單 token-level F1:
    - 以空白切詞
    - 用多重集(multiset)交集算 TP
    """
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0

    from collections import Counter

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    # true positives = 交集裡每個 token 出現次數的 min
    tp = sum((pred_counter & gold_counter).values())
    fp = len(pred_tokens) - tp
    fn = len(gold_tokens) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def recall_precision_f1_at_k(pred: str, gold: str, k: int):
    pred_tokens = get_tokens(pred)[:k]
    gold_tokens = get_tokens(gold)
    if not gold_tokens:
        if not pred_tokens:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    from collections import Counter

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    tp = sum((pred_counter & gold_counter).values())
    fp = len(pred_tokens) - tp
    fn = len(gold_tokens) - tp

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return rec, prec, f1


HARD_NEGATIVE_PHRASES = {
    "hard negative",
    "hard negative example",
    "irrelevant note",
    "negative sample",
}


def contains_hard_negative(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in HARD_NEGATIVE_PHRASES)


def balanced_accuracy(pred: str) -> float:
    detection = contains_hard_negative(pred)
    specificity = 1.0 if not detection else 0.0
    return specificity


def pass_at_k(candidates: List[str], gold: str, k: int) -> float:
    target = normalize_text(gold)
    hits = sum(
        1 for candidate in candidates[:k] if normalize_text(candidate) == target
    )
    return 1.0 if hits > 0 else 0.0


ASSESSMENT_PATTERN = re.compile(
    r"<assessment>(.*?)</assessment>",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_assessment(text: str) -> str:
    match = ASSESSMENT_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def build_ehr_fewshot_prompt(shots, ex, max_shots=None):
    """
    shots: list of {input, label}
    ex:    單一 eval example
    max_shots: 可用來 hard clip shots
    """
    if max_shots is not None:
        shots = shots[:max_shots]

    header = (
        "You are a clinical assistant.\n"
        "Given the Subjective (S) and Objective (O) parts of a SOAP note,\n"
        "write the Assessment (A) in concise clinical language.\n"
        "Wrap your assessment between <assessment> and </assessment> tags.\n\n"
    )

    fewshot_parts = []
    for i, e in enumerate(shots, start=1):
        part = (
            f"Example {i}:\n"
            f"Input:\n{e['input']}\n\n"
            f"Assessment:\n<assessment>{e['label']}</assessment>\n\n"
        )
        fewshot_parts.append(part)

    test_part = (
        f"Now answer the next question.\n\nInput:\n{ex['input']}\n\nAssessment:\n"
    )

    return header + "".join(fewshot_parts) + test_part


@torch.no_grad()
def gen_answer(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
    )
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def fewshot_eval_ehr(
    model,
    tokenizer,
    train_examples,
    eval_examples,
    k_shot: int = 2,
    seed: int = 42,
    max_new_tokens: int = 256,
    candidate_count: int = 4,
    pass_k: int = 3,
    metric_k: int = 10,
):
    rng = random.Random(seed)

    n = len(eval_examples)
    exact_sum = 0.0
    f1_sum = 0.0
    recall_k_sum = 0.0
    prec_k_sum = 0.0
    f1_k_sum = 0.0
    pass_sum = 0.0
    balanced_sum = 0.0

    records = []

    for ex in eval_examples:
        # 抽 few-shot 示範
        k = min(k_shot, len(train_examples))
        shots = rng.sample(train_examples, k) if k > 0 else []

        prompt = build_ehr_fewshot_prompt(shots, ex)
        candidates = [
            extract_assessment(
                gen_answer(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            )
            for _ in range(candidate_count)
        ]
        raw_output = candidates[0]
        pred_assessment = raw_output
        gold_assessment = ex["label"]

        # metrics
        exact = float(
            normalize_text(pred_assessment) == normalize_text(gold_assessment)
        )
        f1 = token_f1(pred_assessment, gold_assessment)
        recall_k, prec_k, f1_k = recall_precision_f1_at_k(
            pred_assessment, gold_assessment, metric_k
        )
        pass_score = pass_at_k(candidates, gold_assessment, pass_k)
        balanced = balanced_accuracy(pred_assessment)

        exact_sum += exact
        f1_sum += f1
        recall_k_sum += recall_k
        prec_k_sum += prec_k
        f1_k_sum += f1_k
        pass_sum += pass_score
        balanced_sum += balanced

        records.append(
            {
                "id": ex.get("id"),
                "input": ex["input"],
                "gold": gold_assessment,
                "pred": pred_assessment,
                "raw_output": raw_output,
                "exact": exact,
                "f1": f1,
                "recall_at_k": recall_k,
                "precision_at_k": prec_k,
                "f1_at_k": f1_k,
                "pass_at_k": pass_score,
                "balanced_accuracy": balanced,
                "candidates": candidates,
            }
        )

    avg_exact = exact_sum / n if n > 0 else 0.0
    avg_f1 = f1_sum / n if n > 0 else 0.0
    avg_recall_k = recall_k_sum / n if n > 0 else 0.0
    avg_precision_k = prec_k_sum / n if n > 0 else 0.0
    avg_f1_k = f1_k_sum / n if n > 0 else 0.0
    avg_pass = pass_sum / n if n > 0 else 0.0
    avg_balanced = balanced_sum / n if n > 0 else 0.0

    return {
        "avg_exact": avg_exact,
        "avg_f1": avg_f1,
        "avg_recall_at_k": avg_recall_k,
        "avg_precision_at_k": avg_precision_k,
        "avg_f1_at_k": avg_f1_k,
        "avg_pass_at_k": avg_pass,
        "avg_balanced_accuracy": avg_balanced,
        "n": n,
        "records": records,
    }


def icd_precision_recall_f1_at_k(pred, gold, k):
    gold_set = set(gold)
    topk = pred[:k]
    topk_set = set(topk)

    tp = len(gold_set & topk_set)
    fp = len(topk_set - gold_set)
    fn = len(gold_set - topk_set)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def main():
    train_examples = load_ehr_eval_examples("train.jsonl", limit=500)
    if len(train_examples) > 0:
        print("\n=== Data Check: First Example ===")
        print(f"ID: {train_examples[0]['id']}")
        print(f"Input (S/O):\n{train_examples[0]['input']}")
        print(f"Label (A):\n{train_examples[0]['label']}")
        print("=================================\n")
    else:
        print("Error: No examples loaded! Please check train.jsonl")
        return
    eval_examples = load_ehr_eval_examples("val.jsonl", limit=50)
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    result = fewshot_eval_ehr(
        model,
        tokenizer,
        train_examples,
        eval_examples,
        k_shot=3,
        seed=0,
        candidate_count=4,
        pass_k=3,
        metric_k=10,
    )

    print(f"Avg exact: {result['avg_exact']:.4f}")
    print(f"Avg token F1: {result['avg_f1']:.4f}")
    print(f"Avg recall@10: {result['avg_recall_at_k']:.4f}")
    print(f"Avg precision@10: {result['avg_precision_at_k']:.4f}")
    print(f"Avg f1@10: {result['avg_f1_at_k']:.4f}")
    print(f"Avg pass@3: {result['avg_pass_at_k']:.4f}")
    print(f"Avg balanced accuracy: {result['avg_balanced_accuracy']:.4f}")
    print(f"N: {result['n']}")


if __name__ == "__main__":
    main()
