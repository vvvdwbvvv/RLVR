import json
import random
import re

import torch


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


ASSESSMENT_PATTERN = re.compile(
    r"<assessment>(.*?)</assessment>",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_assessment(text: str) -> str:
    """
    從模型輸出中抓 <assessment>...</assessment>
    如果沒有 tag，就回傳原始文字 (或你可以改成回傳空字串)
    """
    match = ASSESSMENT_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def build_ehr_fewshot_prompt(shots, ex, max_shots=None):
    """
    shots: list of {input, label}
    ex:    單一 eval example
    max_shots: 可用來 hard clip shots 數量 (通常前面就控制了)
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
):
    rng = random.Random(seed)

    n = len(eval_examples)
    exact_sum = 0.0
    f1_sum = 0.0

    records = []

    for ex in eval_examples:
        # 抽 few-shot 示範
        k = min(k_shot, len(train_examples))
        shots = rng.sample(train_examples, k) if k > 0 else []

        prompt = build_ehr_fewshot_prompt(shots, ex)
        raw_output = gen_answer(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        pred_assessment = extract_assessment(raw_output)
        gold_assessment = ex["label"]

        # metrics
        exact = float(
            normalize_text(pred_assessment) == normalize_text(gold_assessment)
        )
        f1 = token_f1(pred_assessment, gold_assessment)

        exact_sum += exact
        f1_sum += f1

        records.append(
            {
                "id": ex.get("id"),
                "input": ex["input"],
                "gold": gold_assessment,
                "pred": pred_assessment,
                "raw_output": raw_output,
                "exact": exact,
                "f1": f1,
            }
        )

    avg_exact = exact_sum / n if n > 0 else 0.0
    avg_f1 = f1_sum / n if n > 0 else 0.0

    return {
        "avg_exact": avg_exact,
        "avg_f1": avg_f1,
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


train_examples = load_ehr_eval_examples("data/ehr_structured.jsonl", limit=100)
eval_examples = load_ehr_eval_examples("data/ehr_structured.jsonl", limit=50)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# 3. 執行 few-shot eval
result = fewshot_eval_ehr(
    model,
    tokenizer,
    train_examples,
    eval_examples,
    k_shot=3,
    seed=0,
)

print("Avg exact:", result["avg_exact"])
print("Avg token F1:", result["avg_f1"])
print("N:", result["n"])

if __name__ == "__main__":
    main()
