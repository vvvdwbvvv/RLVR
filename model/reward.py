import json
import re


def reward_len(
    completions, ideal_length: int = 512, max_penalty: float = 1.0, **kwargs
) -> list[float]:
    """
    以字元長度接近 ideal_length 為目標，回傳 [-max_penalty, 0] 區間的 reward。
    """
    rewards = []
    for comp in completions:
        text = comp[-1]["content"]
        L = len(text)
        # 偏離比例，超過 2 * ideal_length 之後當作最差
        deviation = abs(L - ideal_length)
        deviation_ratio = min(deviation / ideal_length, 1.0)  # in [0, 1]
        reward = -max_penalty * deviation_ratio  # in [-max_penalty, 0]
        rewards.append(reward)
    return rewards


STRICT_PATTERN = re.compile(
    r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$",
    flags=re.DOTALL,
)

SOFT_PATTERN = re.compile(
    r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>",
    flags=re.DOTALL,
)


def format_reward_func(
    completions, strict_weight: float = 1.0, soft_weight: float = 0.5, **kwargs
) -> list[float]:
    """
    分級格式 reward：
    - strict match: +strict_weight
    - soft match only: +soft_weight
    - else: 0
    """
    rewards = []
    for comp in completions:
        text = comp[-1]["content"]
        if STRICT_PATTERN.match(text):
            rewards.append(strict_weight)
        elif SOFT_PATTERN.search(text):
            rewards.append(soft_weight)
        else:
            rewards.append(0.0)
    return rewards


ANSWER_BLOCK_PATTERN = re.compile(
    r"<answer>\s*(\{.*\})\s*</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_icds_from_response(text: str) -> list[str]:
    m = ANSWER_BLOCK_PATTERN.search(text)
    if not m:
        return []
    json_str = m.group(1)
    try:
        obj = json.loads(json_str)
    except Exception:
        return []
    codes = []
    for item in obj.get("icd9_predictions", []):
        code = str(item.get("code", "")).strip().upper()
        if code:
            codes.append(code)
    # 去重保順序
    seen = set()
    ordered = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def icd_coverage_reward(
    completions,
    k: int = 5,
    lambda_neg: float = 0.7,
    lambda_len: float = 0.3,
    **kwargs,
) -> list[float]:
    """
    ICD coverage reward：
    - 主軸：recall@k
    - penalty1：hit hard negatives 的比例
    - penalty2：預測 ICD 過多
    最後回傳 [-1, 1] 區間的分數。
    """
    metas = kwargs.get("metas", None)
    if metas is None:
        raise ValueError(
            "icd_coverage_reward 需要 metas (positive_icds / negative_icds)"
        )

    rewards = []

    for comp, meta in zip(completions, metas):
        text = comp[-1]["content"]
        pred_icds = extract_icds_from_response(text)

        pos_icds = meta.get("positive_icds", [])
        neg_icds = meta.get("negative_icds", [])

        # ---- coverage: recall@k ----
        pred_k = pred_icds[:k]
        pred_set = set(pred_k)
        pos_set = set(pos_icds)

        tp = len(pred_set & pos_set)
        fn = len(pos_set - pred_set)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # in [0,1]

        coverage_reward = recall

        # ---- penalty1: 命中 hard negatives ----
        neg_set = set(neg_icds)
        if len(neg_set) > 0:
            hit_neg_ratio = len(pred_set & neg_set) / len(neg_set)  # in [0,1]
        else:
            hit_neg_ratio = 0.0
        penalty_neg = lambda_neg * hit_neg_ratio  # in [0, lambda_neg]

        # ---- penalty2: 預測 ICD 數量過多 ----
        pos_n = max(len(pos_icds), 1)
        pred_n = len(pred_icds)
        if pred_n <= 2 * pos_n:
            over_ratio = 0.0
        else:
            over_ratio = min((pred_n - 2 * pos_n) / pos_n, 1.0)  # in [0,1]
        penalty_len = lambda_len * over_ratio  # in [0, lambda_len]

        reward = coverage_reward - penalty_neg - penalty_len
        reward = max(-1.0, min(1.0, reward))
        rewards.append(reward)

    return rewards


def combined_rlvr_reward(
    completions,
    **kwargs,
) -> list[float]:
    """
    把多個 reward component 加權合併成單一 reward。
    kwargs 裡至少要包含 metas 給 ICD coverage 用。
    """

    # 個別 reward
    icd_r = icd_coverage_reward(completions, **kwargs)  # [-1,1]
    fmt_r = format_reward_func(completions, **kwargs)  # [0,1] or [0,strict_weight]
    len_r = reward_len(completions, ideal_length=512, **kwargs)  # 比如 [-1,0]

    # 你自己決定的權重（示意）
    W_ICD = 0.7
    W_FMT = 0.2
    W_LEN = 0.1

    rewards = []
    for r_icd, r_fmt, r_len in zip(icd_r, fmt_r, len_r):
        # 注意：r_len 如果是 [-1,0]，會變成惩罰項；如果想當 [0,1]，就改前面實作
        raw = W_ICD * r_icd + W_FMT * r_fmt + W_LEN * r_len
        # 再 clip 一次確保數值乾淨
        final = max(-1.0, min(1.0, raw))
        rewards.append(final)

    return rewards
