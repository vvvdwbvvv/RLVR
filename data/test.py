import re

from dataset import PROMPT

test_data = {
    "note_id": "12345",
    "hadm_id": "67890",
    "icd_code": "2550.00",
    "short_title": "Diabetes mellitus without mention of complication",
    "long_title": "Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled",
    "text": "Patient is a 45-year-old male with a 2-year history of polyuria and polydipsia. Blood glucose 450 mg/dL.",
}

# 使用你的PROMPT模板
formatted_prompt = PROMPT.format(**test_data)

# 檢查是否有遺漏的佔位符

missing_vars = re.findall(r"\{(\w+)\}", formatted_prompt)
if missing_vars:
    print(f"警告：以下變數未被替換：{missing_vars}")
else:
    print("✓ 所有參數已正確傳入")

# 檢查最終prompt長度
print(formatted_prompt)
print(f"Prompt總長度：{len(formatted_prompt)} 字元")
