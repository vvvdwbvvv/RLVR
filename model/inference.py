import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert clinical diagnostician.
Your job is to read the full EHR note, infer the most likely diagnoses, and predict appropriate ICD-9 codes.
You MUST:
1. Case Summary: Summarize the key patient information (demographics, chief complaint, relevant history).
2. Clinical Significance: Highlight the most important positive and negative findings and explain why they matter.
3. Differential Diagnosis: List reasonable alternative diagnoses and briefly justify each.
4. Most Likely Diagnosis: Clearly state and justify the final primary diagnosis.
5. ICD-9 Code Prediction: Predict a ranked list of ICD-9 codes consistent with your final diagnosis, from most to least likely.
Use ONLY the information contained in the EHR note. Do NOT hallucinate findings or diagnoses that are not supported by the note.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Patient note review:
Encounter ID: {hadm_id}
Note ID: {note_id}

EHR note text:
{text}

Task:
Summarize key findings, propose differential diagnoses, select the most likely diagnosis, and predict appropriate ICD-9 codes with justification.

Requirements:
1. Show your entire clinical reasoning process in ONE single <think>...</think> block.
   - Do NOT open or close the <think> tag more than once.
2. Your final output MUST be in valid JSON format, wrapped inside <answer>...</answer> tags.
3. The JSON MUST have the following structure (no extra keys, no trailing commas):

<think>
[entire reasoning process here]
</think>
<answer>
{
  "case_summary": "string, brief summary of patient and key context.",
  "clinical_significance": "string, important positive/negative findings and why they matter.",
  "differential_diagnosis": [
    "string, diagnosis A",
    "string, diagnosis B"
  ],
  "most_likely_diagnosis": "string, final primary diagnosis.",
  "icd9_predictions": [
    {
      "code": "string, ICD-9 code (e.g., '250.00')",
      "description": "string, short clinical description of the code.",
      "confidence": 0.0,
      "rationale": "string, specific evidence from the note that supports this code."
    }
  ],
  "primary_icd9_code": "string, ICD-9 code from icd9_predictions that you consider primary."
}
</answer>

Rules:
- Use double quotes for all JSON keys and string values.
- Do NOT include comments or explanations outside the JSON.
- Do NOT output anything after the </answer> tag.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Let me reason through the case step by step.
<think>"""


def build_prompt(note_id: str, hadm_id: str, text: str) -> str:
    """Fill the inference prompt without disturbing JSON braces."""
    prompt = PROMPT
    prompt = prompt.replace("{note_id}", note_id or "N/A")
    prompt = prompt.replace("{hadm_id}", hadm_id or "N/A")
    prompt = prompt.replace("{text}", text or "No text provided.")
    return prompt


# 設定參數
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
INPUT_JSONL = "data/notes_icd_long_sample_500.jsonl"  # 來源資料
OUTPUT_JSONL = "data/inference_results_500.jsonl"  # 輸出結果
MAX_NEW_TOKENS = 512


def main():
    # 1. 設定裝置 (Mac 使用 MPS，Linux 使用 CUDA)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # 2. 載入模型與 Tokenizer
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto" if device != "mps" else None,
    )
    if device == "mps":
        model.to(device)

    model.eval()

    # 3. 讀取資料
    if not Path(INPUT_JSONL).exists():
        print(f"Error: Input file {INPUT_JSONL} not found.")
        return

    print(f"Reading from {INPUT_JSONL}...")

    with open(INPUT_JSONL, "r") as f:
        lines = f.readlines()

    # 4. 進行推論
    print(f"Starting inference on {len(lines)} examples...")

    with open(OUTPUT_JSONL, "w") as f_out:
        for line in tqdm(lines):
            data = json.loads(line)

            # 構建推論 Prompt（與訓練風格一致）
            note_id = data.get("note_id") or data.get("id") or "N/A"
            hadm_id = data.get("hadm_id") or data.get("HADM_ID") or "N/A"
            text = data.get("text", "")

            full_prompt = build_prompt(note_id=note_id, hadm_id=hadm_id, text=text)

            # Tokenize
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,  # 取樣模式
                    temperature=0.7,  # 溫度
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            response = (
                generated_text[len(full_prompt) :]
                if generated_text.startswith(full_prompt)
                else generated_text
            )

            result_entry = {
                "note_id": data.get("note_id"),
                "prompt": full_prompt,
                "generated_response": response,
                "full_text": generated_text,
            }

            f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"✓ Inference complete! Results saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
