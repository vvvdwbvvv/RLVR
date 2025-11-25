import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.reward import format_reward_func, reward_len

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


DEFAULT_CONFIG_PATH = Path(__file__).parent / "grpo_config.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    return config or {}


def build_prompt(note_id: str, hadm_id: str, text: str) -> str:
    prompt = PROMPT
    prompt = prompt.replace("{note_id}", note_id or "N/A")
    prompt = prompt.replace("{hadm_id}", hadm_id or "N/A")
    prompt = prompt.replace("{text}", text or "No text provided.")
    return prompt


def load_prompts_from_jsonl(path: str, limit: int | None = None) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            note_id = str(obj.get("id", obj.get("note_id", "N/A")))
            hadm_id = str(obj.get("HADM_ID", obj.get("hadm_id", "N/A")))
            text = obj.get("text", "")
            prompt = build_prompt(note_id=note_id, hadm_id=hadm_id, text=text)
            records.append({"prompt": prompt})
            if limit is not None and len(records) >= limit:
                break
    return Dataset.from_list(records)


def reward_func(
    prompts=None,
    completions=None,
    completion_ids=None,
    trainer_state=None,
    **kwargs,
) -> list[float]:
    if completions is None:
        return []

    if completions and isinstance(completions[0], str):
        wrapped = [[{"content": s}] for s in completions]
    else:
        wrapped = completions

    fmt_r = format_reward_func(wrapped)
    len_r = reward_len(wrapped, ideal_length=512)
    w_fmt = 0.7
    w_len = 0.3
    rewards = []
    for fr, lr in zip(fmt_r, len_r):
        raw = w_fmt * fr + w_len * lr
        rewards.append(max(-1.0, min(1.0, raw)))
    return rewards


wandb.init(
    project="SmolLM2-Clinical-R1", config={"dataset": "MIMICIII", "type": "RLVR-GRPO"}
)


def main():
    parser = argparse.ArgumentParser(
        description="Run GRPO training using a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config for GRPO training.",
    )
    args = parser.parse_args()
    wandb.config.update(args)

    run_config = load_config(args.config)
    model_config = run_config.get("model", {})
    data_config = run_config.get("data", {})
    training_config = run_config.get("training", {})

    # model_id = model_config.get("id", "HuggingFaceTB/SmolLM2-360M-Instruct")
    data_path = data_config.get("jsonl_path", "data/notes_icd_long_sample_100.jsonl")
    output_dir = data_config.get("output_dir", "outputs/grpo-clinical")

    dataset = load_prompts_from_jsonl(data_path)

    config = GRPOConfig(
        # model_id=model_id,
        output_dir=str(output_dir),
        learning_rate=float(training_config.get("learning_rate", 2e-5)),
        max_steps=int(training_config.get("max_steps", 100)),
        per_device_train_batch_size=int(
            training_config.get("per_device_train_batch_size", 1)
        ),
        gradient_accumulation_steps=int(
            training_config.get("gradient_accumulation_steps", 1)
        ),
        max_prompt_length=int(training_config.get("max_prompt_length", 1024)),
        max_completion_length=int(training_config.get("max_completion_length", 512)),
        num_generations=int(training_config.get("num_generations", 4)),
        generation_batch_size=training_config.get("generation_batch_size", 4),
    )

    trainer = GRPOTrainer(
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        args=config,
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
