import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from data.util import MIMICNoteParser
from model.tokenizer import Tokenizer

PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert of clinical diagnosis.
Your job is to read the full EHR note, understand why the provided ICD-9 code is appropriate (or partially appropriate) based on the documented evidence, and outline a structured clinical assessment through step-by-step reasoning.

You MUST:
1. Case Summary: Summarize the key patient information (demographics, chief complaint, relevant history).
2. Clinical Significance: Explain the most important positive and negative findings and why they matter in this context.
3. Differential Diagnosis: List reasonable alternative diagnoses based on the presentation and briefly justify each.
4. Most Likely Diagnosis: Clearly state and justify the final primary diagnosis derived from your clinical reasoning.
5. ICD-9 Code Assessment:
   - Evaluate the appropriateness of the provided "ground truth" ICD-9 code against the evidence in the note.
   - Optionally list additional plausible ICD-9 codes if they are clearly supported by the note and distinct from the ground truth.

Ensure medical accuracy and avoid hallucinations. Base all statements strictly on the information provided in the EHR note text.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Patient note review (ID {note_id}, Encounter {hadm_id}):
- ICD-9 code (ground truth): {icd_code}
- Short title: {short_title}
- Long title: {long_title}

EHR note text:
{text}

Task:
Summarize key findings, propose differential diagnoses, determine the most likely clinical diagnosis, and rigorously evaluate why the provided ICD-9 code is appropriate (or why it might be only partially appropriate or incorrect). Structure your entire reasoning process and final assessment in the required format.

Requirements:
1. Show your entire clinical reasoning process in exactly ONE <think>...</think> block before the final JSON output.
   - Do NOT open or close the <think> tag more than once.
2. Your final response MUST be valid JSON wrapped inside <answer>...</answer> tags.
3. The JSON MUST have the following specific structure (no extra keys, no trailing commas, no comments):

<think>
[Put your entire step-by-step clinical reasoning here. This should cover summary, significance, differentials, final diagnosis selection, and a detailed analysis of the ground truth ICD-9 code.]
</think>
<answer>
{{
  "case_summary": "string, brief summary of the patient demographics, chief complaint, and key context.",
  "clinical_significance": "string, highlights of important positive/negative findings and their clinical relevance.",
  "differential_diagnosis": [
    "string, potential diagnosis A",
    "string, potential diagnosis B"
  ],
  "most_likely_diagnosis": "string, the final primary diagnosis derived from clinical reasoning.",
  "ground_truth_evaluation": {{
    "code_analyzed": "{icd_code}",
    "description": "string, short description of the provided code context.",
    "assessment_status": "string, MUST be exactly one of: ['Fully Appropriate', 'Partially Appropriate', 'Potentially Incorrect']",
    "detailed_explanation": "string, detailed rationale supporting the assessment status, linking specific clinical findings from the note to the definition of the code.",
    "evidence_quotes": "string, specific text verbatim QUOTED from the note that directly supports this evaluation. If no direct evidence exists, state 'None'.",
    "is_fully_supported_by_note": true
  }},
  "suggested_alternative_codes": [
    {{
      "code": "string, an alternative plausible ICD-9 code (e.g., '250.00')",
      "description": "string, short clinical description of this alternative code.",
      "rationale": "string, brief explanation of why this alternative code is supported by the note."
    }}
  ]
}}
</answer>

Rules:
- The output within <answer> tags must be strictly valid JSON.
- Use double quotes for all JSON keys and string values.
- "suggested_alternative_codes" can be an empty list [] if no other codes are clearly supported.
- Do NOT include any text, explanations, or comments outside the JSON structure within the <answer> tags.
- Do NOT output anything after the </answer> tag.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Let me reason through the case step by step.
<think>"""


class ClinicalJSONLDataset(Dataset):
    def __init__(
        self,
        jsonl_data,
        tokenizer: Union[Tokenizer, PreTrainedTokenizerBase],
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        self.data = (
            jsonl_data.loc[:-test_size]
            if split == "train"
            else jsonl_data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["id"], item["text"]

    def apply_chat_template(
        self,
        text: str,
        note_id: Optional[str] = None,
        hadm_id: Optional[str] = None,
        icd_code: Optional[str] = None,
        short_title: Optional[str] = None,
        long_title: Optional[str] = None,
    ):
        """Build the clinical prompt string and token ids for SFT."""
        prefix = PROMPT.format(
            note_id=note_id or "N/A",
            hadm_id=hadm_id or "N/A",
            icd_code=icd_code or "N/A",
            short_title=short_title or "N/A",
            long_title=long_title or "N/A",
            text=text or "No text provided.",
        )
        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        else:
            tokens = self.tokenizer.tokenize(prefix)
            input_ids = tokens.ids

        labels = input_ids.copy()
        return {"input_ids": input_ids, "labels": labels, "text": prefix}


class MIMICDataset(Dataset):
    def __init__(
        self,
        tokenizer: Union[Tokenizer, PreTrainedTokenizerBase],
        data_path: Optional[str] = "data/notes_icd_long_sample_100.csv",
        dataframe: Optional[pd.DataFrame] = None,
        jsonl_output: Optional[str] = "data/notes_icd_long_sample_100.jsonl",
        parse_notes: bool = False,
        split: str = "train",
        test_size: int = 20,
    ):
        if dataframe is None:
            if data_path is None:
                raise ValueError("Provide either data_path or dataframe")
            dataframe = self._load_dataframe(data_path)

        if jsonl_output:
            self.write_jsonl(dataframe, jsonl_output, parse_notes=parse_notes)

        cutoff = max(len(dataframe) - test_size, 0)
        if split == "train" and test_size > 0:
            self.data = dataframe.iloc[:cutoff].reset_index(drop=True)
        elif split != "train" and test_size > 0:
            self.data = dataframe.iloc[cutoff:].reset_index(drop=True)
        else:
            self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.column_names = ["input_ids", "labels", "text"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        tpl = self.apply_chat_template(
            text=row["text"],
            note_id=row.get("id"),
            hadm_id=row.get("HADM_ID") if "HADM_ID" in row else None,
            icd_code=row.get("ICD9_CODE") if "ICD9_CODE" in row else None,
            short_title=row.get("SHORT_TITLE") if "SHORT_TITLE" in row else None,
            long_title=row.get("LONG_TITLE") if "LONG_TITLE" in row else None,
        )

        input_ids = torch.tensor(tpl["input_ids"], dtype=torch.long)
        labels = torch.tensor(tpl["labels"], dtype=torch.long)

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "text": tpl["text"],
        }

    def apply_chat_template(
        self,
        text: str,
        note_id: Optional[str] = None,
        hadm_id: Optional[str] = None,
        icd_code: Optional[str] = None,
        short_title: Optional[str] = None,
        long_title: Optional[str] = None,
    ):
        prefix = PROMPT.format(
            note_id=note_id or "N/A",
            hadm_id=hadm_id or "N/A",
            icd_code=icd_code or "N/A",
            short_title=short_title or "N/A",
            long_title=long_title or "N/A",
            text=text or "No text provided.",
        )

        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        else:
            tokens = self.tokenizer.tokenize(prefix)
            input_ids = tokens.ids  # list[int]

        labels = input_ids.copy()

        # tag the position of <answer> tag to ignore the loss of
        answer_start_tag = "<answer>\n"
        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            answer_start_token_ids = self.tokenizer.encode(
                answer_start_tag, add_special_tokens=False
            )
        else:
            answer_tokens = self.tokenizer.tokenize(answer_start_tag)
            answer_start_token_ids = answer_tokens.ids

        answer_start_index = -1
        for i in range(len(input_ids) - len(answer_start_token_ids) + 1):
            if input_ids[i : i + len(answer_start_token_ids)] == answer_start_token_ids:
                answer_start_index = i + len(answer_start_token_ids)
                break

        if answer_start_index != -1:
            for i in range(answer_start_index):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "text": prefix,
        }

    @staticmethod
    def _load_dataframe(data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        df = df.rename(columns={"ROW_ID_x": "id", "TEXT": "text"})
        if "id" not in df.columns or "text" not in df.columns:
            raise ValueError("Expected columns 'ROW_ID_x' and 'TEXT' in CSV")
        df["id"] = df["id"].astype(str)
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        return df[
            [
                "id",
                "text",
                "ICD9_CODE",
                "SHORT_TITLE",
                "LONG_TITLE",
            ]
        ]

    @staticmethod
    def write_jsonl(df: pd.DataFrame, output_path: str, parse_notes: bool = False):
        records = df.to_dict(orient="records")
        with open(output_path, "w") as f:
            for record in records:
                if parse_notes:
                    ehr_record = MIMICNoteParser.parse_row(record)
                    record = json.loads(ehr_record.to_json())
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the MIMIC notes sample into JSONL and optionally instantiate the dataset."
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--source-csv",
        type=str,
        default="data/notes_icd_long_sample_500.csv",
    )
    parser.add_argument(
        "--jsonl-output",
        type=str,
        default="data/notes_icd_long_sample_500.jsonl",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer_path) if args.tokenizer_path else None

    dataframe = MIMICDataset._load_dataframe(args.source_csv)
    MIMICDataset.write_jsonl(dataframe, args.jsonl_output, parse_notes=True)

    total = len(dataframe)
    train_size = int(args.train_ratio * total)
    val_size = total - train_size

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]

    train_subset = dataframe.iloc[train_indices].reset_index(drop=True)
    val_subset = dataframe.iloc[val_indices].reset_index(drop=True)

    if tokenizer is not None:
        train_dataset = MIMICDataset(
            tokenizer=tokenizer,
            dataframe=train_subset,
            jsonl_output=None,
            parse_notes=False,
            split="train",
            test_size=0,
        )
        val_dataset = MIMICDataset(
            tokenizer=tokenizer,
            dataframe=val_subset,
            jsonl_output=None,
            parse_notes=False,
            split="val",
            test_size=0,
        )
        print(
            f"Tokenizer provided; Train: {len(train_dataset)}, Val: {len(val_dataset)}"
        )
    else:
        print(
            "Tokenizer path not provided; install/load the pretrained tokenizer and "
            "run this script again with --tokenizer-path /path/to/tokenizer.json if you want to instantiate datasets."
        )
        print(f"Derived splits -> Train: {len(train_subset)}, Val: {len(val_subset)}")

    def save_jsonl(indices, path):
        with open(path, "w") as f:
            for idx in indices:
                record = dataframe.iloc[idx].to_dict()
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

    save_jsonl(train_indices, "train.jsonl")
    save_jsonl(val_indices, "val.jsonl")
