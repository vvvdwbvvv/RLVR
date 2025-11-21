import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import Tokenizer
from torch.utils.data import Dataset
from data.util import MIMICNoteParser

SYSTEM_MESSAGE = (
    """You are an expert of clinical diagnosis.
Your job is to read the full EHR note to diagnose the case, predict the ICD code, and outline the assessment through step-by-step reasoning.
DO 
1. Case Summary: Summarize the key patient information., 
2. Clinical Significance: Explain the important findings., 
3. Differential Diagnosis: Consider possible alternatives., 
4. Most Likely Diagnosis: Justify the final diagnosis.
Ensure medical accuracy, no hallucination. base on groud truth in the note., 
5. ICD Code Prediction: Predict a series of appropriate ICD-9 codes based on the diagnosis, sorted by possibility."""
)
USER_TEMPLATE = (
    "Patient note review (ID {note_id}, Encounter {hadm_id}):\n"
    "- ICD-9 code: {icd_code}\n"
    "- Short title: {short_title}\n"
    "- Long title: {long_title}\n\n"
    "{text}\n\n"
    "Summarize key findings, propose the most likely diagnosis (or differential), and explain how the note supports it. "
    "Put your reasoning inside <think> </think> and the final diagnosis/assessment inside <answer> </answer>."
)
RESPONSE_PROMPT = "Let me reason through the case step by step.\n<think>"


class ClinicalJSONLDataset(Dataset):
    def __init__(
        self,
        jsonl_data,
        tokenizer: Tokenizer,
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
        """Build the clinical prompt prefix for the model."""
        user_message = USER_TEMPLATE.format(
            note_id=note_id or "N/A",
            hadm_id=hadm_id or "N/A",
            icd_code=icd_code or "N/A",
            short_title=short_title or "N/A",
            long_title=long_title or "N/A",
            text=text or "No text provided.",
        )
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }


class MIMICDataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
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

        # 🔑 新增 attention_mask：目前每個 sample 都是全長有效，先全部設 1
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "text": tpl["text"],  # optional, debug 用
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
        # 1. 組 user message
        user_message = USER_TEMPLATE.format(
            note_id=note_id or "N/A",
            hadm_id=hadm_id or "N/A",
            icd_code=icd_code or "N/A",
            short_title=short_title or "N/A",
            long_title=long_title or "N/A",
            text=text or "No text provided.",
        )

        # 2. 用 encode_chat_with_response_prompt 組 system + user + RESPONSE_PROMPT
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )

        # 3. tokenize，拿到 ids
        tokens = self.tokenizer.tokenize(prefix)
        input_ids = tokens.ids  # list[int]

        # 4. SFT：目前簡單版 → 全序列當 label
        labels = input_ids.copy()

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
        default="data/notes_icd_long_sample_100.csv",
    )
    parser.add_argument(
        "--jsonl-output",
        type=str,
        default="data/notes_icd_long_sample_100.jsonl",
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
        print(f"Tokenizer provided; Train: {len(train_dataset)}, Val: {len(val_dataset)}")
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
